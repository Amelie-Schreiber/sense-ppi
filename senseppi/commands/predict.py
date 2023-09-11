from torch.utils.data import DataLoader
import pytorch_lightning as pl
from itertools import permutations, product
import numpy as np
import pandas as pd
import pathlib
import argparse
from ..dataset import PairSequenceData
from ..model import SensePPIModel
from ..utils import *
from ..esm2_model import add_esm_args, compute_embeddings


def predict(params):
    test_data = PairSequenceData(emb_dir=params.output_dir_esm, actions_file=params.pairs_file,
                                 max_len=params.max_len, labels=False)

    pretrained_model = SensePPIModel(params)

    if params.device == 'gpu':
        checkpoint = torch.load(params.model_path)
    elif params.device == 'mps':
        checkpoint = torch.load(params.model_path, map_location=torch.device('mps'))
    else:
        checkpoint = torch.load(params.model_path, map_location=torch.device('cpu'))

    pretrained_model.load_state_dict(checkpoint['state_dict'])

    trainer = pl.Trainer(accelerator=params.device, logger=False)

    test_loader = DataLoader(dataset=test_data,
                             batch_size=params.batch_size,
                             num_workers=4)

    preds = trainer.predict(pretrained_model, test_loader)
    preds = [batch.squeeze().tolist() for batch in preds]
    if any(isinstance(i, list) for i in preds):
        preds = [item for batch in preds for item in batch]
    preds = np.asarray(preds)

    return preds


def generate_pairs(fasta_file, output_path, with_self=False):
    ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        ids.append(record.id)

    if with_self:
        all_pairs = [p for p in product(ids, repeat=2)]
    else:
        all_pairs = [p for p in permutations(ids, 2)]

    pairs = []
    for p in all_pairs:
        if (p[1], p[0]) not in pairs and (p[0], p[1]) not in pairs:
            pairs.append(p)

    pairs = pd.DataFrame(pairs, columns=['seq1', 'seq2'])

    pairs.to_csv(output_path, sep='\t', index=False, header=False)


def add_args(parser):
    parser = add_general_args(parser)

    predict_args = parser.add_argument_group(title="Predict args")
    parser._action_groups[0].add_argument("fasta_file", type=pathlib.Path,
                                          help="FASTA file on which to extract the ESM2 representations and then test.",
                                          )
    predict_args.add_argument("--model_path", type=str, default=os.path.join(os.path.dirname(__file__), "..", "default_model", "senseppi.ckpt"),
                                          help="A path to .ckpt file that contains weights to a pretrained model. If "
                                               "None, the senseppi trained version is used.")
    predict_args.add_argument("--pairs_file", type=str, default=None,
                              help="A path to a .tsv file with pairs of proteins to test (Optional). If not provided, "
                                   "all-to-all pairs will be generated.")
    predict_args.add_argument("-o", "--output", type=str, default="predictions",
                              help="A path to a file where the predictions will be saved. "
                                   "(.tsv format will be added automatically)")
    predict_args.add_argument("--with_self", action='store_true',
                              help="Include self-interactions in the predictions."
                                   "By default they are not included since they were not part of training but"
                                   "they can be included by setting this flag to True.")
    predict_args.add_argument("-p", "--pred_threshold", type=float, default=0.5,
                              help="Prediction threshold to determine interacting pairs that "
                                   "will be written to a separate file. Range: (0, 1).")

    parser = SensePPIModel.add_model_specific_args(parser)
    remove_argument(parser, "--lr")

    add_esm_args(parser)

    parser.set_defaults(max_len=None)
    parser.set_defaults(min_len=0)

    return parser


def get_max_len(fasta_file):
    max_len = 0
    for record in SeqIO.parse(fasta_file, "fasta"):
        if len(record.seq) > max_len:
            max_len = len(record.seq)
    return max_len


def get_protein_names(fasta_file):
    names = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        names.append(record.id)
    return set(names)


def main(params):
    tmp_pairs = 'protein.pairs.tsv'

    fasta_max_len = get_max_len(params.fasta_file)
    if params.max_len is None:
        params.max_len = fasta_max_len
    process_string_fasta(params.fasta_file, min_len=params.min_len, max_len=params.max_len)

    if params.pairs_file is None:
        generate_pairs(params.fasta_file, tmp_pairs, with_self=params.with_self)
        params.pairs_file = tmp_pairs
    else:
        if params.max_len < fasta_max_len:
            proteins_in_fasta = get_protein_names(params.fasta_file)

            data_tmp = pd.read_csv(params.pairs_file, delimiter='\t', header=None)
            data_tmp = data_tmp[data_tmp.iloc[:, 0].isin(proteins_in_fasta) &
                                data_tmp.iloc[:, 1].isin(proteins_in_fasta)]
            data_tmp.to_csv(tmp_pairs, sep='\t', index=False, header=False)
            params.pairs_file = tmp_pairs

    compute_embeddings(params)

    logging.info('Predicting...')
    preds = predict(params)

    data = pd.read_csv(params.pairs_file, delimiter='\t', header=None)

    if len(data.columns) == 3:
        data.columns = ['seq1', 'seq2', 'label']
    elif len(data.columns) == 2:
        data.columns = ['seq1', 'seq2']
    else:
        raise ValueError('The tab-separated pairs file must have 2 or 3 columns (without header): '
                         'protein name 1, protein name 2 and label(optional)')
    data['preds'] = preds

    data.to_csv(params.output + '.tsv', sep='\t', index=False, header=True)

    data_positive = data[data['preds'] >= params.pred_threshold]
    data_positive.to_csv(params.output + '_positive_interactions.tsv', sep='\t', index=False, header=True)

    if os.path.isfile(tmp_pairs):
        os.remove(tmp_pairs)


if __name__ == '__main__':
    pred_parser = argparse.ArgumentParser()
    pred_parser = add_args(pred_parser)
    pred_params = pred_parser.parse_args()

    main(pred_params)
