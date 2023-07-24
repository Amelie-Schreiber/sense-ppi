from torch.utils.data import DataLoader
import pytorch_lightning as pl
from itertools import permutations, product
import numpy as np
import pandas as pd
import logging
from ..dataset import PairSequenceData
from ..model import SensePPIModel
from ..utils import *
from ..esm2_model import add_esm_args, compute_embeddings


def predict(params):
    test_data = PairSequenceData(emb_dir=params.output_dir_esm, actions_file=params.pairs,
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

    preds = [pred for batch in trainer.predict(pretrained_model, test_loader) for pred in batch.squeeze().tolist()]
    preds = np.asarray(preds)

    data = pd.read_csv(params.pairs, delimiter='\t', names=["seq1", "seq2"])
    data['preds'] = preds

    return data


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
    parser._action_groups[0].add_argument("model_path", type=str,
                              help="A path to .ckpt file that contains weights to a pretrained model.")
    predict_args.add_argument("--pairs", type=str, default=None,
                              help="A path to a .tsv file with pairs of proteins to test (Optional). If not provided, all-to-all pairs will be generated.")
    predict_args.add_argument("-o", "--output", type=str, default="predictions",
                              help="A path to a file where the predictions will be saved. (.tsv format will be added automatically)")
    predict_args.add_argument("--with_self", action='store_true',
                              help="Include self-interactions in the predictions."
                                   "By default they are not included since they were not part of training but"
                                   "they can be included by setting this flag to True.")
    predict_args.add_argument("-p", "--pred_threshold", type=float, default=0.5,
                              help="Prediction threshold to determine interacting pairs that will be written to a separate file. Range: (0, 1).")

    parser = SensePPIModel.add_model_specific_args(parser)
    remove_argument(parser, "--lr")

    add_esm_args(parser)
    return parser


def main(params):
    logging.info("Device used: ", params.device)

    process_string_fasta(params.fasta_file, min_len=params.min_len, max_len=params.max_len)
    if params.pairs is None:
        generate_pairs(params.fasta_file, 'protein.pairs.tsv', with_self=params.with_self)
        params.pairs = 'protein.pairs.tsv'

    compute_embeddings(params)

    logging.info('Predicting...')
    data = predict(params)

    data.to_csv(params.output + '.tsv', sep='\t', index=False, header=False)

    data_positive = data[data['preds'] >= params.pred_threshold]
    data_positive.to_csv(params.output + '_positive_interactions.tsv', sep='\t', index=False, header=False)


if __name__ == '__main__':
    parser = add_args()
    params = parser.parse_args()

    main(params)
