from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
import pathlib
import argparse
from ..dataset import PairSequenceData
from ..model import SensePPIModel
from ..utils import *
from ..esm2_model import add_esm_args, compute_embeddings


def test(params):
    eval_data = PairSequenceData(emb_dir=params.output_dir_esm, actions_file=params.pairs_file,
                                 max_len=params.max_len, labels=True)

    pretrained_model = SensePPIModel(params)

    if params.device == 'gpu':
        checkpoint = torch.load(params.model_path)
    elif params.device == 'mps':
        checkpoint = torch.load(params.model_path, map_location=torch.device('mps'))
    else:
        checkpoint = torch.load(params.model_path, map_location=torch.device('cpu'))

    pretrained_model.load_state_dict(checkpoint['state_dict'])

    trainer = pl.Trainer(accelerator=params.device, logger=False)

    eval_loader = DataLoader(dataset=eval_data,
                             batch_size=params.batch_size,
                             num_workers=4)

    return trainer.test(pretrained_model, eval_loader)


def add_args(parser):
    parser = add_general_args(parser)

    test_args = parser.add_argument_group(title="Predict args")
    parser._action_groups[0].add_argument("pairs_file", type=str, default=None,
                                          help="A path to a .tsv file with pairs of proteins to test.")
    parser._action_groups[0].add_argument("fasta_file",
                                          type=pathlib.Path,
                                          help="FASTA file on which to extract the ESM2 "
                                               "representations and then evaluate.",
                                          )
    test_args.add_argument("--model_path", type=str, default=os.path.join(os.path.dirname(__file__), "..", "default_model", "senseppi.ckpt"),
                           help="A path to .ckpt file that contains weights to a pretrained model. If "
                                "None, the senseppi trained version is used.")
    test_args.add_argument("-o", "--output", type=str, default="test_metrics",
                           help="A path to a file where the test metrics will be saved. "
                                "(.tsv format will be added automatically)")
    test_args.add_argument("--crop_data_to_model_lims", action="store_true",
                           help="If set, the data will be cropped to the limits of the model: "
                                "evaluations will be done only for proteins >50aa and <800aa. WARNING: "
                                "this will modify the original input files.")

    parser = SensePPIModel.add_model_specific_args(parser)
    remove_argument(parser, "--lr")

    add_esm_args(parser)
    return parser


def main(params):
    if params.crop_data_to_model_lims:
        process_string_fasta(params.fasta_file, min_len=params.min_len, max_len=params.max_len)

        data = pd.read_csv(params.pairs_file, delimiter='\t', names=["seq1", "seq2", "label"])
        data = data[data['seq1'].isin(get_fasta_ids(params.fasta_file))]
        data = data[data['seq2'].isin(get_fasta_ids(params.fasta_file))]
        data.to_csv(params.pairs_file, sep='\t', index=False, header=False)

    compute_embeddings(params)

    logging.info('Evaluating...')
    test_metrics = test(params)[0]

    test_metrics_df = pd.DataFrame.from_dict(test_metrics, orient='index')
    test_metrics_df.to_csv(params.output + '.tsv', sep='\t', header=False)


if __name__ == '__main__':
    test_parser = argparse.ArgumentParser()
    test_parser = add_args(test_parser)
    test_params = test_parser.parse_args()

    main(test_params)
