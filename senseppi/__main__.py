import argparse
import logging
import os
import torch
from .commands import *
from senseppi import __version__
from senseppi.utils import ArgumentParserWithDefaults, block_mps, determine_device


def main():
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParserWithDefaults(
        description="SENSE_PPI: Sequence-based EvolutIoNary ScalE Protein-Protein Interaction prediction",
        usage="senseppi <command> [<args>]",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "-v", "--version", action="version", version="SENSE-PPI v{} ".format(__version__))

    subparsers = parser.add_subparsers(title="The list of SEINE-PPI commands", required=True, dest="cmd")

    modules = {'train': train,
               'predict': predict,
               'create_dataset': create_dataset,
               'test': test,
               'predict_string': predict_string
               }

    for name, module in modules.items():
        sp = subparsers.add_parser(name)
        sp = module.add_args(sp)
        sp.set_defaults(func=module.main)

    params = parser.parse_args()

    if hasattr(params, 'device'):
        if params.device == 'auto':
            params.device = determine_device()

        if params.device == 'gpu':
            torch.set_float32_matmul_precision('high')

        block_mps(params)

        logging.info('Device used: {}'.format(params.device))

    if hasattr(params, 'model_path'):
        if params.model_path is None:
            params.model_path = os.path.join(os.path.dirname(__file__), "default_model", "senseppi.ckpt")

    params.func(params)


if __name__ == "__main__":
    main()
