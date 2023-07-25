import argparse
import logging
from .commands import *
from senseppi import __version__


def main():
    parser = argparse.ArgumentParser(
        description="SENSE_PPI: Sequence-based EvolutIoNary ScalE Protein-Protein Interaction prediction",
        usage="senseppi <command> [<args>]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-v", "--version", action="version", version="SENSE-PPI v{} ".format(__version__))

    subparsers = parser.add_subparsers(title="The list of SEINE-PPI commands:", required=True, dest="cmd")

    modules = {'train': train, 'predict': predict}

    for name, module in modules.items():
        sp = subparsers.add_parser(name)
        sp = module.add_args(sp)
        sp.set_defaults(func=module.main)

    params = parser.parse_args()

    #WARNING: due to some internal issues of torch, the mps backend is temporarily disabled
    if params.device == 'mps':
        logging.warning('WARNING: due to some internal issues of torch, the mps backend is temporarily disabled.'
                        'The cpu backend will be used instead.')
        params.device = 'cpu'

    params.func(params)


if __name__ == "__main__":
    main()