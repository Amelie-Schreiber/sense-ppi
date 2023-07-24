import argparse
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
    params.func(params)


if __name__ == "__main__":
    main()
