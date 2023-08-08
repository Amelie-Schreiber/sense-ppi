from Bio import SeqIO
import os
from senseppi import __version__
import torch
import logging


def add_general_args(parser):
    parser.add_argument("-v", "--version", action="version", version="SENSE_PPI v{}".format(__version__))
    parser.add_argument("--min_len", type=int, default=50,
                        help="Minimum length of the protein sequence. The sequences with smaller length will not be "
                             "considered and will be deleted from the fasta file.")
    parser.add_argument("--max_len", type=int, default=800,
                        help="Maximum length of the protein sequence. The sequences with larger length will not be "
                             "considered and will be deleted from the fasta file.")
    parser.add_argument("--device", type=str, default=determine_device(), choices=['cpu', 'gpu', 'mps'],
                        help="Device to used for computations. Options include: cpu, gpu, mps (for MacOS)."
                             "If not selected the device is set by torch automatically.")

    return parser


def determine_device():
    if torch.cuda.is_available():
        device = 'gpu'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
    else:
        device = 'cpu'
    return device


def block_mps(params):
    # WARNING: due to some internal issues of pytorch, the mps backend is temporarily disabled
    if hasattr(params, 'device'):
        if params.device == 'mps':
            logging.warning('WARNING: due to some internal issues of torch, the mps backend is temporarily disabled.'
                            'The cpu backend will be used instead.')
            if torch.cuda.is_available():
                params.device = 'gpu'
            else:
                params.device = 'cpu'


def process_string_fasta(fasta_file, min_len, max_len):
    with open('file.tmp', 'w') as f:
        for record in SeqIO.parse(fasta_file, "fasta"):
            if len(record.seq) < min_len or len(record.seq) > max_len:
                continue
            record.id = record.id.split(' ')[0]
            record.description = ''
            record.name = ''
            SeqIO.write(record, f, "fasta")
    # Rename the temporary file to the original file
    os.remove(fasta_file)
    os.rename('file.tmp', fasta_file)


def get_fasta_ids(fasta_file):
    ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        ids.append(record.id)
    return ids


def remove_argument(parser, arg):
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            opts = group_action.option_strings
            if (opts and opts[0] == arg) or group_action.dest == arg:
                action._group_actions.remove(group_action)
                return
