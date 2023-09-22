#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Konstantin Volzhenin, Sorbonne University, 2023

import argparse
import pathlib
import torch
import os
import logging
from esm import FastaBatchedDataset, pretrained
from copy import copy
from Bio import SeqIO


def add_esm_args(parent_parser):
    parser = parent_parser.add_argument_group(title="ESM2 model args",
                                              description="ESM2: Extract per-token representations and model "
                                                          "outputs for sequences in a FASTA file. "
                                                          "The representations are saved in --output_dir_esm folder so "
                                                          "they can be reused in multiple runs. In order to reuse the "
                                                          "embeddings, make sure that --output_dir_esm is set to the "
                                                          "correct folder.")
    parser.add_argument(
        "--model_location_esm",
        type=str, default="esm2_t36_3B_UR50D",
        # help="PyTorch model file OR name of pretrained model to download. If not default, "
        #      "the number of encoder_features has to be modified according to the embedding dimensionality. "
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--output_dir_esm",
        type=pathlib.Path, default=pathlib.Path('esm2_embs_3B'),
        help="output directory for extracted representations",
    )

    parser.add_argument("--toks_per_batch_esm", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--repr_layers_esm",
        type=int,
        default=[-1],
        nargs="+",
        # help="layers indices from which to extract representations (0 to num_layers, inclusive)",
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--truncation_seq_length_esm",
        type=int,
        default=1022,
        # help="truncate sequences longer than the given value",
        help=argparse.SUPPRESS
    )


def run(params):
    model, alphabet = pretrained.load_model_and_alphabet(params.model_location_esm)
    model.eval()

    if params.device == 'gpu':
        model = model.cuda()
        print("Transferred the ESM2 model to GPU")
    elif params.device == 'mps':
        model = model.to('mps')
        print("Transferred the ESM2 model to MPS")

    dataset = FastaBatchedDataset.from_file(params.fasta_file)
    batches = dataset.get_batch_indices(params.toks_per_batch_esm, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(params.truncation_seq_length_esm), batch_sampler=batches
    )
    print(f"Read {params.fasta_file} with {len(dataset)} sequences")

    params.output_dir_esm.mkdir(parents=True, exist_ok=True)

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in params.repr_layers_esm)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in params.repr_layers_esm]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if params.device == 'gpu':
                toks = toks.to(device="cuda", non_blocking=True)
            elif params.device == 'mps':
                toks = toks.to(device="mps", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                params.output_file_esm = params.output_dir_esm / f"{label}.pt"
                params.output_file_esm.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(params.truncation_seq_length_esm, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                result["representations"] = {
                    layer: t[i, 1: truncate_len + 1].clone()
                    for layer, t in representations.items()
                }

                torch.save(
                    result,
                    params.output_file_esm,
                )


def compute_embeddings(params):
    # Compute ESM embeddings

    logging.info('Computing ESM embeddings. If all the files already exist in {} folder, '
                 'this step will be skipped.'.format(params.output_dir_esm))

    if not os.path.exists(params.output_dir_esm):
        run(params)
    else:
        with open(params.fasta_file, 'r') as f:
            # dict of only id and sequences from parsing fasta file
            seq_dict = SeqIO.to_dict(SeqIO.parse(f, 'fasta'))
            seq_ids = list(seq_dict.keys())
        for seq_id in seq_ids:
            if os.path.exists(os.path.join(params.output_dir_esm, seq_id + '.pt')):
                seq_dict.pop(seq_id)
        if len(seq_dict) > 0:
            params_esm = copy(params)
            params_esm.fasta_file = 'tmp_for_esm.fasta'
            with open(params_esm.fasta_file, 'w') as f:
                for seq_id in seq_dict.keys():
                    f.write('>' + seq_id + '\n')
                    f.write(str(seq_dict[seq_id].seq) + '\n')
            run(params_esm)
            os.remove(params_esm.fasta_file)
        else:
            logging.info('All ESM embeddings already computed')


if __name__ == "__main__":
    esm_parser = argparse.ArgumentParser()
    add_esm_args(esm_parser)
    args = esm_parser.parse_args()
    run(args)
