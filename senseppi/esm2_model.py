#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import pathlib
import torch
import os
import logging
from esm import FastaBatchedDataset, pretrained


def add_esm_args(parent_parser):
    parser = parent_parser.add_argument_group(title="ESM2 model args",
                                              description="ESM2: Extract per-token representations and model "
                                                          "outputs for sequences in a FASTA file. "
                                                          "If you would like to use the basic version of SENSE-PPI "
                                                          "do no edit the default values of the arguments below. ")
    parser.add_argument(
        "--model_location_esm",
        type=str, default="esm2_t36_3B_UR50D",
        help="PyTorch model file OR name of pretrained model to download. If not default, "
             "the number of encoder_features has to be modified according to the embedding dimensionality. "
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
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--truncation_seq_length_esm",
        type=int,
        default=1022,
        help="truncate sequences longer than the given value",
    )


def run(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location_esm)
    model.eval()

    if args.device == 'gpu':
        model = model.cuda()
        print("Transferred the ESM2 model to GPU")
    elif args.device == 'mps':
        model = model.to('mps')
        print("Transferred the ESM2 model to MPS")

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch_esm, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length_esm), batch_sampler=batches
    )
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    args.output_dir_esm.mkdir(parents=True, exist_ok=True)

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers_esm)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers_esm]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if args.device == 'gpu':
                toks = toks.to(device="cuda", non_blocking=True)
            elif args.device == 'mps':
                toks = toks.to(device="mps", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                args.output_file_esm = args.output_dir_esm / f"{label}.pt"
                args.output_file_esm.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(args.truncation_seq_length_esm, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                result["representations"] = {
                    layer: t[i, 1 : truncate_len + 1].clone()
                    for layer, t in representations.items()
                }

                torch.save(
                    result,
                    args.output_file_esm,
                )


def compute_embeddings(params):
    # Compute ESM embeddings

    logging.info('Computing ESM embeddings if they are not already computed. '
                 'If all the files alreaady exist in {} folder, this step will be skipped.'.format(params.output_dir_esm))

    if not os.path.exists(params.output_dir_esm):
        run(params)
    else:
        with open(params.fasta_file, 'r') as f:
            seq_ids = [line.strip().split(' ')[0].replace('>', '') for line in f.readlines() if line.startswith('>')]
        for seq_id in seq_ids:
            if not os.path.exists(os.path.join(params.output_dir_esm, seq_id + '.pt')):
                run(params)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_esm_args(parser)
    args = parser.parse_args()
    run(args)
