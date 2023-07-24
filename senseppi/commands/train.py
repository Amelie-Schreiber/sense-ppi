import os
from pathlib import Path
import torch
import pytorch_lightning as pl
import sys
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from ..model import SensePPIModel
from ..dataset import PairSequenceData
from ..utils import *
from ..esm2_model import add_esm_args, compute_embeddings



# training_th.py
def main(params):
    if params.seed is not None:
        pl.seed_everything(params.seed, workers=True)

    dataset = PairSequenceData(emb_dir=params.output_dir_esm, actions_file=params.pairs,
                               max_len=params.max_len, labels=False)

    model = SensePPIModel(params)

    model.load_data(dataset=dataset, valid_size=0.1)

    train_set = model.train_dataloader()
    val_set = model.val_dataloader()

    logger = pl.loggers.TensorBoardLogger("logs", name='SENSE-PPI')

    callbacks = [
        TQDMProgressBar(refresh_rate=250),
        ModelCheckpoint(filename='chkpt_loss_based_{epoch}-{val_loss:.3f}-{val_BinaryF1Score:.3f}', verbose=True,
                        monitor='val_loss', mode='min', save_top_k=1)
    ]

    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=params.devices, num_nodes=params.num_nodes, max_epochs=100,
                         logger=logger, callbacks=callbacks, strategy=params.strategy)

    trainer.fit(model, train_set, val_set)


def esm_check(fasta_file, output_dir, params):
    params.model_location = 'esm2_t36_3B_UR50D'
    params.fasta_file = fasta_file
    params.output_dir = output_dir

    with open(params.fasta_file, 'r') as f:
        seq_ids = [line.strip().split(' ')[0].replace('>', '') for line in f.readlines() if line.startswith('>')]

    if not os.path.exists(params.output_dir):
        print('Computing ESM embeddings...')
        esm2_model.run(params)
    else:
        for seq_id in seq_ids:
            if not os.path.exists(os.path.join(params.output_dir, seq_id + '.pt')):
                print('Computing ESM embeddings...')
                esm2_model.run(params)
                break

def add_args(parser):
    parser = add_general_args(parser)

    predict_args = parser.add_argument_group(title="Training args")

    parser = SensePPIModel.add_model_specific_args(parser)
    remove_argument(parser, "--lr")

    add_esm_args(parser)
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    params = parser.parse_args()

    esm_check(Path(os.path.join('Data', 'Dscript', 'human.fasta')),
                Path(os.path.join('Data', 'Dscript', 'esm_emb_3B_human')),
                params)

    main(params)