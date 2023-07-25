import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from ..model import SensePPIModel
from ..dataset import PairSequenceData
from ..utils import *
from ..esm2_model import add_esm_args, compute_embeddings


def main(params):
    if params.seed is not None:
        pl.seed_everything(params.seed, workers=True)

    compute_embeddings(params)

    dataset = PairSequenceData(emb_dir=params.output_dir_esm, actions_file=params.pairs_file,
                               max_len=params.max_len, labels=True)

    model = SensePPIModel(params)

    model.load_data(dataset=dataset, valid_size=params.valid_size)
    train_set = model.train_dataloader()
    val_set = model.val_dataloader()

    # logger = pl.loggers.TensorBoardLogger("logs", name='SENSE-PPI')
    logger = None

    callbacks = [
        # TQDMProgressBar(refresh_rate=250),
        ModelCheckpoint(filename='chkpt_loss_based_{epoch}-{val_loss:.3f}-{val_BinaryF1Score:.3f}', verbose=True,
                        monitor='val_loss', mode='min', save_top_k=1)
    ]

    trainer = pl.Trainer(accelerator=params.device, devices=params.num_devices, num_nodes=params.num_nodes,
                         max_epochs=params.num_epochs, logger=logger, callbacks=callbacks)

    trainer.fit(model, train_set, val_set)


def add_args(parser):
    parser = add_general_args(parser)

    train_args = parser.add_argument_group(title="Training args")
    parser._action_groups[0].add_argument("pairs_file", type=str,
                                          help="A path to a .tsv file containing training pairs. "
                                               "Required format: 3 tab separated columns: first protein, "
                                               "second protein (protein names have to be present in fasta_file), "
                                               "label (0 or 1).")
    train_args.add_argument("--valid_size", type=float, default=0.1,
                            help="Fraction of the training data to use for validation.")
    train_args.add_argument("--seed", type=int, default=None, help="Global training seed.")
    train_args.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    train_args.add_argument("--num_devices", type=int, default=1,
                            help="Number of devices to use for multi GPU training.")
    train_args.add_argument("--num_nodes", type=int, default=1,
                            help="Number of nodes to use for training on a cluster.")

    parser = SensePPIModel.add_model_specific_args(parser)

    add_esm_args(parser)
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    params = parser.parse_args()

    main(params)