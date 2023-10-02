Usage
=====

.. _usage:

Quick start
------------

SENSE-PPI can be used to predict pairwise physical interactions between proteins. The simplest input is single a FASTA file with protein sequences.
The output is a .tsv file with all predictions as well as a secondary .tsv file that contains only positive interactions. By default, the predictions are made in "all vs all" manner: all possible protein pairs from the input file are considered.

In order to copmute the predictions for all possible pairs from FASTA file, the following command can be used:

.. code-block:: bash

    $ senseppi predict proteins.fasta

By default, if no model is provided, the pre-trained model on human PPIs is used.

List of commands
------------

There are 5 commands available in the package:

- `train`: trains SENSE-PPI on a given dataset
- `test`: computes test metrics (AUROC, AUPRC, F1, MCC, Presicion, Recall, Accuracy) on a given dataset
- `predict`: predicts interactions for a given dataset
- `predict_string`: predicts interactions for a given dataset using STRING database: the interactions are taken from the STRING database (based on seed proteins). Predictions are compared with the STRING database. Optionally, the graphs can be constructed.
- `create_dataset`: creates a dataset from the STRING database based on the taxonomic ID of the organism.


Predict
------------

.. code-block:: bash

    usage: senseppi <command> [<args>] predict [-h] [-v] [--min_len MIN_LEN] [--max_len MAX_LEN] [--device {cpu,gpu,mps,auto}] [--model_path MODEL_PATH] [--pairs_file PAIRS_FILE]
                                           [-o OUTPUT] [--with_self] [-p PRED_THRESHOLD] [--batch_size BATCH_SIZE] [--output_dir_esm OUTPUT_DIR_ESM]
                                           [--toks_per_batch_esm TOKS_PER_BATCH_ESM]
                                           fasta_file

    positional arguments:
      fasta_file            FASTA file on which to extract the ESM2 representations and then test.

    options:
      -h, --help            show this help message and exit
      -v, --version         show program's version number and exit
      --min_len MIN_LEN     Minimum length of the protein sequence. The sequences with smaller length will not be considered and will be deleted from the fasta file. (Default: 50)
      --max_len MAX_LEN     Maximum length of the protein sequence. The sequences with larger length will not be considered and will be deleted from the fasta file. (Default: 800)
      --device {cpu,gpu,mps,auto}
                            Device to use for computations. Options include: cpu, gpu, mps (for MacOS), and auto.If not selected the device is set by torch automatically. WARNING: mps
                            is temporarily disabled, if it is chosen, cpu will be used instead. (Default: auto)

    Predict args:
      --model_path MODEL_PATH
                            A path to .ckpt file that contains weights to a pretrained model. If None, the preinstalled senseppi.ckpt trained version is used. (Trained on human PPIs)
                            (Default: None)
      --pairs_file PAIRS_FILE
                            A path to a .tsv file with pairs of proteins to test (Optional). If not provided, all-to-all pairs will be generated. (Default: None)
      -o OUTPUT, --output OUTPUT
                            A path to a file where the predictions will be saved. (.tsv format will be added automatically) (Default: predictions)
      --with_self           Include self-interactions in the predictions.By default they are not included since they were not part of training but they can be included by setting this
                            flag to True.
      -p PRED_THRESHOLD, --pred_threshold PRED_THRESHOLD
                            Prediction threshold to determine interacting pairs that will be written to a separate file. Range: (0, 1). (Default: 0.5)

    Args_model:
      --batch_size BATCH_SIZE
                            Batch size for training/testing. (Default: 32)

    ESM2 model args:
      ESM2: Extract per-token representations and model outputs for sequences in a FASTA file. The representations are saved in --output_dir_esm folder so they can be reused in
      multiple runs. In order to reuse the embeddings, make sure that --output_dir_esm is set to the correct folder.

      --output_dir_esm OUTPUT_DIR_ESM
                            output directory for extracted representations (Default: esm2_embs_3B)
      --toks_per_batch_esm TOKS_PER_BATCH_ESM
                            maximum batch size (Default: 4096)


Test
------------

.. code-block:: bash

    usage: senseppi <command> [<args>] test [-h] [-v] [--min_len MIN_LEN] [--max_len MAX_LEN] [--device {cpu,gpu,mps,auto}] [--model_path MODEL_PATH] [-o OUTPUT]
                                            [--crop_data_to_model_lims] [--batch_size BATCH_SIZE] [--output_dir_esm OUTPUT_DIR_ESM] [--toks_per_batch_esm TOKS_PER_BATCH_ESM]
                                            pairs_file fasta_file

    positional arguments:
      pairs_file            A path to a .tsv file with pairs of proteins to test.
      fasta_file            FASTA file on which to extract the ESM2 representations and then evaluate.

    options:
      -h, --help            show this help message and exit
      -v, --version         show program's version number and exit
      --min_len MIN_LEN     Minimum length of the protein sequence. The sequences with smaller length will not be considered and will be deleted from the fasta file. (Default: 50)
      --max_len MAX_LEN     Maximum length of the protein sequence. The sequences with larger length will not be considered and will be deleted from the fasta file. (Default: 800)
      --device {cpu,gpu,mps,auto}
                            Device to use for computations. Options include: cpu, gpu, mps (for MacOS), and auto.If not selected the device is set by torch automatically. WARNING: mps
                            is temporarily disabled, if it is chosen, cpu will be used instead. (Default: auto)

    Predict args:
      --model_path MODEL_PATH
                            A path to .ckpt file that contains weights to a pretrained model. If None, the preinstalled senseppi.ckpt trained version is used. (Trained on human PPIs)
                            (Default: None)
      -o OUTPUT, --output OUTPUT
                            A path to a file where the test metrics will be saved. (.tsv format will be added automatically) (Default: test_metrics)
      --crop_data_to_model_lims
                            If set, the data will be cropped to the limits of the model: evaluations will be done only for proteins >50aa and <800aa. WARNING: this will modify the
                            original input files.

    Args_model:
      --batch_size BATCH_SIZE
                            Batch size for training/testing. (Default: 32)

    ESM2 model args:
      ESM2: Extract per-token representations and model outputs for sequences in a FASTA file. The representations are saved in --output_dir_esm folder so they can be reused in
      multiple runs. In order to reuse the embeddings, make sure that --output_dir_esm is set to the correct folder.

      --output_dir_esm OUTPUT_DIR_ESM
                            output directory for extracted representations (Default: esm2_embs_3B)
      --toks_per_batch_esm TOKS_PER_BATCH_ESM
                            maximum batch size (Default: 4096)


Train
------------

.. code-block:: bash

    usage: senseppi <command> [<args>] train [-h] [-v] [--min_len MIN_LEN] [--max_len MAX_LEN] [--device {cpu,gpu,mps,auto}] [--valid_size VALID_SIZE] [--seed SEED]
                                             [--num_epochs NUM_EPOCHS] [--num_devices NUM_DEVICES] [--num_nodes NUM_NODES] [--early_stop EARLY_STOP] [--lr LR]
                                             [--batch_size BATCH_SIZE] [--output_dir_esm OUTPUT_DIR_ESM] [--toks_per_batch_esm TOKS_PER_BATCH_ESM]
                                             pairs_file fasta_file

    positional arguments:
      pairs_file            A path to a .tsv file containing training pairs. Required format: 3 tab separated columns: first protein, second protein (protein names have to be present
                            in fasta_file), label (0 or 1).
      fasta_file            FASTA file on which to extract the ESM2 representations and then train.

    options:
      -h, --help            show this help message and exit
      -v, --version         show program's version number and exit
      --min_len MIN_LEN     Minimum length of the protein sequence. The sequences with smaller length will not be considered and will be deleted from the fasta file. (Default: 50)
      --max_len MAX_LEN     Maximum length of the protein sequence. The sequences with larger length will not be considered and will be deleted from the fasta file. (Default: 800)
      --device {cpu,gpu,mps,auto}
                            Device to use for computations. Options include: cpu, gpu, mps (for MacOS), and auto.If not selected the device is set by torch automatically. WARNING: mps
                            is temporarily disabled, if it is chosen, cpu will be used instead. (Default: auto)

    Training args:
      Arguments for training the model.

      --valid_size VALID_SIZE
                            Fraction of the training data to use for validation. (Default: 0.1)
      --seed SEED           Global training seed. (Default: None)
      --num_epochs NUM_EPOCHS
                            Number of training epochs. (Default: 100)
      --num_devices NUM_DEVICES
                            Number of devices to use for multi GPU training. (Default: 1)
      --num_nodes NUM_NODES
                            Number of nodes to use for training on a cluster. (Default: 1)
      --early_stop EARLY_STOP
                            Number of epochs to wait before stopping the training (tracking is done with validation loss). By default, the is no early stopping. (Default: None)

    Args_model:
      --lr LR               Learning rate for training. Cosine warmup will be applied. (Default: 0.0001)
      --batch_size BATCH_SIZE
                            Batch size for training/testing. (Default: 32)

    ESM2 model args:
      ESM2: Extract per-token representations and model outputs for sequences in a FASTA file. The representations are saved in --output_dir_esm folder so they can be reused in
      multiple runs. In order to reuse the embeddings, make sure that --output_dir_esm is set to the correct folder.

      --output_dir_esm OUTPUT_DIR_ESM
                            output directory for extracted representations (Default: esm2_embs_3B)
      --toks_per_batch_esm TOKS_PER_BATCH_ESM
                            maximum batch size (Default: 4096)


Predict_string
------------

.. code-block:: bash

    usage: senseppi <command> [<args>] predict_string [-h] [-v] [--min_len MIN_LEN] [--max_len MAX_LEN] [--device {cpu,gpu,mps,auto}] [--model_path MODEL_PATH] [-s SPECIES] [-n NODES]
                                                      [-r SCORE] [-p PRED_THRESHOLD] [--graphs] [-o OUTPUT] [--network_type {physical,functional}]
                                                      [--delete_proteins DELETE_PROTEINS [DELETE_PROTEINS ...]] [--batch_size BATCH_SIZE] [--output_dir_esm OUTPUT_DIR_ESM]
                                                      [--toks_per_batch_esm TOKS_PER_BATCH_ESM]
                                                      genes [genes ...]

    positional arguments:
      genes                 Name of gene to fetch from STRING database. Several names can be typed (separated by whitespaces).

    options:
      -h, --help            show this help message and exit
      -v, --version         show program's version number and exit
      --min_len MIN_LEN     Minimum length of the protein sequence. The sequences with smaller length will not be considered and will be deleted from the fasta file. (Default: 50)
      --max_len MAX_LEN     Maximum length of the protein sequence. The sequences with larger length will not be considered and will be deleted from the fasta file. (Default: 800)
      --device {cpu,gpu,mps,auto}
                            Device to use for computations. Options include: cpu, gpu, mps (for MacOS), and auto.If not selected the device is set by torch automatically. WARNING: mps
                            is temporarily disabled, if it is chosen, cpu will be used instead. (Default: auto)

    General options:
      --model_path MODEL_PATH
                            A path to .ckpt file that contains weights to a pretrained model. If None, the preinstalled senseppi.ckpt trained version is used. (Trained on human PPIs)
                            (Default: None)
      -s SPECIES, --species SPECIES
                            Species from STRING database. Default: H. Sapiens (Default: 9606)
      -n NODES, --nodes NODES
                            Number of nodes to fetch from STRING database. (Default: 10)
      -r SCORE, --score SCORE
                            Score threshold for STRING connections. Range: (0, 1000). (Default: 0)
      -p PRED_THRESHOLD, --pred_threshold PRED_THRESHOLD
                            Prediction threshold. Range: (0, 1000). (Default: 500)
      --graphs              Enables plotting the heatmap and a network graph.
      -o OUTPUT, --output OUTPUT
                            A path to a file where the predictions will be saved. (.tsv format will be added automatically) (Default: preds_from_string)
      --network_type {physical,functional}
                            Network type to fetch from STRING database. (Default: physical)
      --delete_proteins DELETE_PROTEINS [DELETE_PROTEINS ...]
                            List of proteins to delete from the graph. Several names can be specified separated by whitespaces. (Default: None)

    Args_model:
      --batch_size BATCH_SIZE
                            Batch size for training/testing. (Default: 32)

    ESM2 model args:
      ESM2: Extract per-token representations and model outputs for sequences in a FASTA file. The representations are saved in --output_dir_esm folder so they can be reused in
      multiple runs. In order to reuse the embeddings, make sure that --output_dir_esm is set to the correct folder.

      --output_dir_esm OUTPUT_DIR_ESM
                            output directory for extracted representations (Default: esm2_embs_3B)
      --toks_per_batch_esm TOKS_PER_BATCH_ESM
                            maximum batch size (Default: 4096)


Create_dataset
------------

.. code-block:: bash

    usage: senseppi <command> [<args>] create_dataset [-h] [--interactions INTERACTIONS] [--sequences SEQUENCES] [--not_remove_long_short_proteins] [--min_length MIN_LENGTH]
                                                      [--max_length MAX_LENGTH] [--max_positive_pairs MAX_POSITIVE_PAIRS] [--combined_score COMBINED_SCORE]
                                                      [--experimental_score EXPERIMENTAL_SCORE]
                                                      species

    positional arguments:
      species               The Taxon identifier of the organism of interest.

    options:
      -h, --help            show this help message and exit
      --interactions INTERACTIONS
                            The physical links (full) file from STRING for the organism of interest. (Default: None)
      --sequences SEQUENCES
                            The sequences file downloaded from the same page of STRING. For both files see https://string-db.org/cgi/download (Default: None)
      --not_remove_long_short_proteins
                            If specified, does not remove proteins shorter than --min_length and longer than --max_length. By default, long and short proteins are removed.
      --min_length MIN_LENGTH
                            The minimum length of a protein to be included in the dataset. (Default: 50)
      --max_length MAX_LENGTH
                            The maximum length of a protein to be included in the dataset. (Default: 800)
      --max_positive_pairs MAX_POSITIVE_PAIRS
                            The maximum number of positive pairs to be included in the dataset. If None, all pairs are included. If specified, the pairs are selected based on the
                            combined score in STRING. (Default: None)
      --combined_score COMBINED_SCORE
                            The combined score threshold for the pairs extracted from STRING. Ranges from 0 to 1000. (Default: 500)
      --experimental_score EXPERIMENTAL_SCORE
                            The experimental score threshold for the pairs extracted from STRING. Ranges from 0 to 1000. Default is None, which means that the experimental score is
                            not used. (Default: None)
