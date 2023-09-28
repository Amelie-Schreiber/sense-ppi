Usage
=====

.. _installation:

Installation
------------

To use SENSE-PPI, first install it using pip:

.. code-block:: console

   (.venv) $ pip install senseppi

Commands
------------

There are 5 commands available in the package:

- `train`: trains SENSE-PPI on a given dataset
- `test`: computes test metrics (AUROC, AUPRC, F1, MCC, Presicion, Recall, Accuracy) on a given dataset
- `predict`: predicts interactions for a given dataset
- `predict_string`: predicts interactions for a given dataset using STRING database: the interactions are taken from the STRING database (based on seed proteins). Predictions are compared with the STRING database. Optionally, the graphs can be constructed.
- `create_dataset`: creates a dataset from the STRING database based on the taxonomic ID of the organism.


