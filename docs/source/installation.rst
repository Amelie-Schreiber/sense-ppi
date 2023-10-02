Installation
=====

.. _installation:

To use SENSE-PPI, install it using pip:

.. code-block:: bash

   $ pip install senseppi

SENSE-PPI can also be installed from source:

.. code-block:: bash

   $ git clone http://gitlab.lcqb.upmc.fr/Konstvv/SENSE-PPI.git
   $ cd SENSE-PPI
   $ python setup.py build; python setup.py install

SENSE-PPI requires Python 3.9 or later.
Additionally, required packages include:

- numpy
- pandas
- wget
- scipy
- networkx
- torch>=1.12
- matplotlib
- seaborn
- tqdm
- scikit-learn
- pytorch-lightning==1.9.0
- torchmetrics
- biopython
- fair-esm