from setuptools import setup, find_packages
import senseppi

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dscript_data",
    version=senseppi.__version__,
    description="SENSE_PPI: Sequence-based EvolutIoNary ScalE Protein-Protein Interaction prediction",
    author="Konstantin Volzhenin",
    author_email="konstantin.volzhenin@sorbonne-universite.fr",
    url="",
    license="MIT",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "torch>=1.12",
        "matplotlib",
        "tqdm",
        "scikit-learn",
        "pytorch-lightning==1.9.0",
        "torchmetrics",
        "biopython",
        "fair-esm"
    ],
)