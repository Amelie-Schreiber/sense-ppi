from setuptools import setup, find_packages
import senseppi

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="senseppi",
    version=senseppi.__version__,
    description="SENSE-PPI: Sequence-based EvolutioNary ScalE Protein-Protein Interaction prediction",
    author="Konstantin Volzhenin",
    author_email="konstantin.volzhenin@sorbonne-universite.fr",
    url="",
    license="MIT",
    packages=find_packages(),
    package_data={'senseppi': ['default_model/*']},
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'senseppi=senseppi.__main__:main',
        ],
    },
    install_requires=[
        "numpy",
        "pandas",
        "wget",
        "scipy",
        "networkx",
        "torch>=1.12",
        "matplotlib",
        "seaborn",
        "tqdm",
        "scikit-learn",
        "pytorch-lightning==1.9.0",
        "torchmetrics",
        "biopython",
        "fair-esm"
    ],
)