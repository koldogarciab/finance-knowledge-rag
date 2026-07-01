# Environment setup

The project uses a dedicated Conda environment with Python 3.12.

## Create the environment

Command:

    conda env create -f environment.yml

## Activate it on Windows PowerShell

When Conda is already initialised:

    conda activate finance-rag

If the activation hook is not loaded:

    & "$env:USERPROFILE\anaconda3\shell\condabin\conda-hook.ps1"
    conda activate finance-rag

## Install the project dependencies

    python -m pip install -r requirements.txt

PyTorch is installed from the official CPU wheel repository. A GPU is not required.

## Validate the environment

    python -m pip check
    python --version

The expected Python version is 3.12.

The embedding pipeline uses:

- sentence-transformers/all-MiniLM-L6-v2
- 384-dimensional normalised embeddings
- exact cosine-similarity retrieval with NumPy

The public embedding model is downloaded from Hugging Face during the first execution and is then stored in the local model cache.
