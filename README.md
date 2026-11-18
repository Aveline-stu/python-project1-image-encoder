# Project 1: NumPy Image Encoder — Run Instructions

## Environment
Python 3.9 or later. Install dependencies:

    python -m pip install -r requirements.txt

No GPU, PyTorch, or network connection is required.

## Run all numerical checks

    python check.py --stage all

Or run stage by stage:

    python check.py --stage assets
    python check.py --stage preprocess
    python check.py --stage patches
    python check.py --stage tokens
    python check.py --stage block0
    python check.py --stage last_tokens
    python check.py --stage output
    python check.py --stage batch

## Generate output tokens

    python run.py --output output_tokens.npz

This saves a `(10, 512)` array of normalized image tokens for the local images.

## What was edited
Only `image_encoder.py` was modified (TODO-01 through TODO-06). `helpers.py`, `preprocess_image.py`, `check.py`, `run.py`, and the supplied weights/reference files were left unchanged.
