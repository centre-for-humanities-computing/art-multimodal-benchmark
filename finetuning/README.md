# A Benchmark Dilemma

<a href="https://chc.au.dk"><img src="https://github.com/centre-for-humanities-computing/intra/raw/main/images/onboarding/CHC_logo-turquoise-full-name.png" width="25%" align="right"/></a>


This code trains a lightweight projection head on top of frozen, pretrained backbones — "finetuning" is used informally throughout the file and folder names here; see the paper *A Benchmark Dilemma* for the precise methodology.

## Structure

```
art-multimodal-benchmark
├── finetuning/
│   ├── finetune-landscape.py            # script for finetuning models
│   ├── requirements.txt                 # requirements for finetuning specifically (be mindful of the correct torch version for your system)
│   ├── README.md                        # readme specifically for the finetuning code
│   ├── cache/                           # created when running finetuning, used for caching
│   ├── checkpoints/                     # created when running finetuning, contains weights for the best-performing epoch
│   ├── logs/                            # created when running finetuning, contains log files with info on the run; currently contains the logs of the runs presented in the paper
```

## Data

The data accompanying the finetuning can be downloaded from Hugging Face at the following link: https://huggingface.co/datasets/MKipke/finetuning-landscape-painting. The code is built to run directly with the data from Hugging Face — no manual download is needed.

## Prerequisites

Please install the packages listed in `requirements.txt`, but be mindful of getting the correct torch version with CUDA support for your specific system — see https://pytorch.org/get-started/locally/. If your system doesn't have CUDA support, you can still run the code on CPU, but it's significantly slower; running the larger models (e.g. CLIP-ViT-bigG-14, SigLIP-SO400M) on CPU is not recommended.

You can install the packages by following these steps:

Navigate into the finetuning folder and create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

- **Windows:**
```bash
.venv\Scripts\activate
```
- **Mac:**
```bash
source .venv/bin/activate
```

Then, install the correct version of PyTorch first: go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), select your operating system and CUDA version (or **CPU** if you don't have an NVIDIA GPU), and it will generate an install command for you. Copy and paste that exact command into your terminal and press Enter.

Then, install everything else:

```bash
pip install -r requirements.txt
```

## Usage

The script has an extensive config section at the top where you can choose between the nine models benchmarked in our paper (via `RUN_MODELS` — by default only `DINOv2-base` is set to run), hyperparameters for each model (epochs, learning rates, etc.), and between two pooling methods (attention-pooling, stats-pooling), then re-run it as you wish. Note that the setup is inherently non-deterministic, so re-running it may result in slightly different models than the ones presented in the paper.

You can then run the script with:

```bash
python finetune-landscape.py
```