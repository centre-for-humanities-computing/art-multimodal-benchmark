# A Benchmark Dilemma 

<a href="https://chc.au.dk"><img src="https://github.com/centre-for-humanities-computing/intra/raw/main/images/onboarding/CHC_logo-turquoise-full-name.png" width="25%" align="right"/></a>

This repository contains the code accompanying the paper *A Benchmark Dilemma*.

## Project Structure

```
art-multimodal-benchmark/
│
├── data/                                # (these files are not pushed, but created as when running the code)
│   ├── wikiart/                         # -> download WikiArt dataset from HuggingFace and save to disk as 'wikiart'
│   ├── wikiart_embeddings/              # created when running extract_embeddings.py with wikiart data
│   ├── wikidata/                        # -> download WikiData dataset from HuggingFace and save to disk as 'wikidata'
│   ├── wikidata_embeddings/             # created when running extract_embeddings.py with wikidata data
│   └── aug_embeddings/                  # created when running extract_augmented_embeddings.py with wikiart data
│
├── src/                                 # Source code
│   ├── classification_utils.py          # utils for building initial classification model
│   ├── classification_wikidata.py       # classify artists from wikidata dataset
│   ├── classify_augmentations.py        # run classification with augmented data
│   ├── custom_augmentations.py          # contains code to create custom data augmentations
│   ├── extract_embeddings.py            # extract embeddings from HuggingFace dataset with MIEB/timm models
│   ├── extract_augmented_embeddings.py  # extract embeddings from augmented images
│   ├── initial_classification.py        # code to run initial classification task on embeddings
│   ├── segmentation_task.py             # code to run tree segmentation + augmentation with SAM3
│   └── subclassification.py             # code to run subclassification with input subclassification task
│
├── out/                                 # All outputs
│   ├── classification_reports/          # classification reports of initial classification task on genre, styles and artists
│   ├── extraction_times/                # feature extraction times for wikidata embeddings
│   ├── misclassified_examples_subclassifications/ # plots of misclassification examples for the subclassification task
│   ├── subclassification_conf_matrices/ # confusion matric plots for subclassification task
│   ├── subclassification_reports/       # classification reports for subclassification task
│   ├── test_augmentation_results/       # classification reports for data augmentation classification task
│   ├── segmentations/                   # .csv scripts with segmentation results metadata
│   └── wikidata_clf_results/            # results from wikidata classification
│
├── .env                                 # contains HuggingFace token (currently empty, needs to be specified by the user)
├── all_models.txt                       # list of embedding models used for this project                    
├── extract_embeddings.sh                # run script to extract embeddings for dataset with specified list of models
├── README.md               
├── requirements.txt                     # Python dependencies
├── run_augmentations.sh                 # runs extract_augmented_embeddings.py and classify_augmentations.py to extract augmented embeddings and classify them
├── run_subclf.sh                        # run subclassification.py script with defined subclassification task
├── run_wikidata_clf.sh                  # run classification_wikidata.py scrip
├── SAM_requirements.txt                 # required packages + versions to run SAM3 + segmentation code
├── sam_setup.sh                         # download SAM3 model and install required packages with specified versions
└── setup.sh                             # set up virtual environment and install required packages
```

## Prerequisites

First, clone the project's repository:

```
git clone https://github.com/centre-for-humanities-computing/art-multimodal-benchmark.git
```

In order to use SAM3, you need to agree to share your contact information and specify a personal HuggingFace token. See https://huggingface.co/facebook/sam3. Next, create a HuggingFace token (which allows usage of the models) and insert it in the ```.env``` file in the repo. 

## Data

A filtered & cleaned version of WikiArt ([Huggan link](https://huggingface.co/datasets/huggan/wikiart)) version can be found on HuggingFace HERE.

The WikiData dataset used can be found HERE.

We recommed downloading the datasets via HuggingFace and placing it in the ```data``` folder:

```
import datasets
import os
import argparse 
from functools import partial

# load dataset from the hub
hf_data = datasets.load_dataset('dataset_name', split='train', streaming=True)

# convert dataset to iterable generator
def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

# convert to dataset to be saved locally
ds = datasets.Dataset.from_generator(partial(gen_from_iterable_dataset, hf_data), features=hf_data.features)

# create datafolder
os.makedirs('data', exist_ok=True)

# save to disk
ds.save_to_disk(os.path.join('data', args['local_name']))
```

## Usage

First, clone the repo with 

```
git clone https://github.com/centre-for-humanities-computing/art-multimodal-benchmark.git

```

In the main folder, set up virtual environment and install required packages with

```
bash setup.sh
```

### Extract embeddings
To run feature extraction, run: 

```
???
```

### Benchmarking tasks 
To run classifications, run:

```
???
```