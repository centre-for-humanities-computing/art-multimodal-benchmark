# A Benchmark Dilemma 

<a href="https://chc.au.dk"><img src="https://github.com/centre-for-humanities-computing/intra/raw/main/images/onboarding/CHC_logo-turquoise-full-name.png" width="25%" align="right"/></a>

This repository contains the code accompanying the paper *A Benchmark Dilemma*.

## Project Structure

```
art-multimodal-benchmark/
│
├── data/                                # this is where loaded HuggingFace datasets and extracted embeddings should be placed
│   # should anything else be here?
│
├── src/                                 # Source code 
│   ├── classification_wikidata.py       # classify artists from wikidata dataset
│   ├── classify_augmentations.py        # run classification with augmented data
│   ├── custom_augmentations.py          # contains code to create custom data augmentations
│   ├── embeddings_wikidata.py           # extract embeddings from wikidata dataset
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
│   └── wikidata_clf_results/            # results from wikidata classification
│
├── .env                                 # contains HuggingFace token (currently empty, needs to be specified by the user)
├── all_models.txt                       # list of embedding models used for this project                    
├── extract_embeddings_wikidata.sh       # run embeddings_wikidata.py with specified list of models
├── README.md               
├── requirements.txt                     # Python dependencies
├── run_augmentations.sh                 # runs extract_augmented_embeddings.py and classify_augmentations.py to extract augmented embeddings and classify them
├── run_subclf.sh                        # run subclassification.py script with defined subclassification task
├── run_wikidata_clf.sh                  # run classification_wikidata.py scrip
├── SAM_requirements.txt                 # required packages + versions to run SAM3 + segmentation code
├── sam_setup.sh                         # download SAM3 model and install required packages with specified versions
└── setup.sh                             # set up virtual environment and install required packages
```

## Data & Code

### Data

The datasets used can be found on [HuggingFace](https://huggingface.co/). # link? 

for code for extraction of EVA-02-CLIP features see **link_to_bachelor_github**


### Usage

First, clone the repo with 

```
git clone https://github.com/centre-for-humanities-computing/art-multimodal-benchmark.git

```

In the main folder, set up virtual environment and install required packages with

```
bash setup.sh
```

To run feature extraction, run: 

```
???
```

To run classifications, run:

```
???
```