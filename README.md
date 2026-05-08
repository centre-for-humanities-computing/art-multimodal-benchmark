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
├── plots/                               # plots
│   ├── violin_cosine.png                # violin plot for cosine similarity
│   ├── violin_llm_judge.png             # violin plot for llm-as-a-judge
│   ├── violin_cosine_DUMMY.png          # violin plot for cosine similarity for dummy data
│   └── violin_llm_judge_DUMMY.png       # violin plot for llm-as-a-judge for dummy data
│
├── .env                                 # contains HuggingFace token (currently empty, needs to be specified by the user)
├── README.md                    
├── requirements.txt                     # Python dependencies
├── run_python_scripts_dummy.sh          # run python scripts from src/ with dummy data on GPU
├── run_python_scripts_dummy_CPU.sh      # run python scripts from src/ with dummy data on CPU
├── run_r_scripts_dummy.sh               # Run R scripts from src/ with set dummy data
├── env_to_jupyter.sh                    # creates kernel from .venv to be used for jupyter 
└── setup.sh                             # Set up virtual environment and install required packages
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