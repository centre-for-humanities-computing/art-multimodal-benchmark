print('Importing modules...')
import pandas as pd
import datasets
from datasets import Dataset
import os
from tqdm import tqdm
import torch
from collections import Counter
#from datasets import load_dataset
import numpy as np
import argparse 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# IMPORT fit_and_predict from classify.py script
import sys
sys.path.append(os.path.dirname(__file__))
from classify_updated import fit_and_predict
import traceback

def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='name of HuggingFace dataset') 
    parser.add_argument('--subclasses', nargs='+', help= 'List of classes to run subclassification on')
    parser.add_argument('--subclass_label', type=str, help='whether chosen subclassification task is for genre, styles or artists')
    parser.add_argument('--epochs', type=int, help="how many epochs to run the model for")
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--log_file_name', type=str, help='what to call the output logfile')
    args = vars(parser.parse_args())
    
    return args


def main():

    # modulize this later

    # load data
    data_name = args['dataset']

    # load full dataset with all images from disk
    ds_full = datasets.load_from_disk(os.path.join('data', data_name))

    # add column with indices
    ds_full = ds_full.add_column('old_emb_indices', range(len(ds_full)))

    # only select rows with selected artist/styles/genres

    

    # save confusion matrix


