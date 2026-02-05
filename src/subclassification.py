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
    parser.add_argument('--label_cols', nargs='+', help= 'List of classification labels/tasks, must be columns in the dataset of type ClassLabel')
    parser.add_argument('--epochs', type=int, help="how many epochs to run the model for")
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--log_file_name', type=str, help='what to call the output logfile')
    args = vars(parser.parse_args())
    
    return args
