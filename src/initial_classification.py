print('Importing modules...')
import pandas as pd
import datasets
from datasets import Dataset
import os
from tqdm import tqdm
import torch
from collections import Counter
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

LOG_FILE_NAME = None

def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='name of HuggingFace dataset') 
    parser.add_argument('--label_cols', nargs='+', help= 'List of classification labels/tasks, must be columns in the dataset of type ClassLabel')
    parser.add_argument('--epochs', type=int, help="how many epochs to run the model for")
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int, help='seed for train/test splitting')
    parser.add_argument('--log_file_name', type=str, help='what to call the output logfile')
    args = vars(parser.parse_args())
    
    return args

def log(message):
    global LOG_FILE_NAME
    log_path = os.path.join('out', 'logs')
    os.makedirs(log_path, exist_ok=True)

    with open(os.path.join(log_path, f'{LOG_FILE_NAME}.txt'), "a") as f:
        f.write(message + "\n")

def classify_all_features(ds_splits, 
                          model_name, 
                          batch_size, 
                          epochs,
                          labels, 
                          device):

    for label in labels:
        try:
            fit_and_predict(ds_splits,
                            model_name,
                            label,
                            batch_size, 
                            epochs, 
                            device)
            
            print(f'Classification done for {model_name} - {label}')
            log(f'Classification done for {model_name} - {label}')

        except Exception as e:

            # print full traceback
            tb_str = traceback.format_exc()
            print(f"Classification failed for: {model_name} - {label}")
            print(tb_str)

            log(f"Classification failed for {model_name} - {label} - Error:\n{tb_str}")
            continue

def main():
    
    global LOG_FILE_NAME

    # parse arguments
    args = argument_parser()

    LOG_FILE_NAME = args['log_file_name']

    data_name = args['dataset']

    # load full dataset with all images from disk
    ds_full = datasets.load_from_disk(os.path.join('data', data_name))

    # add column with indices
    ds_full = ds_full.add_column('old_emb_indices', range(len(ds_full)))

    # we remove some rows by the most frequent artists
    idx_to_remove = []
    rng = np.random.default_rng(2830) # random number generator

    for item in Counter(ds_full['artist_str']).most_common(5):
        artist = item[0]
        n = item[1]

        artist_indices = [idx for idx, a in enumerate(ds_full['artist_str']) if a == artist]
        artist_remove = rng.choice(artist_indices, size=len(artist_indices)-800, replace=False).tolist()
        idx_to_remove.extend(artist_remove)
    
    # only select rows not in idx_to_remove
    ds_full = ds_full.select([i for i in range(len(ds_full)) if i not in idx_to_remove])

    # verify correct rows have been removed
    #print(Counter(ds_full['artist_str']))
    #print(Counter(ds_full['style_str']))
    #print(Counter(ds_full['genre_str']))

    # split data
    ds_split = ds_full.train_test_split(test_size=0.2, seed=args['seed']) # implicit shuffle=True
    ds_train = ds_split['train']
    ds_test = ds_split['test']

    # split test data into test and validation
    ds_test_split = ds_test.train_test_split(test_size=0.5, seed=args['seed'])# implicit shuffle=True
    ds_val = ds_test_split['train']
    ds_test = ds_test_split['test']

    # save test data explicitly to disk for later manipulation and prediction
    ds_test.save_to_disk(os.path.join('data', f"wikiart_filtered_test_split"))

    ds_splits = {
            'train': ds_train,
            'test': ds_test,
            'val': ds_val}

    print(f"SIZE OF TRAIN SPLIT:{len(ds_splits['train'])}")
    print(f"SIZE OF VAL SPLIT:{len(ds_splits['val'])}")
    print(f"SIZE OF TEST SPLIT:{len(ds_splits['test'])}")

    # use GPU if available, otherwise run with CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # loop over all models which embeddings have been extracted from
    model_list = os.listdir(os.path.join('data', 'filtered_embeddings_FINAL'))

    for model_name in model_list:
        classify_all_features(ds_splits, 
                          model_name, 
                          args['batch_size'], 
                          args['epochs'],
                          args['label_cols'],
                          device)

if __name__ == '__main__':
    main()