print('Importing modules...')
import pandas as pd
import datasets
from datasets import Dataset
import os
from tqdm import tqdm
import torch
#from datasets import load_dataset
import numpy as np
import argparse 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import train_test_split

# IMPORT fit_and_predict from classify.py script
import sys
sys.path.append(os.path.dirname(__file__))
from classify_updated import fit_and_predict

LOG_FILE_NAME = None

def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path of model in MTEB to use') # FIX
    parser.add_argument('--dataset', type=str, help='name of HuggingFace dataset') 
    parser.add_argument('--label_cols', nargs='+', help= 'List of classification labels/tasks, must be columns in the dataset of type ClassLabel')
    parser.add_argument('--epochs', type=int, help="how many epochs to run the model for")
    parser.add_argument('--hidden_layer_size', type=int, help='size of the hidden layer in classification model')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--log_file_name', type=str, help='what to call the output logfile')
    args = vars(parser.parse_args())
    
    return args

# log error messages and save to file

def log(message):
    global LOG_FILE_NAME
    log_path = os.path.join('out', 'logs')
    os.makedirs(log_path, exist_ok=True)

    with open(os.path.join(log_path, f'{LOG_FILE_NAME}.txt'), "a") as f:
        f.write(message + "\n")

def classify_all_features(ds_full, 
                          model_name,
                          hidden_layer_size, 
                          batch_size, 
                          epochs,
                          labels, 
                          device):
               
        try:
            fit_and_predict(ds_splits,
                            model_name,
                            label,
                            batch_size,
                            hidden_layer_size,  
                            epochs, 
                            device)
            
            print(f'Classification done for {model_name} - {label}')
            log(f'Classification done for {model_name} - {label}')

        except Exception as e:
            log(f"Classification failed for {model_name} - {label} - Error: {e}")
            print(f"Classification failed for: {model_name} - {label} - Error: {e}")
            continue

def main():

    global LOG_FILE_NAME
    # parse arguments
    args = argument_parser()

    LOG_FILE_NAME = args['log_file_name']

    data_name = args['dataset']

    # load full dataset with all images from disk
    ds_full = datasets.load_from_disk(os.path.join('data', data_name))
    ds_full = ds_full.add_column('old_emb_indices', range(len(ds_full))) # verify whether this works!

    # seperate classification for labels
    labels = args['label_cols']
    for label in labels:
        # we'll loop over labels to make sure that datasets are stratified by it

        if label == 'artist':
            # downsample van gogh!

            # van-gogh indices:
            van_gogh_indices = [idx for idx, a in enumerate(ds_full['artist_str']) if a == 'vincent-van-gogh']
             
            # create random number generator
            rng = np.random.default_rng(2830)

            # use this generator to randomly select rows with Van Gogh to remove:
            vg_remove = set(rng.choice(van_gogh_indices, size=len(van_gogh_indices)//2, replace=False)) # removes 50% of van Goghs images

            # only select rows not in vg_remove
            ds_full = ds_full.select([i for i in range(len(ds_full)) if i not in vg_remove])

            # verify correct rows have been removed
            print(Counter(ds_full['artist_str']))

        # train, test, split - stratifying by LABEL
        ds_split = ds_full.train_test_split(test_size=0.2, seed=2830, stratify_by_column = label)
        ds_train = ds_split['train']
        ds_test = ds_split['test']

        # split test data into test and validation
        ds_test_split = ds_test.train_test_split(test_size=0.5, seed=2830, stratify_by_column = label)
        ds_val = ds_test_split['train']
        ds_test = ds_test_split['test']

        # save to disk for later manipulation and prediction
        ds_test.save_to_disk(os.path.join('data', f"wikiart_test_{label}"))

        ds_splits = {
            'train': ds_train,
            'test': ds_test,
            'val': ds_val}

        # use GPU if available, otherwise run with CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)

        # loop over all models which embeddings have been extracted from
        model_list = os.listdir(os.path.join('data', 'filtered_embeddings'))

        # for model and label:
        for model_name in model_list:
    
            # for model and label, fit classification and predict on test data:
            try:
                fit_and_predict(ds_splits, # loads dataset splits without the embeddings
                                model_name, # just name of the model, not path like "google/siglip...."
                                label, 
                                batch_size,
                                hidden_layer_size,  
                                epochs, 
                                device)
                
                print(f'Classification done for {model_name} - {model_name}')
                log(f'Classification done for {model_name} - {model_name}')

            except Exception as e:
                log(f"Classification failed for {model_name} - {model_name} - Error: {e}")
                print(f"Classification failed for: {model_name} - {model_name} - Error: {e}")
                continue

                log(f'Classification completed for {model_name}!')

if __name__ == '__main__':
    main()