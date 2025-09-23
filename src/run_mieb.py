print('Importing modules...')
import pandas as pd
import datasets
#from datasets import Image as Image_ds # change name because of similar PIL module
from datasets import Dataset
import os
#from PIL import Image
from tqdm import tqdm
import torch
#from datasets import load_dataset
import numpy as np
from PIL import Image
import mteb
import argparse 
from functools import partial
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# IMPORT fit_and_predict from classify.py script
import sys
sys.path.append(os.path.dirname(__file__))
from classify import fit_and_predict

LOG_FILE_NAME = None

def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path of model in MTEB to use')
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

def extract_embeddings(dataset, model):
    # for large datasets, it is better to input DataLoaders to get_image_embeddings function rather than a list

    # specify transforms; convert dataset image to PIL and np array (necessary as input to DataLoader)
    
    def convert_to_rgb(img):
        return img.convert("RGB")

    def to_numpy_array(img):
        return np.array(img)

    transform = transforms.Compose([
        convert_to_rgb,
        to_numpy_array
    ])  

    # create custom HuggingFace dataset class to input to DataLoader
    class HFImageDataset(Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.dataset = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image = self.dataset[idx]['image']
            if self.transform:
                image = self.transform(image)
            
            return image
    
    # need to define custom data collater, create batch of list instead of stacking (not possible as input are not tensors)
    def pil_collate_fn(batch):
        return batch

    # apply
    try:
        wrapped_dataset = HFImageDataset(hf_dataset=dataset, transform=transform)

        dataloader = DataLoader(wrapped_dataset, batch_size=32, shuffle=False, collate_fn=pil_collate_fn)
    
        # process images in batches from dataloader
        embeddings = model.get_image_embeddings(dataloader)

        return embeddings

    except Exception as e:
        log(f'Error processing images with model: {model}')
        print(f'Error processing images with model: {model}')
        return None

def embeddings_from_splits(ds_splits, model_path):

    # get model meta and load model
    model_meta = mteb.get_model_meta(model_path)

    try:
        print('LOADING MODEL...')
        model = model_meta.load_model()

    except Exception as e:
        log(f'Error loading model: {model_path}, {e}')
        print(f'Error loading model: {model_path}, {e}')
    
    # now we only want the name of the model, not the entire HuggingFace path
    model_name = model_path.split('/')[1]

    # create folder to save embeddings to
    embeddings_outpath = os.path.join('data', 'embeddings')
    os.makedirs(embeddings_outpath, exist_ok=True)

    # extract image embeddings for all images across splits with model
    print(f'Extracting embeddings with {model_name}')

    for split_name in ds_splits:
        dataset_split = ds_splits[split_name] # get i.e., train dataset
        
        try:
            embeddings = extract_embeddings(dataset_split, model)

        except Exception as e:
            print(e)
        
        if embeddings is None:
            raise ValueError("No embeddings returned")

        # save embeddings to folder for model:
        os.makedirs(os.path.join(embeddings_outpath, model_name), exist_ok=True)
        torch.save(embeddings, os.path.join(embeddings_outpath, model_name, f'{model_name}_{split_name}.pt'))

        # delete embeddings for that split from memory
        del embeddings
    
    # after extracting embeddings, delete model from memory:
    del model 

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def classify_all_features(ds_splits, 
                          model_name,
                          hidden_layer_size, 
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

    ds_train = datasets.load_from_disk(os.path.join('data', f'{data_name}_train'))
    ds_test = datasets.load_from_disk(os.path.join('data', f'{data_name}_test'))
    ds_val = datasets.load_from_disk(os.path.join('data', f'{data_name}_val'))

    ds_splits = {
    'train': ds_train,
    'test': ds_test,
    'val': ds_val}

    # use GPU if available, otherwise run with CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # extract embeddings and save to file
    #embeddings_from_splits(ds_splits, args['model_path'])

    model_name = args['model_path'].split('/')[1]

    # Now I should have all data needed for running classify.py scripts
    classify_all_features(ds_splits, 
                          model_name,
                          args['hidden_layer_size'], 
                          args['batch_size'], 
                          args['epochs'],
                          args['label_cols'],
                          device)

    log(f'Feature extraction and classification completed for {model_name}!')

if __name__ == '__main__':
    main()