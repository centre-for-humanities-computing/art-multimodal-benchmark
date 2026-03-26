print('Importing modules...')
import pandas as pd
import datasets
from datasets import Image as Image_ds # change name because of similar PIL module
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
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import timm

def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', help='lists with full path of models in MTEB to use')
    parser.add_argument('--dataset', type=str, help='name of huggingface dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    args = vars(parser.parse_args())
    
    return args

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

def convert_to_rgb(img):
    return img.convert("RGB")

def extract_embeddings(dataset, model):
    # for large datasets, it is better to input DataLoaders to get_image_embeddings function rather than a list

    # specify transforms; convert dataset image to PIL and np array (necessary as input to DataLoader)

    def to_numpy_array(img):
        return np.array(img)

    transform = T.Compose([
        convert_to_rgb
    ])  
    
    # need to define custom data collater, create batch of list instead of stacking (not possible as input are not tensors)
    def pil_collate_fn(batch):
        return {"image": batch}

    # apply
    try:
        wrapped_dataset = HFImageDataset(hf_dataset=dataset, transform=transform)

        dataloader = DataLoader(wrapped_dataset, batch_size=32, shuffle=False, collate_fn=pil_collate_fn)
    
        # process images in batches from dataloader
        embeddings = model.get_image_embeddings(dataloader)

        return embeddings

    except Exception as e:
        print(f'Error processing images with model: {model}, {e}')
        return None

def extract_eva_embeddings(ds):

    # load model from timm
    model = timm.create_model('eva02_large_patch14_clip_336.merged2b', pretrained=True, num_classes=0)
    model.eval() # turn on evaluation mode

    # save preprocessing information from the pretrained model
    data_config = timm.data.resolve_model_data_config(model)
    transforms_model = timm.data.create_transform(**data_config, is_training=False)

    # convert to RBG and apply model-specific transforms
    transforms_list = [convert_to_rgb] + transforms_model.transforms
    transforms = T.Compose(transforms_list)

    # create custom dataset for input to dataloaders
    wrapped_dataset = HFImageDataset(hf_dataset=ds, transform=transforms)

    # create dataloader
    dataloader = DataLoader(wrapped_dataset, batch_size=32, shuffle=False) #(no need for custom collater because input are already tensors)

    # extract embeddings from batches:
    all_embeddings = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings", total=len(dataloader)):
            batch = batch.to(device)
            embeddings = model(batch)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)

def main():

    args = argument_parser()
    # load data parquet file:
    
    data_path = os.path.join('data', args['dataset'])

    ds = datasets.load_from_disk(data_path) 
    # remove old 'image' column
    ds = ds.remove_columns(["image"])

    # now rename 'images' to 'image'
    ds = ds.rename_column("images", "image")

    # add folder to save embedding extraction times
    times_folder = os.path.join('out', 'extraction_times')
    os.makedirs(times_folder, exist_ok=True)

    # create folder to save embeddings to:
    embeddings_outpath = os.path.join('data', 'wikidata_embeddings')
    os.makedirs(embeddings_outpath, exist_ok=True)

    for model_path in args['models']:

        if model_path == "__/eva02_clip_336":
            try: 
                # start timer
                start_time = time.time()
                embeddings = extract_eva_embeddings(ds)
                end_time = time.time() - start_time

                with open(os.path.join(times_folder, f'eva02_clip_336_embedding_time.txt'), 'w') as f:
                    f.write(str(end_time))

                # save embedding to folder for model
                os.makedirs(os.path.join(embeddings_outpath, 'eva02_clip_336'), exist_ok=True)
                torch.save(embeddings, os.path.join(embeddings_outpath, 'eva02_clip_336', 'eva02_clip_336_all_splits.pt'))
                
                del embeddings
            except Exception as e:
                print(e)

        else:
            model_meta = mteb.get_model_meta(model_path)

            try:
                print('LOADING MODEL...')
                model = model_meta.load_model()

            except Exception as e:
                print(f'Error loading model: {model_path}, {e}')
        
            # now we only want the name of the model, not the entire HuggingFace path
            model_name = model_path.split('/')[1]

            # extract image embeddings for all images with model
            print(f'Extracting embeddings with {model_name}')

            try:
                # start timer
                start_time = time.time()

                # extract embeddings
                embeddings = extract_embeddings(ds, model)

                end_time = time.time() - start_time

                # save embedding to folder for model
                os.makedirs(os.path.join(embeddings_outpath, model_name), exist_ok=True)
                torch.save(embeddings, os.path.join(embeddings_outpath, model_name, f'{model_name}_all_splits.pt'))
                
                with open(os.path.join(times_folder, f'{model_name}_embedding_time.txt'), 'w') as f:
                    f.write(str(end_time))

                # delete embeddings from memory
                del embeddings
                del model
            
            except Exception as e:
                print(e)
        
if __name__ == '__main__':
    main()
