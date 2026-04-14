import datasets
import numpy as np
import os
from datasets import ClassLabel
import pandas as pd 
import torch
import os
from torch import optim, nn, utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import random
import math
import matplotlib.pyplot as plt
import timm
import torchvision.transforms as T
import mteb
from PIL import Image
from tqdm import tqdm

import cv2
cv2.setNumThreads(0)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def remap_features(ds_original, ds_filtered, label):

    '''
    When a HuggingFace dataset with a ClassLabel is filtered, it still saves the original Class mapping - even if some classes are filtered out.
    This function resets that.
    '''

    original_feature = ds_original.features[label] # the ClassLabel feature
    original_names = original_feature.names

    # classes(names) in new subclassification dataset:
    used_class_names = sorted(list(set(ds_filtered[f"{label}_str"])))
    new_class_label = ClassLabel(names=used_class_names)

    # set up function to remap from str -> int for new ClassLabels
    def remap_labels(example):
        example[label] = new_class_label.str2int(example[f"{label}_str"])
        return example
    
    # use map to remap classlabels
    ds_filtered = ds_filtered.map(remap_labels)

    # recast the class label feature to new labels
    new_features = ds_filtered.features.copy()
    new_features[label] = new_class_label
    ds_filtered = ds_filtered.cast(new_features)

    return ds_filtered

def filter_data(ds, label, subclassification_task, seed, cv):

    # add old column to keep track of embedding indices
    ds = ds.add_column('old_indices', range(len(ds)))

    # find the rows that matches the subset criterion
    subclass_indices = [idx for idx, a in enumerate(ds[f'{label}_str']) if a in subclassification_task]
    ds_subset = ds.select(subclass_indices)

    # remap labels to fit to new number of classes for subclassification task
    ds_subset = remap_features(ds, ds_subset, label)

    # this is not necessary and should be deleted
    if cv==True:
        #ds_split = ds_subset.train_test_split(test_size=0.2, seed=seed, stratify_by_column=label)

        #ds_splits = {
         #   'train': ds_split['train'], # train/val set
          #  'test': ds_split['test'] # hold-out test set - we're not touching this until the end
          #  }
        return ds_subset

def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='name of huggingface dataset')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_names', nargs='+', help='list of models to run classification task with')

    args = vars(parser.parse_args())
    
    return args 

# create custom dataset class
class HFImageDataset(Dataset):

    """
    Basically just converts a huggingface dataset with images in an 'image' column to a format that MTEB will accept
    """
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

def to_numpy_array(img):
    return np.array(img).astype(np.float32) / 255.0

def extract_embeddings(dataset, model, batch_size):
    # for large datasets, it is better to input DataLoaders to get_image_embeddings function rather than a list

    # specify transforms; convert dataset image to PIL and np array (necessary as input to DataLoader)
    transforms = T.Compose([
        convert_to_rgb,
    ]) 
    
    # need to define custom data collater
    def pil_collate_fn(batch):
        return {"image": batch}

        # apply
    try:
        wrapped_dataset = HFImageDataset(hf_dataset=dataset, transform=transforms)

        dataloader = DataLoader(wrapped_dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                collate_fn=pil_collate_fn)
    
        # process images in batches from dataloader
        embeddings = model.get_image_embeddings(dataloader)

        return embeddings
    
    except Exception as e:
        print(f'Error processing images with model: {model}, {e}')
        return None

def embeddings_w_eva(test_data, batch_size):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model from timm
    model = timm.create_model('eva02_large_patch14_clip_336.merged2b', pretrained=True, num_classes=0)
    model.eval() # turn on evaluation mode
    model.to(device)

    # save preprocessing information from the pretrained model
    data_config = timm.data.resolve_model_data_config(model)

    # use this information to transform the data
    transforms_model = timm.data.create_transform(**data_config, is_training=False)

    transforms_list = []

    # create transforms list - RGB --> AUG --> MODEL_SPECIFIC_PREPROCESSINGS
    transforms_list.append(convert_to_rgb)
    transforms_list.extend(transforms_model.transforms)

    transforms = T.Compose(transforms_list)

    wrapped_dataset = HFImageDataset(hf_dataset=test_data, transform=transforms)
    dataloader = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=False) # no need for custom collating because data is tensors for timm input

    # extract embeddings from batches:
    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            embeddings = model(batch)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def add_model_prefix(model_name):

    name = model_name.lower()

    # --- OpenAI CLIP ---
    if name == "clip-vit-large-patch14":
        return f"openai/{model_name}"

    # --- LAION CLIP ---
    if name.startswith("clip-vit"):
        return f"laion/{model_name}"

    # --- SigLIP ---
    if name.startswith("siglip"):
        return f"google/{model_name}"

    # --- DINOv2 ---
    if name.startswith("dinov2"):
        return f"facebook/{model_name}"

def filter_test_from_sketches(ds_test):

    illustrations_classes = ['sketch_and_study', 'illustrations']

    no_illu_indices = [idx for idx, a in enumerate(ds_test['genre_str']) if a not in illustrations_classes]

    ds_test_filtered = ds_test.select(no_illu_indices)

    ds_test_filtered = remap_features(ds_test, ds_test_filtered, 'genre')

    return ds_test_filtered

class AugmentFn:
    def __init__(self, aug):
        self.aug = aug
    
    def __call__(self, example):
        import cv2
        img = example['image'].convert("RGB")
        img = self.aug(img)
        img = img.convert("RGB")
        return {'image': img}

class PILImageDataset:
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {'image': self.images[idx]}

def main():

    # import custom augmentation functions
    from custom_augmentations_new import AddLayeredFrame, JPEGCompression, AddVignette, AddGrain, AddLightArtifact, RelativeGaussianBlur, FixedContrast, CannySketch, PencilSketchCustom
    
    args = argument_parser()

    # create parent save folder
    os.makedirs(os.path.join('data', 'aug_embeddings'), exist_ok=True)

    augmentations = {
        'strong_blur': RelativeGaussianBlur(strength=0.7, sigma=7),
        'grayscale': T.Grayscale(), # grayscale
        'contrast': FixedContrast(factor = 10), # changing contrasts
        'frame': AddLayeredFrame(border_sizes=(100, 100, 100)), # frame #
        'jpeg_compr': JPEGCompression(quality=6), # jpeg compression
        'vignette': AddVignette(strength = 0.9), # adding vignette
        'weak_grain': AddGrain(std=0.3), 
        'light_artifact': AddLightArtifact(max_intensity=0.4, max_radius_ratio=0.4),
        'Canny_sketch': CannySketch(), # edge detection algorithm to create a fake drawing
        'pencil_sketch': PencilSketchCustom() # other edge detection algorithm to create a fake drawing
    }

    # load data
    data_name = args['dataset']
    ds = datasets.load_from_disk(os.path.join('data', data_name), keep_in_memory=False)

    # if wikidata, maybe need to map int2str

    if data_name == 'wikidata_remapped':
        def map_int_to_str(example):
            return {
            'artist_str': ds.features['artist'].int2str(example['artist'])
            }
        ds = ds.map(map_int_to_str, batched=False)

        subset = [
            'Eugene Louis Boudin',
            'Paul Cezanne',
            'Camille Pissarro',
            'Alfred Sisley',
            'Edouard Manet'
        ]

        # filter
        ds_filtered = filter_data(ds, 'artist', subset, 2830, cv=True) # ignore last two parameters, just reusing function from other script

    else:  # if wikiart

        subset = [
            "camille-pissarro",
            "paul-cezanne",
            "alfred-sisley",
            "edouard-manet",
            "eugene-boudin"
        ]

        # filter 
        ds_filtered = filter_data(ds, 'artist', subset, 2830, cv=True) # ignore last two parameters, just reusing function from other script

        ds_filtered = filter_test_from_sketches(ds_filtered)

    # save the filtered dataframe
    ds_filtered.save_to_disk(os.path.join('data', f"{data_name}_AUG_SUBSET"))

    batch_size = args['batch_size']

    for aug_name, aug in augmentations.items():

        try:
            augmented_images = []
            for idx in tqdm(range(len(ds_filtered)), desc=f"Augmenting [{aug_name}]"):
                img = ds_filtered[idx]['image'].convert("RGB")
                aug_img = aug(img).convert("RGB") # for grayscale images, we are not converting back to color - we give it three channels but it is still grayscaled
                augmented_images.append(aug_img)

                del img, aug_img

            # packaging the augmented images in a custom dataset class 
            ds_augmented = PILImageDataset(augmented_images)
            img_example = augmented_images[100]
            del augmented_images

            # sanity check image augmentation
            aug_img_out_path = os.path.join('out', f'{data_name}_aug_img_ex')
            os.makedirs(aug_img_out_path, exist_ok=True)
            img_example.save(os.path.join(aug_img_out_path, f"{aug_name}.png"))

        except Exception as e:
            print(e)
            continue

        # create folder for saving the embeddings
        aug_folder_path = os.path.join('data', 'aug_embeddings', aug_name)
        os.makedirs(aug_folder_path, exist_ok=True)

        # then extract embeddings:
        for model_name in args['model_names']:   # make this if else statement less awkward !
            if model_name == 'eva02_clip_336':
                pass #  
            
            else:
                model_path = add_model_prefix(model_name)
                try:
                    model_meta = mteb.get_model_meta(model_path)
                    print('LOADING MODEL...')
                    mteb_model = model_meta.load_model()
                
                except Exception as e:
                    print(f'Error loading model: {model_path}, {e}')

                    continue # skip this model if error
            
            #embeddings = None
            if model_name == 'eva02_clip_336':
                try:
                    embeddings = embeddings_w_eva(ds_augmented, batch_size)
                
                except Exception as e:
                    print(f'Error extraction features, {e}')
            
            else:
                try: 
                    embeddings = extract_embeddings(ds_augmented, mteb_model, batch_size)

                except Exception as e: 
                    print(e)
            
            # save embeddings
            aug_emb_model_path = os.path.join(aug_folder_path, f'{model_name}.pt')
            torch.save(embeddings, aug_emb_model_path)

            del embeddings 

            if model_name != 'eva02_clip_336':
                del mteb_model

        del ds_augmented
        import gc
        gc.collect()

if __name__ == '__main__':
    main()


