import datasets
import numpy as np
import os
from datasets import ClassLabel
import pandas as pd 
import torch
import os
from torch import optim, nn, utils
from torch.utils.data import Dataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
#from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torchmetrics.classification import ConfusionMatrix
import argparse
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import timm

from subclf_updated import remap_features, create_dataloader, build_model, define_class_weights, SubclassModel, save_conf_matrix, plot_misclassifications, save_results
from custom_augmentations import AddFrame, JPEGCompression, AddVignette, AddGrain, AddLightArtifact

def argument_parser():
    ..

    return args 

def extract_embeddings(dataset, model, aug):
    # for large datasets, it is better to input DataLoaders to get_image_embeddings function rather than a list

    # specify transforms; convert dataset image to PIL and np array (necessary as input to DataLoader)
    
    def convert_to_rgb(img):
        return img.convert("RGB")

    def to_numpy_array(img):
        return np.array(img)

    transform = transforms.Compose([
        convert_to_rgb,
        aug, # add augmentation to the transforms
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

# create function for each augmentation

def create_test_loader(test_data, test_embeddings, label, batch_size):
    class EmbeddingsDataset(Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = embeddings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]
    
    embeddings_tensor = test_embeddings.float()

    y = test_data[label]
    labels_tensor = torch.tensor(y)

    dataset = EmbeddingsDataset(embeddings_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader

def embeddings_w_eva():
    

def main():
    L.seed_everything(2830)

    args = argument_parser()

    # define augmentations: (make this a bit more elegant?)
    augmentations = [
        T.GaussianBlur(kernel_size=(9, 9), sigma=(5,5)), # weak blurring
        T.GaussianBlur(kernel_size=(15, 15), sigma=(5,5)), # stronger blurring
        T.Grayscale(), # grayscale
        T.ColorJitter(contrast = 8), # changing contrasts
        AddFrame(), # frame
        JPEGCompression(quality=8), # jpeg compression
        AddVignette(strength = 0.9), # adding vignette
        AddGrain(std=0.07),
        AddGrain(std=0.15),
        AddLightArtifact(max_intensity=0.4, max_radius_ratio=0.4)
    ]

    # load data
    data_name = args['dataset']

    # load full dataset with all images from disk
    ds_full = datasets.load_from_disk(os.path.join('data', data_name))

    # add indices as column
    ds_full = ds_full.add_column('old_indices', range(len(ds_full)))
    batch_size = args['batch_size']

    skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2830) # NOT SURE WE SHOULD STRATIFY??
    
    # THIS WILL NOT WORK!!!!!!
    labels = np.array(ds_full[label])
    indices = np.arange(len(ds_full))

    model_scores = {m: [] for m in args["model_names"]}

    for label in labels:
        for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):

            # monitor folds
                print(f"Fold {fold+1}")

                ds_train = ds_splits.select(train_idx.tolist())
                ds_test = ds_splits.select(val_idx.tolist())

                ds_splits_for_cv = {
                    'train': ds_train,
                    'test': ds_test}

            for model_name in args['model_names']:

                full_embedding_pt = torch.load(os.path.join('data', 'filtered_embeddings_FINAL', model_name, f'{model_name}_all_splits.pt'))
                train_loader, inp_size = create_dataloader(ds_splits_for_cv, full_embedding_pt, label, 'train', batch_size, 'old_indices')
                
                if model_name == 'eva02_clip_336':
                    # extract embeddings via TIMM

                else:

                    model_path = + model_name 

                    try:
                        print('LOADING MODEL...')
                        model = model_meta.load_model()
                    
                    except Exception as e:
                        log(f'Error loading model: {model_path}, {e}')
                        print(f'Error loading model: {model_path}, {e}')
                    
                    # fit seperate model for each data augmentation
                    for aug in augmentations: 

                        try:
                            # extract embeddings with each transform
                            embeddings = extract_embeddings(ds_splits_for_cv['test'], model, aug)

                        except Exception as e:
                            print(e)
        
                        if embeddings is None:
                            raise ValueError("No embeddings returned")

                        # create test dataloader:
                        test_loader = create_test_loader(ds_splits_for_cv['test'], embeddings, label, batch_size)
                        model_architecture = build_model(args['hidden_layer_size'], label, inp_size, 0.3, ds_splits_for_cv)

                        # define class weights
                        class_weights = define_class_weights(ds_splits_for_cv, label)

                        # define lightning model
                        model = SubclassModel(model_architecture, class_weights, lr=args['lr'], weight_decay=0.01, num_classes = ds_splits_for_cv['train'].features[label].num_classes)

                        trainer = L.Trainer(
                                        max_epochs=args['epochs'],
                                        #callbacks=[checkpoint_callback],
                                        accelerator="gpu" if torch.cuda.is_available() else "cpu",
                                        devices="auto",
                                        deterministic=True
                                         )
                
                        trainer.fit(model, train_loader)

                        test_metrics = trainer.test(model, test_loader)

                        # save across folds
                        model_scores[model_name].append({
                        "acc": test_metrics[0]["test_acc"],
                        "precision": test_metrics[0]["test_precision"],
                        "recall": test_metrics[0]["test_recall"],
                        "f1": test_metrics[0]["test_f1"]
                            })

                        all_preds_batches = trainer.predict(model, test_loader)
                        all_preds = torch.cat(all_preds_batches).cpu().numpy()

                        if fold == 4:
                            save_results(
                                test_data = ds_splits_for_cv['test'],
                                y_pred = all_preds,
                                model_name = model_name,
                                label_col = label,
                                task_name = f"{args['savefile_suffix']}_fold{fold+1}_{}"
                            )

                        save_model_scores(model_scores, args['savefile_suffix'])

        del full_embedding_pt

                    
                    
                    
                    
                    # delete model from memory
                    del model 

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    








