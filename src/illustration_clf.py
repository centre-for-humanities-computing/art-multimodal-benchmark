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

from subclf_updated import remap_features, create_dataloader, build_model, define_class_weights, SubclassModel, save_conf_matrix, plot_misclassifications, save_results

def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='name of HuggingFace dataset') 
    parser.add_argument('--subclasses', nargs='+', help= 'List of classes to run subclassification on')
    parser.add_argument('--subclass_label', type=str, help='whether chosen subclassification task is for genre, styles or artists')
    parser.add_argument('--hidden_layer_size', type=int, help= 'size of hidden layer in clf model', default=200)
    parser.add_argument('--epochs', type=int, help="how many epochs to run the model for", default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--model_names', nargs='+', help='list of models to run classification task with')
    parser.add_argument('--savefile_suffix', type=str, help='suffix to add to saved files to identify classification task')
    args = vars(parser.parse_args())
    
    return args

def remap_features(ds_original, ds_filtered, label):

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

def filter_for_illustrations(ds, label, subclassification_task):
    ds = ds.add_column('old_indices', range(len(ds)))

    # find the rows that matches the subclassification task
    subclass_indices = [idx for idx, a in enumerate(ds[f'{label}_str']) if a in subclassification_task]
    ds_subset = ds.select(subclass_indices)

    # remap labels to fit to new number of classes for subclassification task
    ds_subset = remap_features(ds, ds_subset, label)

    # also remap the genre label
    ds_subset = remap_features(ds, ds_subset, 'genre')

    # create test set with only illustrations
    illustrations = ['sketch_and_study', 'illustration']
    illustration_indices = [idx for idx, a in enumerate(ds_subset[f'genre_str']) if a in illustrations]
    ds_test = ds_subset.select(illustration_indices)

    # train set with everything else
    all_genres_indices = set(range(len(ds_subset)))
    train_indices = list(all_genres_indices - set(illustration_indices))
    ds_train = ds_subset.select(train_indices)

    ds_splits = {
        'train': ds_train,
        'test': ds_test
    }

    return ds_splits

def main():

    L.seed_everything(2830)

    # parse command line arguments
    args = argument_parser()

    # load data
    data_name = args['dataset']

    # load full dataset with all images from disk
    ds_full = datasets.load_from_disk(os.path.join('data', data_name))

    # subset dataset based on chosen subclassification task
    classification_task = args['subclasses']
    batch_size = args['batch_size']
    label = args['subclass_label']

    ds_splits = filter_for_illustrations(ds_full, label, classification_task)

# loop over model(s) to be tested for the classification task
    for model_name in args['model_names']:
        # create dataloaders
        full_embedding_pt = torch.load(os.path.join('data', 'filtered_embeddings_FINAL', model_name, f'{model_name}_all_splits.pt'))
        train_loader, inp_size = create_dataloader(ds_splits, full_embedding_pt, label, 'train', batch_size, 'old_indices')
        test_loader, _ = create_dataloader(ds_splits, full_embedding_pt, label, 'test', batch_size, 'old_indices')

        # create model
        model_architecture = build_model(args['hidden_layer_size'], label, inp_size, 0.3, ds_splits)

        # define class weights
        class_weights = define_class_weights(ds_splits, label)

        # define lightning model
        model = SubclassModel(model_architecture, class_weights, lr=args['lr'], weight_decay=0.01, num_classes=ds_splits['train'].features[label].num_classes)

        # set callback & early stopping:
        #check_path = os.path.join('out', 'subclassification_checkpoints')
        #os.makedirs(check_path, exist_ok=True)
        #checkpoint_callback = ModelCheckpoint(
                                    #dirpath=os.path.join(check_path, model_name),
                                    #monitor="val_loss",
                                    #filename=args['savefile_suffix']+"-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
                                    #save_top_k=2,
                                    #mode="min",
                                    #)
        
        #early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=False)

        # fit model
        trainer = L.Trainer(
                    max_epochs=args['epochs'],
                    #callbacks=[checkpoint_callback],
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices="auto",
                    deterministic=True
                    )
        
        trainer.fit(model, train_loader)

        # test
        trainer.test(model, test_loader)

        # + predict
        all_preds_batches = trainer.predict(model, test_loader)
        all_preds = torch.cat(all_preds_batches).cpu().numpy()

        # save classification report + confusion matrix
        save_results(ds_splits['test'], all_preds, model_name, label, args['savefile_suffix'])

        # trainer needs to run in the main script!!!

if __name__ == '__main__':
    main()