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
    ..

    return args 


# create function for each augmentation

def main():
    L.seed_everything(2830)

    args = argument_parser()

    # load data
    data_name = args['dataset']

    # load full dataset with all images from disk
    ds_full = datasets.load_from_disk(os.path.join('data', data_name))

    batch_size = args['batch_size']

    skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2830)
    labels = np.array(ds_full[label])
    indices = np.arange(len(ds_full))

    model_scores = {m: [] for m in args["model_names"]}

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):

        # monitor folds
            print(f"Fold {fold+1}")

            ds_train = ds_splits.select(train_idx.tolist())
            ds_test = ds_splits.select(val_idx.tolist())

            ds_splits_for_cv = {
                'train': ds_train,
                'test': ds_test}

            





