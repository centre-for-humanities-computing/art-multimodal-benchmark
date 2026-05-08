"""
Classify based on chosen 'subclassification', e.g., only using certain artists in the dataset.
"""

import datasets
import numpy as np
import os
from datasets import ClassLabel
import pandas as pd 
import torch
import os
from torch import optim, nn
from torch.utils.data import Dataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
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
    parser.add_argument('--cv', action=argparse.BooleanOptionalAction, help='whether to run cross-validation', default=False)
    args = vars(parser.parse_args())
    
    return args

def remap_features(ds_original: datasets.Dataset, ds_filtered: datasets.Dataset, label: str) -> datasets.Dataset:
    '''
    Resets ClassLabel mappings after filtering a HuggingFace dataset. 
    When a HuggingFace dataset with a ClassLabel is filtered, it still saves the original Class mapping - even if some classes are filtered out.
    This function resets that.

    Args:
        ds_original: The original unfiltered dataset, used to access the original ClassLabel feature.
        ds_filtered: The filtered dataset whose ClassLabel mapping needs to be reset.
        label: The name of the ClassLabel column (e.g. 'artist' or 'genre').

    Returns:
        The filtered dataset with a corrected ClassLabel mapping.
    '''
    # get original classlabel features
    original_feature = ds_original.features[label]
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

def filter_data(ds: datasets.Dataset, label: str, subclassification_task: list, seed: int, cv: bool) -> datasets.Dataset:
    """
    Filters a dataset to a subset of classes and remaps its ClassLabel feature.

    Selects only the rows whose label (as a string) is in the filtering parameter,
    then remaps the ClassLabel feature to reflect only the remaining classes.

    Args:
        ds: The full dataset to filter.
        label: The name of the ClassLabel column to filter on (e.g. 'artist' or 'genre').
        subclassification_task: The class names to keep.
        seed: unused here
        cv: unused here

    Returns:
        The filtered dataset with remapped ClassLabel and an added 'old_indices' column
        tracking each row's position in the original dataset.
    """

    ds = ds.add_column('old_indices', range(len(ds)))

    # find the rows that matches the subclassification task
    subclass_indices = [idx for idx, a in enumerate(ds[f'{label}_str']) if a in subclassification_task]
    ds_subset = ds.select(subclass_indices)

    # remap labels to fit to new number of classes for subclassification task
    ds_subset = remap_features(ds, ds_subset, label)

    if cv==True:
        #ds_split = ds_subset.train_test_split(test_size=0.2, seed=seed, stratify_by_column=label)

        #ds_splits = {
         #   'train': ds_split['train'], # train/val set
          #  'test': ds_split['test'] # hold-out test set - we're not touching this until the end
          #  }
        return ds_subset
    
    else:
        # split into train, val and test: 
        ds_split = ds_subset.train_test_split(test_size=0.3, seed=seed, stratify_by_column = label)
        ds_train = ds_split['train']
        ds_test = ds_split['test']

        # split test data into test and validation
        ds_test_split = ds_test.train_test_split(test_size=2/3, seed=seed, stratify_by_column = label)
        ds_val = ds_test_split['train']
        ds_test = ds_test_split['test']

        ds_splits = {
                'train': ds_train,
                'test': ds_test,
                'val': ds_val}

    return ds_splits

# DATALOADERS
def create_dataloader(ds_splits: dict, full_embedding_pt: torch.Tensor, label: str, split: str, batch_size: int, idx_column: str) -> tuple:
   
    """
    Creates a DataLoader for a dataset split using precomputed embeddings.

    Selects the relevant embeddings from a full embedding tensor using index
    values stored in idx_column, pairs them with their labels, and wraps
    them in a DataLoader. Training splits are shuffled, others are not.

    Args:
        ds_splits: A dict mapping split names to their HuggingFace datasets.
        full_embedding_pt: Tensor of precomputed embeddings for all samples,
                           indexed by the values in idx_column.
        label: The ClassLabel column to use as targets (e.g. 'artist').
        split: The dataset split to load, e.g. 'train', 'val', or 'test'.
        batch_size: Number of samples per batch.
        idx_column: Column in the dataset containing the indices into
                    full_embedding_pt (e.g. 'old_indices').

    Returns:
        A tuple of (dataloader, embedding_size)
    """
    class EmbeddingsDataset(Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = embeddings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]

    # load full embedding and split based on correct indices
    split_indices = ds_splits[split][idx_column]

    filtered_embeddings = full_embedding_pt[split_indices] # grab embeddings based on column VALUES not positions

    # cast to float32
    embeddings_tensor = filtered_embeddings.float()

    y = ds_splits[split][label]
    labels_tensor = torch.tensor(y)

    shuffle=False

    if split == 'train':
        shuffle=True

    dataset = EmbeddingsDataset(embeddings_tensor, labels_tensor)

    # input to data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) # set shuffle=True for train

    embedding_size = embeddings_tensor.shape[1]

    return dataloader, embedding_size

def build_model(hidden_layer_size: int, label: str, inp_size: int, dropout_p: int, ds_splits: dict) -> nn.Sequential:

    """
    Builds an MLP classifier.

    Args:
        hidden_layer_size: Number of units in the hidden layer.
        label: The ClassLabel column being classified, used to determine
               the number of output classes.
        inp_size: Dimensionality of the input embeddings.
        dropout_p: Dropout probability.
        ds_splits: A dict mapping split names to their HuggingFace datasets.
                   The train split is used to determine the number of classes.

    Returns:
        MLP model
    """
    num_classes = ds_splits['train'].features[label].num_classes

    model = nn.Sequential(
        nn.Linear(in_features=inp_size, out_features=hidden_layer_size),
        nn.ReLU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features=hidden_layer_size, out_features=num_classes)
            )

    return model 

def define_class_weights(ds_splits: dict, label: str) -> torch.Tensor:

    """
    Computes normalized inverse-frequency class weights from the training split.

    Weights are computed as 1 / class_count and then normalized to sum to 1,
    giving higher weight to underrepresented classes. Intended for use with
    nn.CrossEntropyLoss(weight=...) to handle class imbalance.

    Args:
        ds_splits: A dict mapping split names to their HuggingFace datasets.
                   Only the train split is used.
        label: The ClassLabel column to compute weights for.

    Returns:
        A tensor of normalized class weights, one per class.
    
    """

    y_tensor = torch.tensor(ds_splits['train'][label])
    class_counts = torch.bincount(y_tensor)
    class_weights = 1.0 / class_counts.float() # weight the loss inversely proportional to class frequency
    class_weights /= class_weights.sum() # normalize weights so they sum to one

    return class_weights

class SubclassModel(L.LightningModule):

    """
    A LightningModule for multiclass classification using pre-computed embeddings.

    The model uses a weighted cross-entropy loss, Adam optimizer,
    and an exponential learning rate scheduler. Logs loss and accuracy at each
    step and epoch during training and validation.
    """
    def __init__(self, model, class_weights, lr, weight_decay, num_classes): # options to set some default parameters here

        super().__init__()

        self.model = model
        self.lr = lr 
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        # buffer makes sure that class weights moves automatically to GPU
        self.register_buffer('class_weights', class_weights)
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch 
        output = self(X)
        loss = self.loss_fn(output, y)
        acc = (output.argmax(1) == y).float().mean()

        # log the training loss and accuracy
        # Log the loss at each training step and epoch, create a progress bar
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) # logged per-epoch level
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss 
    
    # lightning automatically runs testing + validation with torch.no_grad() and model.eval()
    def validation_step(self, batch, batch_idx):
        X, y = batch 
        output = self(X)
        loss = self.loss_fn(output, y)

        preds = output.argmax(1)

        acc = (preds == y).float().mean()

        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        X, y = batch 
        output = self(X)
        loss = self.loss_fn(output, y)

        preds = output.argmax(1)

        acc = (preds == y).float().mean()

        self.log('test_loss', loss)
        self.log('test_acc', acc) 
    
    def predict_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        return preds, probs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        return { # has to be returned in a specific format
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",},
                }

def save_conf_matrix(model_name: str, y_true: list | np.ndarray, y_pred: list | np.ndarray, labels: list, task_name: str):

    """
    Compute and save a confusion matrix plot to disk. 

    The matrix is normalized by true labels (i.e., each row sums to 1),
    and saved as a PNG.

    Args:
        model_name: Name of the model, used as part of the output filename.
        y_true:     True class indices.
        y_pred:     Predicted class indices.
        labels:     Class label names, ordered by class index.
        task_name: name of subclassification task, to be used for save files.

    """
    # convert y true and y pred to tensors
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    # create confusion matrix
    num_labels = len(labels)
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_labels, normalize="true")
    confmat(y_pred, y_true)
    
    # plot
    fig, ax = confmat.plot(add_text = True, labels = labels, cmap='winter')

    # save plot to disk
    os.makedirs(os.path.join('out', 'subclassification_conf_matrices'), exist_ok=True)
    out_path = os.path.join("out", "subclassification_conf_matrices", f'{model_name}_{task_name}_confusion_matrix.png')
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

def plot_misclassifications(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, test_data: datasets.Dataset, task_name: str, label_col: str, num_examples: int = 20):

    """
    Saves a plot of randomly sampled misclassified images with their true and predicted labels.

    Output is saved to out/misclassified_examples_subclassifications/<model_name>_<task_name>_misclassified.png.

    Args:
        model_name: Name of the embedding model, used when naming the output file.
        y_true: Array of true class indices.
        y_pred: Array of predicted class indices.
        test_data: HuggingFace dataset containing the test images and label feature.
        task_name: Name of the classification task, used in the plot title and output filename.
        label_col: The ClassLabel column name, used to convert indices to label strings.
        num_examples: Maximum number of misclassified examples to plot. Defaults to 20.
    """

    misclass_indices = np.where(np.array(y_true) != np.array(y_pred))[0]
    selected_indices = random.sample(list(misclass_indices), min(num_examples, len(misclass_indices)))

    # determine grid size
    cols = min(5, len(selected_indices))  # max 5 images per row
    rows = math.ceil(len(selected_indices) / cols)

    # plot the images
    plt.figure(figsize=(cols * 3, rows * 3))

    for i, idx in enumerate(selected_indices):
        img = test_data[idx]['image']  # assume PIL.Image
        true_label = test_data.features[label_col].int2str(int(y_true[idx]))
        pred_label = test_data.features[label_col].int2str(int(y_pred[idx]))

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"T: {true_label}\nP: {pred_label}", fontsize=10)

    plt.suptitle(f"{task_name}")
    plt.tight_layout()
    
    os.makedirs(os.path.join('out', 'misclassified_examples_subclassifications'), exist_ok=True)
    save_path = os.path.join('out', 'misclassified_examples_subclassifications', f"{model_name}_{task_name}_misclassified.png")

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_results(test_data: datasets.Dataset, y_pred: np.ndarray, model_name: str, label_col: str, task_name: str) -> None:

    """
    Saves a classification report, confusion matrix, and misclassification plot for a model.

    Output files are saved to:
        - out/subclassification_reports/<model_name>_<task_name>_subclassification_report.txt
        - out/subclassification_conf_matrices/<model_name>_<task_name>_confusion_matrix.png
        - out/misclassified_examples_subclassifications/<model_name>_<task_name>_misclassified.png

    Args:
        test_data: HuggingFace dataset containing the test images and true labels.
        y_pred: Array of predicted class indices.
        model_name: Name of the embedding model, used when naming output files.
        label_col: The ClassLabel column name used as the classification target.
        task_name: Name of the classification task, used when naming output files.
    """
    labels = np.unique(test_data[label_col])
    target_names = [test_data.features[label_col].int2str(int(i)) for i in labels]

    # save classification report for y_true and y_pred
    report = classification_report(np.array(test_data[label_col]),
                           y_pred, target_names = target_names)
    
    # save classification report
    os.makedirs(os.path.join('out', 'subclassification_reports'), exist_ok=True)
    out_path = os.path.join("out", "subclassification_reports", f'{model_name}_{task_name}_subclassification_report.txt')

    with open(out_path, 'w') as file:
                file.write(report)

    # save confusion matrix as well:
    save_conf_matrix(model_name, np.array(test_data[label_col]), y_pred, target_names, task_name)

    # save examples of misclassified images
    plot_misclassifications(model_name, np.array(test_data[label_col]), y_pred, test_data, task_name, label_col)

def aggregate_results(model_scores: dict, savefile_suffix: str) -> None:

    """
    Aggregates cross-validation scores across folds and saves a summary table to disk.

    Args:
        model_scores: A dict mapping model names to a list of per-fold score dicts,
                      each containing 'acc', 'precision', 'recall', and 'f1'.
        savefile_suffix: Suffix used when naming the output file.
    """
    rows = []

    for model_name, scores in model_scores.items():
            df = pd.DataFrame(scores)

            row = {
                "model": model_name,
                "accuracy": f"{df['acc'].mean():.3f} ({df['acc'].std():.3f})",
                "precision": f"{df['precision'].mean():.3f} ({df['precision'].std():.3f})",
                "recall": f"{df['recall'].mean():.3f} ({df['recall'].std():.3f})",
                "f1": f"{df['f1'].mean():.3f} ({df['f1'].std():.3f})",
            }

            rows.append(row)

    results_table = pd.DataFrame(rows).set_index("model")
    print(results_table)

    # Save results_table to a text file
    save_path = os.path.join('out', 'subclassification_reports')
    with open(os.path.join(save_path, f'{savefile_suffix}_CV_results.txt'), 'w') as f:
        f.write(results_table.to_string())

def main():

    # set lightning seed
    L.seed_everything(2830)

    # parse command line arguments
    args = argument_parser()

    # load data
    data_name = args['dataset']

    # load full dataset with all images from disk
    ds_full = datasets.load_from_disk(os.path.join('data', data_name))

    # subset dataset based on chosen subclassification task (if cv=True, ds_splits contains only train+test split)
    classification_task = args['subclasses']
    batch_size = args['batch_size']
    label = args['subclass_label']
    ds_splits = filter_data(ds_full, label, classification_task, 2830, cv = args['cv']) 
        
    # cross-validate
    if args['cv']:

        skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2830) # shuffle=true?

        #ds_train_val = ds_splits['train']

        # get labels + indices
        labels = np.array(ds_splits[label])
        indices = np.arange(len(ds_splits))

        model_scores = {m: [] for m in args["model_names"]}

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
                test_loader, _ = create_dataloader(ds_splits_for_cv, full_embedding_pt, label, 'test', batch_size, 'old_indices')

                model_architecture = build_model(args['hidden_layer_size'], label, inp_size, 0.3, ds_splits_for_cv)
                
                # define class weights
                class_weights = define_class_weights(ds_splits_for_cv, label)

                # define lightning model
                model = SubclassModel(model_architecture, class_weights, lr=args['lr'], weight_decay=0.01, num_classes = ds_splits_for_cv['train'].features[label].num_classes)

                # fit model
                trainer = L.Trainer(
                            max_epochs=args['epochs'],
                            accelerator="gpu" if torch.cuda.is_available() else "cpu",
                            devices="auto",
                            deterministic=True
                            )
                
                trainer.fit(model, train_loader)
                test_metrics = trainer.test(model, test_loader)

                # save across folds
                # need to do torch.cat because all_preds_batches is returned per batch, not for the entire test-set
                all_preds_batches = trainer.predict(model, test_loader)
                all_preds = torch.cat([b[0] for b in all_preds_batches]).cpu().numpy()
                all_probs = torch.cat([b[1] for b in all_preds_batches]).cpu().numpy()

                y_true = torch.tensor(ds_splits_for_cv['test'][label])
                all_preds_tensor = torch.tensor(all_preds)

                # define and save macro-averaged evaluation metrics
                num_classes = ds_splits_for_cv['train'].features[label].num_classes
                
                precision_fn = MulticlassPrecision(num_classes=num_classes, average="macro")
                recall_fn = MulticlassRecall(num_classes=num_classes, average="macro")
                f1_fn = MulticlassF1Score(num_classes=num_classes, average="macro")

                model_scores[model_name].append({
                    "acc": (all_preds_tensor == y_true).float().mean().item(),
                    "precision": precision_fn(all_preds_tensor, y_true).item(),
                    "recall": recall_fn(all_preds_tensor, y_true).item(),
                    "f1": f1_fn(all_preds_tensor, y_true).item(),
                })

                # save fold-specific results for demonstration purposes only
                if fold == 4:
                    save_results(
                        test_data = ds_splits_for_cv['test'],
                        y_pred = all_preds,
                        model_name = model_name,
                        label_col = label,
                        task_name = f"{args['savefile_suffix']}_fold{fold+1}"
                    )

                del full_embedding_pt, model, test_loader, train_loader
        
            del ds_splits_for_cv, ds_train, ds_test
        
        # average results across splits
        aggregate_results(model_scores, args['savefile_suffix'])

            
    else:

        # loop over model(s) to be tested for the classification task
        for model_name in args['model_names']:
            # create dataloaders
            full_embedding_pt = torch.load(os.path.join('data', 'filtered_embeddings_FINAL', model_name, f'{model_name}_all_splits.pt'))
            train_loader, inp_size = create_dataloader(ds_splits, full_embedding_pt, label, 'train', batch_size, 'old_indices')
            val_loader, _ = create_dataloader(ds_splits, full_embedding_pt, label, 'val', batch_size, 'old_indices')
            test_loader, _ = create_dataloader(ds_splits, full_embedding_pt, label, 'test', batch_size, 'old_indices')

            # create model
            model_architecture = build_model(args['hidden_layer_size'], label, inp_size, 0.3, ds_splits)

            # define class weights
            class_weights = define_class_weights(ds_splits, label)

            # define lightning model
            model = SubclassModel(model_architecture, class_weights, lr=args['lr'], weight_decay=0.01, num_classes=ds_splits['train'].features[label].num_classes)

            # set callback & early stopping:
            check_path = os.path.join('out', 'subclassification_checkpoints')
            os.makedirs(check_path, exist_ok=True)
            checkpoint_callback = ModelCheckpoint(
                                        dirpath=os.path.join(check_path, model_name),
                                        monitor="val_loss",
                                        filename=args['savefile_suffix']+"-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
                                        save_top_k=2,
                                        mode="min",
                                        )
            
            early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=False)

            # fit model
            trainer = L.Trainer(
                        max_epochs=args['epochs'],
                        callbacks=[checkpoint_callback, early_stopping],
                        accelerator="gpu" if torch.cuda.is_available() else "cpu",
                        devices="auto",
                        deterministic=True
                        )

            trainer.fit(model, train_loader, val_loader)

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
