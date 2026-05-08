"""
Artist classification for WikiData dataset
"""

import datasets
import numpy as np
import os
import pandas as pd 
import torch
import os
from torch import nn
from torch.utils.data import Dataset
import lightning as L
from torch.utils.data import DataLoader
from torchmetrics.classification import ConfusionMatrix
import argparse
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

from subclassification import filter_data, create_dataloader, build_model, define_class_weights, save_conf_matrix, save_results
from classify_augmentations import SubclassModel

# define argument parser
def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='name of huggingface dataset')
    parser.add_argument('--hidden_layer_size', type=int, help= 'size of hidden layer in clf model', default=200)
    parser.add_argument('--epochs', type=int, help="how many epochs to run the model for", default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--model_names', nargs='+', help='list of models to run classification task with')
    parser.add_argument('--embedding_folder_name', type=str, help='Name of folder in data/ containing the embeddings')

    args = vars(parser.parse_args())
    return args 

def save_conf_matrix(model_name: str, y_true: list | np.ndarray, y_pred: list | np.ndarray, labels: list, task_name: str) -> None:

    """
    Compute and save a confusion matrix plot to disk. 

    The matrix is normalized by true labels (i.e., each row sums to 1),
    and saved as a PNG under out/subclassification_conf_matrices/.

    Args:
        model_name: Name of the model, used as part of the output filename.
        y_true:     True class indices.
        y_pred:     Predicted class indices.
        labels:     Class label names, ordered by class index.
        task_name:  Name of the task, used as part of the output filename
    """

    # convert y-true and y-pred to tensors
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    # create confusion matrix and add to plot
    num_labels = len(labels)
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_labels, normalize="true")
    confmat(y_pred, y_true)
    fig, ax = confmat.plot(add_text = True, labels = labels, cmap='winter')

    # save to out folder
    os.makedirs(os.path.join('out', 'subclassification_conf_matrices'), exist_ok=True)
    out_path = os.path.join("out", "subclassification_conf_matrices", f'{model_name}_{task_name}_confusion_matrix.png')
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

def plot_misclassifications(model_name: str, y_true: list | np.ndarray, y_pred: list | np.ndarray, test_data: datasets.Dataset, task_name: str, label_col: str, num_examples: int = 20) -> None:

    """
    Plot a random sample of misclassified images and save the plot to disk.

    Each subplot shows the image with its true and predicted label. The output
    is saved as a PNG under out/misclassified_examples_subclassifications/.

    Args:
        model_name:   Name of the model, used as part of the output filename.
        y_true:       True class indices.
        y_pred:       Predicted class indices.
        test_data:    HuggingFace Dataset containing an 'image' column and a
                      ClassLabel feature for label_col.
        task_name:    Name of the task, used in the plot title and filename.
        label_col:    Name of the label column in test_data, must be a ClassLabel
                      feature.
        num_examples: Maximum number of misclassified examples to display.
                      Defaults to 20.
    """

    # get indices of misclassifications
    misclass_indices = np.where(np.array(y_true) != np.array(y_pred))[0]

    # select random sample of these
    selected_indices = random.sample(list(misclass_indices), min(num_examples, len(misclass_indices)))

    # determine grid size
    cols = min(5, len(selected_indices))  # max 5 images per row
    rows = math.ceil(len(selected_indices) / cols)

    # plot the images
    plt.figure(figsize=(cols * 3, rows * 3))

    for i, idx in enumerate(selected_indices):
        img = test_data[idx]['image']  # image col must be PIL.Image
        true_label = test_data.features[label_col].int2str(int(y_true[idx]))
        pred_label = test_data.features[label_col].int2str(int(y_pred[idx]))

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"T: {true_label}\nP: {pred_label}", fontsize=10)

    plt.suptitle(f"{task_name}")
    plt.tight_layout()
    
    # save to disk
    os.makedirs(os.path.join('out', 'misclassified_examples_subclassifications'), exist_ok=True)
    save_path = os.path.join('out', 'misclassified_examples_subclassifications', f"{model_name}_{task_name}_misclassified.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_model_scores(model_scores: dict, savefile_suffix: str) -> None:

    """
    Summarize per-fold scores across models and save the results table to disk.

    For each model, computes the mean and standard deviation across folds for
    accuracy, precision, recall, and macro F1. Prints the summary table and
    saves it as a text file under out/wikidata_clf_results/.

    Args:
        model_scores:    A dict that maps model names to a list of per-fold score
                        dicts, each containing keys 'acc', 'precision',
                        'recall', and 'f1'.
        savefile_suffix: String appended to the output filename, e.g. 'WIKIDATA'.
    """
    rows = []

    for model_name, scores in model_scores.items():
        
        df = pd.DataFrame(scores)

        row = {
            "model": model_name,
            "accuracy": f"{df['acc'].mean():.3f} ({df['acc'].std():.3f})",
            "precision": f"{df['precision'].mean():.3f} ({df['precision'].std():.3f})",
            "recall": f"{df['recall'].mean():.3f} ({df['recall'].std():.3f})",
            "macro_f1": f"{df['f1'].mean():.3f} ({df['f1'].std():.3f})",
        }

        rows.append(row)

    results_table = pd.DataFrame(rows).set_index("model")
    print(results_table)

    # save table to disk
    save_path = os.path.join('out', 'wikidata_clf_results')
    with open(os.path.join(save_path, f'WIKIDATA_{savefile_suffix}_CV_results.txt'), 'w') as f:
        f.write(results_table.to_string())

def main():

    # set seed for all lightning functions
    L.seed_everything(2830)

    # parse arguments
    args = argument_parser()

    # define label (classifying artists only)
    label = 'artist'

    # create folder for saving results
    os.makedirs(os.path.join('out', 'wikidata_clf_results'), exist_ok=True)

    # save name of dataset
    data_name = args['dataset']

    # load data from disk
    ds = datasets.load_from_disk(os.path.join('data', data_name))

    # create column with string labels instead of integer class labels
    def map_int_to_str(example):
            return {
            'artist_str': ds.features['artist'].int2str(example['artist'])
            }
    
    ds = ds.map(map_int_to_str, batched=False)

    # define subset of painters for classification task:
    subset = [
        'Eugène Louis Boudin',
        'Paul Cézanne',
        'Camille Pissarro',
        'Alfred Sisley',
        'Édouard Manet'
    ]
    
    # this imported function automatically adds an 'old_indices' column to the input dataframe to make sure we keep track of old embedding indices position
    ds_filtered = filter_data(ds, 'artist', subset, 2830, cv=True) # ignore last two parameters, just reusing function from other script

    # save the filtered dataframe to disk
    ds_filtered.save_to_disk(os.path.join('data', f"{data_name}_impr_subset"))

    # set batch size based on command-line arguments
    batch_size = args['batch_size']

    # define stratified k-fold split with k=5
    skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2830)

    labels = np.array(ds_filtered[label])
    indices = np.arange(len(ds_filtered))

    # create dictionary of model names to append per-fold scores to (we average over them later)
    model_scores = {m: [] for m in args["model_names"]}

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):

        # monitor folds
        print(f"Fold {fold+1}")

        # select train and test set for fold
        ds_train = ds_filtered.select(train_idx.tolist())
        ds_test = ds_filtered.select(val_idx.tolist())

        ds_splits_for_cv = {
                'train': ds_train,
                'test': ds_test}

        # for this split, for each model, fit on train data and predict on test
        for model_name in args['model_names']:

            # load wikidata embeddings from disk
            full_embedding_pt = torch.load(os.path.join('data', args['embedding_folder_name'], model_name, f'{model_name}_all_splits.pt'))
            train_loader, inp_size = create_dataloader(ds_splits_for_cv, full_embedding_pt, label, 'train', batch_size, 'old_indices')
            test_loader, _ = create_dataloader(ds_splits_for_cv, full_embedding_pt, label, 'test', batch_size, 'old_indices')

            # build pytorch lightning model and define class weights to account for class imbalances
            model_architecture = build_model(args['hidden_layer_size'], label, inp_size, 0.3, ds_splits_for_cv)
            class_weights = define_class_weights(ds_splits_for_cv, label)

            # set up model
            model = SubclassModel(model_architecture, class_weights, lr=args['lr'], weight_decay=0.01, num_classes = ds_splits_for_cv['train'].features[label].num_classes)

            # define trainer
            trainer = L.Trainer(
                max_epochs=args['epochs'],
                #callbacks=[checkpoint_callback],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices="auto",
                deterministic=True
            )
            
            # fit model
            trainer.fit(model, train_loader)

            # get all predictions, batched
            all_preds_batches = trainer.predict(model, test_loader)

            # concatenate all batched predictions + probabilities
            all_preds = torch.cat([b[0] for b in all_preds_batches]).cpu().numpy()
            all_probs = torch.cat([b[1] for b in all_preds_batches]).cpu().numpy()

            # convert y-true and y-pred to tensors
            y_true = torch.tensor(ds_splits_for_cv['test'][label])
            all_preds_tensor = torch.tensor(all_preds)

            # save n_classes
            num_classes = ds_splits_for_cv['train'].features[label].num_classes
            
            # define macro F1, precision and recall as evaluation metrics
            precision_fn = MulticlassPrecision(num_classes=num_classes, average="macro")
            recall_fn = MulticlassRecall(num_classes=num_classes, average="macro")
            f1_fn = MulticlassF1Score(num_classes=num_classes, average="macro")

            # append to dict of scores for model
            model_scores[model_name].append({
                "acc": (all_preds_tensor == y_true).float().mean().item(),
                "precision": precision_fn(all_preds_tensor, y_true).item(),
                "recall": recall_fn(all_preds_tensor, y_true).item(),
                "f1": f1_fn(all_preds_tensor, y_true).item(),
            }) 

            # save some plots for last fold ONLY for demonstration purposes
            if fold == 4:
                    save_results(
                        test_data = ds_splits_for_cv['test'],
                        y_pred = all_preds,
                        model_name = model_name,
                        label_col = label,
                        task_name = f"WIKIDATA_{fold+1}"
                    )

            # clean up
            del full_embedding_pt, model, test_loader, train_loader
        
        del ds_splits_for_cv, ds_train, ds_test

    # average scores across splits
    save_model_scores(model_scores, 'WIKIDATA')

if __name__ == '__main__':
    main()

