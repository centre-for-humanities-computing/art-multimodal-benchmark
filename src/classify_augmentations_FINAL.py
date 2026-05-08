"""
This script takes an input an AUGMENTED SUBSET of wikiart; (wikiart_filtered_remapped_FINAL_AUG_SUBSET) 
"""

import datasets
import numpy as np
import os
import torch
import pandas as pd
from torch import optim, nn
from torch.utils.data import Dataset
import lightning as L
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torchmetrics.classification import ConfusionMatrix
import argparse
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

# import from own script
from subclf_updated import create_dataloader, build_model, define_class_weights, filter_data

# define arguments
def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='name of huggingface dataset')
    parser.add_argument('--hidden_layer_size', type=int, help= 'size of hidden layer in clf model', default=200)
    parser.add_argument('--epochs', type=int, help="how many epochs to run the model for", default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--model_names', nargs='+', help='list of models to run classification task with')

    args = vars(parser.parse_args())
    return args 

def create_test_loader(test_data: datasets.Dataset, test_embeddings: torch.Tensor, label: str, batch_size: int) -> DataLoader:

    """
    Create a DataLoader for test embeddings.

    Wraps pre-computed embeddings and their corresponding labels into a
    DataLoader with no shuffling

    Args:
        test_data:       HuggingFace Dataset containing the label column.
        test_embeddings: Embeddings tensor of shape (N, embedding_dim).
        label:           Name of the label column in test_data.
        batch_size:      Number of samples per batch.

    Returns:
        A DataLoader yielding (embedding, label) pairs.
    """

    # define custom embeddings dataset that just has embeddings + corresponding labels
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

    # create embeddings dataset and wrap in dataloader
    dataset = EmbeddingsDataset(embeddings_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader

def save_conf_matrix(model_name: str, y_true: list | np.ndarray, y_pred: list | np.ndarray, labels: list, aug_name: str, save_folder: str) -> None:
    """
    Compute and save a confusion matrix plot to disk. 

    The matrix is normalized by true labels (i.e., each row sums to 1),
    and saved as a PNG.

    Args:
        model_name: Name of the model, used as part of the output filename.
        y_true:     True class indices.
        y_pred:     Predicted class indices.
        labels:     Class label names, ordered by class index.
        aug_name:   Name of the augmentation applied, used in the output filename.
        save_folder: Directory to save the plot.

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
    out_path = os.path.join(save_folder, f'{model_name}_{aug_name}_confusion_matrix.png')
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

def save_results(test_data: datasets.Dataset, y_pred: np.ndarray, model_name: str, label_col: str, aug_name: str, save_folder: str) -> None:

    """
    Save a classification report and confusion matrix for a given model and augmentation.

    Generates a per-class classification report (precision, recall, F1) and a
    normalized confusion matrix, saving both to save_folder.

    Args:
        test_data:   HuggingFace Dataset containing the ground truth label column,
                     which must be a ClassLabel feature.
        y_pred:      Predicted class indices.
        model_name:  Name of the model, used in output filenames.
        label_col:   Name of the label column in test_data.
        aug_name:    Name of the augmentation applied, used in output filenames.
        save_folder: Directory to save the classification report and confusion matrix.
    """

    # get label strings
    labels = np.unique(test_data[label_col])
    target_names = [test_data.features[label_col].int2str(int(i)) for i in labels]

    # save classification report for y_true and y_pred
    report = classification_report(np.array(test_data[label_col]),
                           y_pred, target_names = target_names)
    
    out_path = os.path.join(save_folder, f'{model_name}_{aug_name}_classification_report.txt')

    with open(out_path, 'w') as file:
                file.write(report)

    # save confusion matrix as well:
    save_conf_matrix(model_name, np.array(test_data[label_col]), y_pred, target_names, aug_name, save_folder)

# save mean results across augmentations, across CV folds, for each model
def aggregate_results(model_scores: dict) -> None:

    """
    Summarize per-fold scores across models and save the results table to disk.

    For each model and each augmentation, computes the mean and standard deviation across folds for
    accuracy, precision, recall, and macro F1. Prints the summary table and
    saves it as a text file under out/test_augmentation_results/.

    Args:
        model_scores:    A dict that maps model names to a list of per-fold score
                        dicts, each containing keys 'acc', 'precision',
                        'recall', and 'f1'.
    """
    rows = []

    for model_name, aug_dict in model_scores.items():
        for aug_name, scores in aug_dict.items():

            df = pd.DataFrame(scores)

            row = {
                "model": model_name,
                "augmentation": aug_name,
                "accuracy": f"{df['acc'].mean():.3f} ({df['acc'].std():.3f})",
                "precision": f"{df['precision'].mean():.3f} ({df['precision'].std():.3f})",
                "recall": f"{df['recall'].mean():.3f} ({df['recall'].std():.3f})",
                "f1": f"{df['f1'].mean():.3f} ({df['f1'].std():.3f})",
            }

            rows.append(row)

    results_table = pd.DataFrame(rows).set_index("model")
    print(results_table)

    # Save results_table to a text file
    with open(os.path.join('out', 'test_augmentation_results', 'cv_results_augmentations.txt'), 'w') as f:
        f.write(results_table.to_string())

class SubclassModel(L.LightningModule):

    """
    A LightningModule for multiclass classification using pre-computed embeddings.

    The model uses a weighted cross-entropy loss, Adam optimizer,
    and an exponential learning rate scheduler. Logs loss and accuracy at each
    step and epoch during training and validation.

    """
    def __init__(self, model, class_weights, lr, weight_decay, num_classes):

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
        preds = torch.argmax(logits, dim=1) # argmax on logits; get the most probable class
        probs = torch.softmax(logits, dim=1) # softmax on logits; get probabilities of each class
        return preds, probs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # set gamma or make changeble parameter?

        return { # has to be returned in a specific format
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",},
                }

def main():

    # set seed for all lightning functions
    L.seed_everything(2830)

    # parse args
    args = argument_parser()

    # load data
    data_name = args['dataset']

    # load full dataset with all images from disk
    ds = datasets.load_from_disk(os.path.join('data', data_name))

    # set batch size
    batch_size = args['batch_size']

    # only classify artist label
    label = 'artist'

    # specify list of applied augmentations:
    augmentations = ['strong_blur',
                    'grayscale', 
                    'contrast', 
                    'frame', 
                    'jpeg_compr', 
                    'vignette', 
                    'weak_grain', 
                    'light_artifact', 
                    'Canny_sketch', 
                    'pencil_sketch'
                    ]

    # initialize model_scores dictionary
    model_scores = {}

    for model_name in args['model_names']:
        # Start with all augmentations
        aug_dict = {aug: [] for aug in augmentations}
        # Add the "no augmentation" baseline
        aug_dict['no_aug'] = []
        # Assign to the model
        model_scores[model_name] = aug_dict
    
    # initialize 5-fold cross validation
    skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2830)
    y = np.array(ds[label])
    indices = np.arange(len(ds))

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, y)):

        print(f"Fold {fold+1}")

        ds_train = ds.select(train_idx.tolist())
        ds_test = ds.select(val_idx.tolist())

        ds_splits_for_cv = {
                            'train': ds_train,
                            'test': ds_test}
        
        # for this split, for each model, fit a single model on the non-augmented train data, and predict on a test set for each augmentation (so same image, but augmented differently)
        for model_name in args['model_names']:
            os.makedirs(os.path.join('out', 'test_augmentation_results'), exist_ok=True)

            out_folder = os.path.join('out', 'test_augmentation_results', model_name)
            os.makedirs(out_folder, exist_ok=True)

            # load full, original embedding tensor
            full_embedding_pt = torch.load(os.path.join('data', 'filtered_embeddings_FINAL', model_name, f'{model_name}_all_splits.pt'))

            # get non-augmented embeddings with train loader split 
            train_loader, inp_size = create_dataloader(ds_splits_for_cv, full_embedding_pt, label, 'train', batch_size, 'old_indices') # i can use the old_indices column because i have not reset it

            # define model architecture
            model_architecture = build_model(args['hidden_layer_size'], label, inp_size, 0.3, ds_splits_for_cv)

            # define class weights
            class_weights = define_class_weights(ds_splits_for_cv, label)

            # define lightning model
            model = SubclassModel(model_architecture, class_weights, lr=args['lr'], weight_decay=0.01, num_classes = ds_splits_for_cv['train'].features[label].num_classes)

            # define PyTorch lightning trainer
            trainer = L.Trainer(
                            max_epochs=args['epochs'],
                            accelerator="gpu" if torch.cuda.is_available() else "cpu",
                            devices="auto",
                            deterministic=True
                                )

            # fit on training data (un-augmented embeddings)
            trainer.fit(model, train_loader)

            # test on augmented embeddings of test split
            for aug in augmentations:

                print(f"PREDICTING ON {aug} TEST DATA")

                # load full augmentation embedding 
                full_aug_embeddings = torch.load(os.path.join('data', 'aug_embeddings', aug, f'{model_name}.pt'))

                # filter for test-indices
                test_aug_embeddings = full_aug_embeddings[val_idx] # filter embeddings based on cv-fold indices

                # need to get the indices of only the test set 
                # this is not doing any filtering, simply creating a dataloader with shuffle=false with embeddings + labels
                test_loader = create_test_loader(ds_splits_for_cv['test'], test_aug_embeddings, label, batch_size)

                # predict on test data
                test_metrics = trainer.test(model, test_loader)

                all_preds_batches = trainer.predict(model, test_loader)

                # get both probabilities of each class + argmax prediction  
                # need to do torch.cat because all_preds_batches is returned per batch, not for the entire test-set
                all_preds = torch.cat([b[0] for b in all_preds_batches]).cpu().numpy()
                all_probs = torch.cat([b[1] for b in all_preds_batches]).cpu().numpy()

                y_true = torch.tensor(ds_splits_for_cv['test'][label])
                all_preds_tensor = torch.tensor(all_preds) # make predictions into tensor

                # define and save macro-averaged evaluation metrics
                num_classes = ds_splits_for_cv['train'].features[label].num_classes
                
                precision_fn = MulticlassPrecision(num_classes=num_classes, average="macro")
                recall_fn = MulticlassRecall(num_classes=num_classes, average="macro")
                f1_fn = MulticlassF1Score(num_classes=num_classes, average="macro")

                model_scores[model_name][aug].append({
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
                        aug_name = f"{aug}_fold{fold+1}_WIKIDATA",
                        save_folder=out_folder
                    )

                    # Save probabilities
                    np.save(
                        os.path.join(out_folder, f'{model_name}_{aug}_fold{fold+1}_probs.npy'),
                        all_probs
                    )

                # clean up
                del full_aug_embeddings
                del test_aug_embeddings
                del test_loader
                del test_metrics
                del all_preds_batches
                del all_preds

                import gc
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # test model without augmentations on the same test-split
            test_loader, _ = create_dataloader(ds_splits_for_cv, full_embedding_pt, label, 'test', batch_size, 'old_indices') # 
            test_metrics = trainer.test(model, test_loader)

            all_preds_batches = trainer.predict(model, test_loader)
            all_preds = torch.cat([b[0] for b in all_preds_batches]).cpu().numpy()
            all_probs = torch.cat([b[1] for b in all_preds_batches]).cpu().numpy()

            y_true = torch.tensor(ds_splits_for_cv['test'][label])
            all_preds_tensor = torch.tensor(all_preds)

            num_classes = ds_splits_for_cv['train'].features[label].num_classes
            
            precision_fn = MulticlassPrecision(num_classes=num_classes, average="macro")
            recall_fn = MulticlassRecall(num_classes=num_classes, average="macro")
            f1_fn = MulticlassF1Score(num_classes=num_classes, average="macro")

            model_scores[model_name]['no_aug'].append({
                "acc": (all_preds_tensor == y_true).float().mean().item(),
                "precision": precision_fn(all_preds_tensor, y_true).item(),
                "recall": recall_fn(all_preds_tensor, y_true).item(),
                "f1": f1_fn(all_preds_tensor, y_true).item(),
            })

            if fold == 4:
                save_results(
                    test_data = ds_splits_for_cv['test'],
                    y_pred = all_preds,
                    model_name = model_name,
                    label_col = label,
                    aug_name = f"no_aug_fold{fold+1}_WIKIDATA",
                    save_folder=out_folder
                )

                # Save probabilities
                np.save(
                    os.path.join(out_folder, f'{model_name}_NO_AUG_fold{fold+1}_probs.npy'),
                    all_probs
                )

            # clean up
            del full_embedding_pt
            del model 
            del trainer
            del all_preds_batches
            del all_probs
            del all_preds_tensor
            del test_loader

    # save results across folds
    aggregate_results(model_scores)

if __name__ == '__main__':
    main()














