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

def filter_data(ds, label, subclassification_task, seed, cv):
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
def create_dataloader(ds_splits, full_embedding_pt, label, split, batch_size, idx_column):
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
    #full_embedding_pt = torch.load(os.path.join('data', 'filtered_embeddings_FINAL', model_name, f'{model_name}_all_splits.pt'))

    filtered_embeddings = full_embedding_pt[split_indices]

    # cast to float32
    #embeddings_tensor = filtered_embeddings.float().to(device)
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

def build_model(hidden_layer_size, label, inp_size, dropout_p, ds_splits): # do you need device as well??

    num_classes = ds_splits['train'].features[label].num_classes

    model = nn.Sequential(
        nn.Linear(in_features=inp_size, out_features=hidden_layer_size),
        nn.ReLU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features=hidden_layer_size, out_features=num_classes)
            )

    return model 

def define_class_weights(ds_splits, label):
    y_tensor = torch.tensor(ds_splits['train'][label])
    class_counts = torch.bincount(y_tensor)
    class_weights = 1.0 / class_counts.float() # weight the loss inversely proportional to class frequency
    class_weights /= class_weights.sum() # normalize weights so they sum to one

    return class_weights

class SubclassModel(L.LightningModule):
    def __init__(self, model, class_weights, lr, weight_decay, num_classes): # options to set some default parameters here

        # not really sure what this does:
        super().__init__()

        self.model = model
        self.lr = lr 
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        # buffer makes sure that class weights moves automatically to GPU
        self.register_buffer('class_weights', class_weights)
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

        # define validation metrics
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        # define test metrics
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.test_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
    
    # not exactly sure what this part is
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

        precision = self.val_precision(preds, y)
        recall = self.val_recall(preds, y)
        f1 = self.val_f1(preds, y)

        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_f1', f1)

    def test_step(self, batch, batch_idx):
        X, y = batch 
        output = self(X)
        loss = self.loss_fn(output, y)

        preds = output.argmax(1)

        acc = (preds == y).float().mean()
        precision = self.test_precision(preds, y)
        recall = self.test_recall(preds, y)
        f1 = self.test_f1(preds, y)

        self.log('test_loss', loss)
        self.log('test_acc', acc) 
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_f1', f1)
    
    def predict_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # set gamma or make changeble parameter?

        return { # has to be returned in a specific format
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",},
                }

def save_conf_matrix(model_name, y_true, y_pred, labels, task_name):

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    num_labels = len(labels)
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_labels, normalize="true")
    confmat(y_pred, y_true)
    fig, ax = confmat.plot(add_text = True, labels = labels, cmap='winter')

    os.makedirs(os.path.join('out', 'subclassification_conf_matrices'), exist_ok=True)
    out_path = os.path.join("out", "subclassification_conf_matrices", f'{model_name}_{task_name}_confusion_matrix.png')
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

def plot_misclassifications(model_name, y_true, y_pred, test_data, task_name, label_col, num_examples = 20):

    misclass_indices = np.where(np.array(y_true) != np.array(y_pred))[0]
    #misclassified = test_data.select(misclass_indices)

    # randomly select indices
    #selected_indices = random.sample(indices, min(num_examples, len(indices)))

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

def save_results(test_data, y_pred, model_name, label_col, task_name):

    '''
    Save classification report on predicted versus true data

    Args:
        - test_data: huggingface ds with test data
        - feature_col: label of dataset classified, e.g., 'genre'
        - embedding_col: name of column containing image embeddings
        - y_pred: predicted y labels

    '''
    # FIX THIS? 
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

def main():

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
    ds_splits = filter_data(ds_full, label, classification_task, 2830, cv = args['cv']) # not sure this argument will work
        
    # not sure if this syntax will work?
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
                #val_loader, _ = create_dataloader(ds_splits_for_cv, full_embedding_pt, label, 'val', batch_size, 'old_indices')
                test_loader, _ = create_dataloader(ds_splits_for_cv, full_embedding_pt, label, 'test', batch_size, 'old_indices')

                model_architecture = build_model(args['hidden_layer_size'], label, inp_size, 0.3, ds_splits_for_cv)
                
                # define class weights
                class_weights = define_class_weights(ds_splits_for_cv, label)

                # define lightning model
                model = SubclassModel(model_architecture, class_weights, lr=args['lr'], weight_decay=0.01, num_classes = ds_splits_for_cv['train'].features[label].num_classes)

                #check_path = os.path.join('out', 'subclassification_checkpoints')
                #os.makedirs(check_path, exist_ok=True)

                # set callback & earlystopping
                #checkpoint_callback = ModelCheckpoint(
                 #                       dirpath=os.path.join(check_path, model_name),
                  #                      monitor="val_loss",
                   #                     filename=args['savefile_suffix']+"-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
                    #                    save_top_k=2,
                     #                   mode="min",
                      #                  )
                
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

                # use best model across epochs
                #best_model_path = checkpoint_callback.best_model_path

                #model = SubclassModel.load_from_checkpoint(best_model_path)

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
                        task_name = f"{args['savefile_suffix']}_fold{fold+1}"
                    )

                del full_embedding_pt, model, test_loader, train_loader
        
            del ds_splits_for_cv, ds_train, ds_test
        
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

        # Save results_table to a text file
        save_path = os.path.join('out', 'subclassification_reports')
        with open(os.path.join(save_path, f'{model_name}_{args['savefile_suffix']}_CV_results.txt'), 'w') as f:
            f.write(results_table.to_string())

            
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
