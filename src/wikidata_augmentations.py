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
import torchvision.transforms as T
import mteb

from subclf_updated import remap_features, create_dataloader, build_model, define_class_weights, SubclassModel, save_conf_matrix, plot_misclassifications, save_results, remap_features, filter_data
from custom_augmentations import AddLayeredFrame, JPEGCompression, AddVignette, AddGrain, AddLightArtifact, RelativeGaussianBlur, FixedContrast, CannySketch, PencilSketchCustom

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

# create custom dataset class
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

def to_numpy_array(img):
    return np.array(img)

def extract_embeddings(dataset, model, aug):
    # for large datasets, it is better to input DataLoaders to get_image_embeddings function rather than a list

    # specify transforms; convert dataset image to PIL and np array (necessary as input to DataLoader)

    transform = T.Compose([
        convert_to_rgb,
        aug # add augmentation to the transforms
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

def embeddings_w_eva(test_data, aug):

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
    # make sure img is rgb WHAT TO DO HERE WITH GRAYSCALE ? 
    transforms_list.append(convert_to_rgb)
    #transforms_list = [convert_to_rgb] + transforms_model.transforms
    #transforms = T.Compose(transforms_list)
    transforms_list.append(aug) # add augmentation to list of transforms
    transforms_list.extend(transforms_model.transforms)

    transforms = T.Compose(transforms_list)

    wrapped_dataset = HFImageDataset(hf_dataset=test_data, transform=transforms)
    dataloader = DataLoader(wrapped_dataset, batch_size=32, shuffle=False) # no need for custom collating because data is tensors for timm input

    # extract embeddings from batches:
    all_embeddings = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            embeddings = model(batch)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)

# run this function on all kinds of embeddings
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

# saving functions:
def save_conf_matrix(model_name, y_true, y_pred, labels, aug_name, save_folder):

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    num_labels = len(labels)
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_labels, normalize="true")
    confmat(y_pred, y_true)
    fig, ax = confmat.plot(add_text = True, labels = labels, cmap='winter')

    out_path = os.path.join(save_folder, f'{model_name}_{aug_name}_confusion_matrix.png')
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

def save_results(test_data, y_pred, model_name, label_col, aug_name, save_folder):

    labels = np.unique(test_data[label_col])
    target_names = [test_data.features[label_col].int2str(int(i)) for i in labels]

    # save classification report for y_true and y_pred
    report = classification_report(np.array(test_data[label_col]),
                           y_pred, target_names = target_names)
    
    # save classification report
    out_path = os.path.join(save_folder, f'{model_name}_{aug_name}_classification_report_WIKIDATA.txt')

    with open(out_path, 'w') as file:
                file.write(report)

    # save confusion matrix as well:
    save_conf_matrix(model_name, np.array(test_data[label_col]), y_pred, target_names, aug_name, save_folder)


def aggregate_results(model_scores):
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
    with open(os.path.join('out', 'test_augmentation_results', 'cv_results_WIKIDATA.txt'), 'w') as f:
        f.write(results_table.to_string())

def main():
    L.seed_everything(2830)

    args = argument_parser()

    label = 'artist'

    # create folder for saving results:
    os.makedirs(os.path.join('out', 'test_augmentation_results'), exist_ok=True)

    # define augmentations:
    augmentations = {
       #'weak_blur': RelativeGaussianBlur(strength=0.3, sigma=5), # weak blurring
        #'strong_blur': RelativeGaussianBlur(strength=0.7, sigma=7), # stronger blurring
        #'grayscale': T.Grayscale(), # grayscale
        #'contrast': FixedContrast(factor = 10), # changing contrasts
        #'frame': AddLayeredFrame(border_sizes=(100, 100, 100)), # frame
        #'jpeg_compr': JPEGCompression(quality=6), # jpeg compression
        #'vignette': AddVignette(strength = 0.9), # adding vignette
        #'weak_grain': AddGrain(std=0.3),
        #'strong_grain': AddGrain(std=0.7),
        #'light_artifact': AddLightArtifact(max_intensity=0.4, max_radius_ratio=0.4),
        #'Canny_sketch': CannySketch(),
        'pencil_sketch': PencilSketchCustom()
    }

    # load data
    data_name = args['dataset']

    subset = [
    'Alfred Sisley', 
    'Berthe Morisot', 
    'Camille Pissarro', 
    'Claude Monet', 
    'Eugène Louis Boudin', 
    'Georges Seurat', 
    'Mary Cassatt',  
    'Paul Cézanne', 
    'Pierre-Auguste Renoir', 
    'Henri de Toulouse-Lautrec', 
    'Édouard Manet'
]

    # load full dataset with all images from disk
    ds = datasets.load_from_disk(os.path.join('data', data_name))

    def map_int_to_str(example):
        return {
            'artist_str': ds.features['artist'].int2str(example['artist'])
        }

    ds = ds.map(map_int_to_str, batched=False)

    ds_full = filter_data(ds, 'artist', subset, 2830, cv=True) # ignore last two parameters, just reusing function

    # add indices as column
#    ds_full = ds.add_column('old_indices', range(len(ds)))

    batch_size = args['batch_size']

    # Initialize model_scores dictionary
    model_scores = {}

    for model_name in args['model_names']:
        # Start with all augmentations
        aug_dict = {aug_name: [] for aug_name in augmentations.keys()}
        # Add the "no augmentation" baseline
        aug_dict['no_aug'] = []
        # Assign to the model
        model_scores[model_name] = aug_dict

    skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2830)
    y = np.array(ds_full['artist'])
    indices = np.arange(len(ds_full))

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, y)):
        print(f"Fold {fold+1}") # monitor folds

        ds_train = ds_full.select(train_idx.tolist())
        ds_test = ds_full.select(val_idx.tolist())

        ds_splits_for_cv = {
            'train': ds_train,
            'test': ds_test}

        for model_name in args['model_names']:

            # create folder for saving results:
            out_folder = os.path.join('out', 'test_augmentation_results', model_name)
            os.makedirs(out_folder, exist_ok=True)

            full_embedding_pt = torch.load(os.path.join('data', 'wikidata_embeddings', model_name, f'{model_name}_all_splits.pt'))
            train_loader, inp_size = create_dataloader(ds_splits_for_cv, full_embedding_pt, 'artist', 'train', batch_size, 'old_indices')
            
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

                    continue # skip this model

            # fit seperate model for each data augmentation
            for aug_name, aug in augmentations.items(): 

                #embeddings = None
                if model_name == 'eva02_clip_336':
                    try:
                        embeddings = embeddings_w_eva(ds_splits_for_cv['test'], aug)
                    
                    except Exception as e:
                        print(f'Error extraction features, {e}')
                
                else:
                    try: 
                        embeddings = extract_embeddings(ds_splits_for_cv['test'], mteb_model, aug)

                    except Exception as e: 
                        print(e)
                    
                #if embeddings is None:
                    #   raise ValueError("No embeddings returned")
                

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

                # save for fold
                model_scores[model_name][aug_name].append({
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
                        aug_name = f"{aug_name}_fold{fold+1}_WIKIDATA",
                        save_folder=out_folder
                    )

                #save_model_scores(model_scores, aug_name, out_folder)

                del embeddings
                del test_loader
                del model_architecture
                del model
                del trainer
                del test_metrics
                del all_preds_batches
                del all_preds

                import gc
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            
            # fit model without augmentations: 
            test_loader, _ = create_dataloader(ds_splits_for_cv, full_embedding_pt, label, 'test', batch_size, 'old_indices')
            model_architecture = build_model(args['hidden_layer_size'], label, inp_size, 0.3, ds_splits_for_cv)
            class_weights = define_class_weights(ds_splits_for_cv, label)
            model = SubclassModel(model_architecture, class_weights, lr=args['lr'], weight_decay=0.01, num_classes = ds_splits_for_cv['train'].features['artist'].num_classes)
            trainer = L.Trainer(
                        max_epochs=args['epochs'],
                        #callbacks=[checkpoint_callback],
                        accelerator="gpu" if torch.cuda.is_available() else "cpu",
                        devices="auto",
                        deterministic=True
                        )
            
            trainer.fit(model, train_loader)
            test_metrics = trainer.test(model, test_loader)

            model_scores[model_name]['no_aug'].append({
                "acc": test_metrics[0]["test_acc"],
                "precision": test_metrics[0]["test_precision"],
                "recall": test_metrics[0]["test_recall"],
                "f1": test_metrics[0]["test_f1"]
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

            # figure out how to add results from here to the CV results aggregated across augmentations and folds
            
            del full_embedding_pt

    aggregate_results(model_scores)

if __name__ == '__main__':
    main()