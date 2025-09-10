print('Importing modules...')
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import regularizers
import pandas as pd
import datasets
#from datasets import Image as Image_ds # change name because of similar PIL module
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
from sklearn.metrics import classification_report
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import multiprocessing as mp

LOG_FILE_NAME = None

def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path of model in MTEB to use')
    parser.add_argument('--dataset', type=str, help='name of HuggingFace dataset')
    parser.add_argument('--label_cols', nargs='+', help= 'List of classification labels/tasks, must be columns in the dataset of type ClassLabel')
    parser.add_argument('--epochs', type=int, help="how many epochs to run the model for")
    parser.add_argument('--hidden_layer_size', type=int, help='size of the hidden layer in classification model')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--log_file_name', type=str, help='what to call the output logfile')
    args = vars(parser.parse_args())
    
    return args

# log error messages and save to file

def log(message):
    global LOG_FILE_NAME
    log_path = os.path.join('out', 'logs')
    os.makedirs(log_path, exist_ok=True)

    with open(os.path.join(log_path, f'{LOG_FILE_NAME}.txt'), "a") as f:
        f.write(message + "\n")



def extract_embeddings(dataset, model):
    # for large datasets, it is better to input DataLoaders to get_image_embeddings function rather than a list

    # specify transforms; convert dataset image to PIL and np array (necessary as input to DataLoader)
    
    def convert_to_rgb(img):
        return img.convert("RGB")

    def to_numpy_array(img):
        return np.array(img)

    transform = transforms.Compose([
        convert_to_rgb,
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
    
    # add embeddings to column
    #dataset = dataset.add_column(model_path, embeddings_list)

def embeddings_from_splits(ds_splits, model_path):

    # get model meta and load model
    model_meta = mteb.get_model_meta(model_path)

    try:
        print('LOADING MODEL...')
        model = model_meta.load_model()

    except Exception as e:
        log(f'Error loading model: {model_path}, {e}')
        print(f'Error loading model: {model_path}, {e}')
    
    # now we only want the name of the model, not the entire HuggingFace path
    model_name = model_path.split('/')[1]

    # create folder to save embeddings to
    embeddings_outpath = os.path.join('data', 'embeddings')
    os.makedirs(embeddings_outpath, exist_ok=True)

    # extract image embeddings for all images across splits with model
    print(f'Extracting embeddings with {model_name}')

    for split_name in ds_splits:
        dataset_split = ds_splits[split_name] # get i.e., train dataset
        
        try:
            embeddings = extract_embeddings(dataset_split, model)

        except Exception as e:
            print(e)
        
        if embeddings is None:
            raise ValueError("No embeddings returned")
        
        # save embeddings to npy file:
        embeddings = embeddings.cpu().numpy()

        # save embeddings to folder for model:
        os.makedirs(os.path.join(embeddings_outpath, model_name), exist_ok=True)
        np.save(os.path.join(embeddings_outpath, model_name, f'{model_name}_{split_name}.npy'), embeddings, allow_pickle=True)

        # delete embeddings for that split from memory
        del embeddings
    
    # after extracting embeddings, delete model from memory:
    del model 

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def build_classification_model(train_data, hidden_layer_size, feature_col, embedding_col, batch_size):
    '''
    Build simple neural network with tensorflow

    Args:
        - train_data: huggingface ds with train data
        - hidden_layer_size: specify size of hidden layer
        - feature_col: label of dataset to classify, e.g., 'genre'
        - embedding_col: name of column containing image embeddings
    '''
    # save number of classes (to be used for the last layer of the model)
    num_classes = train_data.features[feature_col].num_classes

    # define input shape
    inp_size = len(train_data[0][embedding_col])
    inp = Input(shape=(inp_size,))

    # define shape of hidden layer
    hidden_layer = Dense(hidden_layer_size, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inp)

    # add classification layer
    classification_layer = Dense(num_classes, activation='softmax')(hidden_layer)

    # define model
    model = Model(inputs=inp, outputs=classification_layer)

    steps_per_epoch = len(train_data) // batch_size
    decay_steps = steps_per_epoch * 2

    # define learning rate schedule and optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=decay_steps,
        decay_rate=0.9)

    adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # compile model
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def save_plot_history(H, epochs, name):
    '''
    Saves the validation and loss history plots of a fitted model in the 'out' folder.
    
    Arguments:
    - H: Saved history of a model fit
    - epochs: Number of epochs the model runs on
    - name: What the plot should be called
    
    Returns:
        None
    '''
    #plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()

    plots_dir_path = os.path.join('out', 'plots')
    os.makedirs(plots_dir_path, exist_ok=True)

    plt.savefig(os.path.join(plots_dir_path, name))

def fit_and_predict(ds_splits, hidden_layer_size, embedding_col, label_col, batch_size, epochs):

    '''
    Fit a compiled model on training data and predict on test dataset

    Args:
        - train_data: huggingface ds with training data
        - test_data: huggingface ds with test data
        - val_data: huggingface ds with val data
        - hidden_layer_size: size of hidden layer
        - embedding_col: name of column containing image embeddings
        - feature_col: label of dataset to classify, e.g., 'genre'
        - batch_size: batch size
        - epochs: how many epochs to run the model for
    '''

    # load npy file:
    model = build_classification_model(ds_splits['train'], 
                                       hidden_layer_size, 
                                       label_col, 
                                       embedding_col,
                                       batch_size)

    # convert to tensorflow datasets
    tf_ds_train = ds_splits['train'].to_tf_dataset(
            columns=embedding_col, # the columns to be used as inputs to the model, X
            label_cols=label_col, # columns containing class labels, y
            batch_size=batch_size,
            shuffle=True # Only shuffle train data
            )
    
    tf_ds_test = ds_splits['test'].to_tf_dataset(
            columns=embedding_col,
            label_cols=label_col, 
            batch_size=batch_size,
            shuffle=False # for test data, set shuffle to false
            )
    
    tf_ds_val = ds_splits['val'].to_tf_dataset(
            columns=embedding_col,
            label_cols=label_col, 
            batch_size=batch_size,
            shuffle=False # same for validation data
            )


    # add early stopping, stop training if validation loss does not improve for three epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # fit model and save history
    H = model.fit(tf_ds_train, 
                    epochs = epochs,
                    validation_data=tf_ds_val,
                    callbacks=[early_stopping])

    # save model history to use for plotting later
    history_path = os.path.join('out', 'history')
    os.makedirs(history_path, exist_ok=True)
    np.save(os.path.join(history_path, f'{embedding_col}_{label_col}_history.npy'), H.history)

    num_epochs = len(H.history['val_loss'])
    print(f"Classification done for {embedding_col} - {label_col}. Model ran for {num_epochs} epochs")
    log(f"Classification done for {embedding_col} - {label_col}. Model ran for {num_epochs} epochs")

    # save history plot in "plots" folder
    save_plot_history(H, num_epochs, f'{embedding_col}_{label_col}_history.png')

    # predict on test data
    predictions = model.predict(tf_ds_test)

    # find class with the highest probability
    predicted_classes = np.argmax(predictions, axis=1)

    # save predicted classes as .npy to be used for plotting
    y_pred_path = os.path.join('out', 'y_pred')
    os.makedirs(y_pred_path, exist_ok=True)
    np.save(os.path.join(y_pred_path, f'{embedding_col}_{label_col}_y_pred.npy'), predicted_classes)

    return predicted_classes

def save_classification_report(test_data, label_col, embedding_col, predicted_classes):

    '''
    Save classification report on predicted versus true data

    Args:
        - test_data: huggingface ds with test data
        - feature_col: label of dataset classified, e.g., 'genre'
        - embedding_col: name of column containing image embeddings
        - predicted_classes: predicted y labels

    '''
    
    # save the class labels
    label_class = test_data.features[label_col]

    # save the number of classes
    num_classes = test_data.features[label_col].num_classes

    # map integer values to class label strings
    mapped_labels = {}

    for i in range(num_classes):
       mapped_labels[i] = label_class.int2str(i)
    
    labels = list(mapped_labels.values())

    # save classification report for y_true and y_pred
    report = classification_report(test_data[label_col],
                           predicted_classes, target_names = labels)
    
    # save classification report
    os.makedirs(os.path.join('out', 'classification_reports'), exist_ok=True)
    out_path = os.path.join("out", "classification_reports", f'{embedding_col}_{label_col}_classification_report.txt')

    with open(out_path, 'w') as file:
                file.write(report)

def classify_all_features(ds_splits, 
                          model_name,
                          hidden_layer_size, 
                          batch_size, 
                          epochs,
                          label_cols):

    for split_name in ds_splits:
        dataset_split = ds_splits[split_name]
        embeddings_path = os.path.join('data', 'embeddings', model_name, f'{model_name}_{split_name}.npy')
        embeddings = np.load(embeddings_path, allow_pickle=True)
        embeddings = embeddings.tolist()
        dataset_split = dataset_split.add_column(model_name, embeddings)
        ds_splits[split_name] = dataset_split

    for label_col in label_cols:
        try:
            predicted_classes = fit_and_predict(ds_splits, 
                                            hidden_layer_size, 
                                            model_name, 
                                            label_col,
                                            batch_size, 
                                            epochs)
            
            if predicted_classes is not None:
                save_classification_report(ds_splits['test'], 
                                        label_col, 
                                        model_name, 
                                        predicted_classes)
        
            #print(f'Classification done for {model_name} - {label_col}')
            #log(f'Classification done for {model_name} - {label_col}')

        except Exception as e:
            log(f"Classification failed for {model_name} - {label_col} - Error: {e}")
            print(f"Classification failed for: {model_name} - {label_col} - Error: {e}")
            continue

def split_data(ds, name, seed):
    '''
    Split data into train, test and validation splits

    Args:
        - ds: saved huggingface dataset
        - name: prefix of saved dataset splits (e.g., 'WikiArt' will save datasets WikiArt_train, WikiArt_test etc.)
        - seed: set seed to ensure reproducability
    '''

    # split data into train and test
    ds_split = ds.train_test_split(test_size=0.2, seed=seed)
    ds_train = ds_split['train']
    ds_test = ds_split['test']

    # split test data into test and validation
    ds_test_split = ds_test.train_test_split(test_size=0.5, seed=seed)
    ds_val = ds_test_split['train']
    ds_test = ds_test_split['test']

    # save the datasets to disk with the same prefix
    os.makedirs('data', exist_ok=True)

    #ds_train.save_to_disk(os.path.join('data', f"{name}_train"))
    #ds_test.save_to_disk(os.path.join('data', f"{name}_test"))
    #ds_val.save_to_disk(os.path.join('data', f"{name}_val"))

    return {
    'train': ds_train,
    'test': ds_test,
    'val': ds_val
}

def main():

    global LOG_FILE_NAME
    # parse arguments
    args = argument_parser()

    LOG_FILE_NAME = args['log_file_name']

    data_name = args['dataset']

    ds_train = datasets.load_from_disk(os.path.join('data', f'{data_name}_train'))
    ds_test = datasets.load_from_disk(os.path.join('data', f'{data_name}_test'))
    ds_val = datasets.load_from_disk(os.path.join('data', f'{data_name}_val'))

    ds_splits = {
    'train': ds_train,
    'test': ds_test,
    'val': ds_val}

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f'GPUs available: {gpus}')
    else:
        print('No GPU found, running on CPU.')

    # extract embeddings and save to file

    embeddings_from_splits(ds_splits, args['model_path'])

    model_name = args['model_path'].split('/')[1]

    # Now I should have all data needed for running classify.py scripts
    classify_all_features(ds_splits, 
                          model_name,
                          args['hidden_layer_size'], 
                          args['batch_size'], 
                          args['epochs'],
                          args['label_cols'])

    log(f'Feature extraction and classification completed for {model_name}!')

if __name__ == '__main__':
    #mp.set_start_method('spawn', force=True)
    main()