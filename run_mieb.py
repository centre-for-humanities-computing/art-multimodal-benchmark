import pandas as pd
import datasets
from datasets import Image as Image_ds # change name because of similar PIL module
from datasets import Dataset
import os
import requests 
from PIL import Image
from tqdm import tqdm
import torch
from datasets import load_dataset
import numpy as np
import mteb
import argparse 
from functools import partial

def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--leaderboard', type=str, help='name of csv file with MIEB leaderboard')
    parser.add_argument('--n_models', type=int, help='top n models to use from leaderboard')
    parser.add_argument('--dataset', type=str, help='name of HuggingFace dataset')
    parser.add_argument('--epochs', type=int, help="how many epochs to run the model for")
    parser.add_argument('--hidden_layer_size', type=int, help='size of the hidden layer in classification model')
    parser.add_argument('--batch_size', type=int)
    args = vars(parser.parse_args())
    
    return args

def load_iter_hf_data(dataset_name):

    '''
    Load a dataset from Huggingface Hub and save locally

    Args:
        - dataset_name: name of dataset from HuggingFace Hub
    '''

    # load dataset from the hub
    hf_data = datasets.load_dataset(dataset_name, split='train', streaming=True)

    # convert dataset to iterable generator
    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    # convert to dataset to be saved locally
    ds = datasets.Dataset.from_generator(partial(gen_from_iterable_dataset, hf_data), features=hf_data.features)

    return ds

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
    ds_train.save_to_disk(os.path.join('datasets', f"{name}_train"))
    ds_test.save_to_disk(os.path.join('datasets', f"{name}_test"))
    ds_val.save_to_disk(os.path.join('datasets', f"{name}_val"))

    return {
    'train': ds_train,
    'test': ds_test,
    'val': ds_val
}

def get_model_names(leaderboard_csv, n_models):
    leaderboard = pd.read_csv(os.path.join('data', leaderboard_csv)) ### FIX PATH !!!

    # get list of all model in mteb
    all_metas = mteb.get_model_metas()

    # filter out image models, save their names
    vision_names = [meta.name for meta in all_metas if "image" in meta.modalities]

    # get model names from column of names + HF links
    leaderboard_model_names = []

    for model_link in leaderboard['Model'].iloc[:n_models]:
        model_name = model_link.split(']')[0][1:]
        leaderboard_model_names.append(model_name)

    # match model name to full HF path
    models_full_paths = []

    for model in leaderboard_model_names:
        for path in vision_names:
            if path.endswith(model):
                models_full_paths.append(path)

    model_metas = []
    for model in models_full_paths:
        model_meta = mteb.get_model_meta(model)
        model_metas.append(dict(model_meta))

    model_metadata = pd.DataFrame(model_metas)

    # save model overview to file:
    model_metadata.to_csv('testy.csv') # FIX PATH !!!

    return model_metadata

def extract_embeddings(dataset, model_path:str):
    
    images = dataset['image']

    # get meta information of specified model
    model_meta = mteb.get_model_meta(model_path)
    
    # load model from mteb
    model = model_meta.load_model()
    
    # extract image embeddings for all images with model
    print(f"Extracting embeddings with {model_path}")
    embeddings = model.get_image_embeddings(images)
    
    # convert to list, otherwise can't save to HF column
    embeddings_list = embeddings.cpu().tolist()

    # add embeddings to column
    #dataset = dataset.add_column(model_path, embeddings_list)

    return embeddings_list

def embeddings_from_splits(ds_splits, models_metadata):

    succesful_models = []
    failed_models = []
    for model_path in models_metadata['name']:
        try:
            print(f'Extracting embeddings for {model_path} over train, test and val splits')

            for split_name in ds_splits:
                dataset_split = ds_splits[split_name]

                # extract embedding from dataset
                embeddings = extract_embeddings(dataset_split, model_path)

                # add column with embeddings, named after model path
                dataset_split = dataset_split.add_column(model_path, embeddings)

                ds_splits[split_name] = dataset_split
            
            succesful_models.append(model_path)

        except Exception as e:
            print(f'Error with {model_path}: {e}')
            failed_models.append(model_path)
    print(f"These models failed: {failed_models}")    

    return ds_splits, succesful_models

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
    hidden_layer = Dense(hidden_layer_size, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inp)

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
    plt.savefig(os.path.join('out', 'plots', name))

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

    model = build_classification_model(ds_splits['train'], 
                                       hidden_layer_size, 
                                       label_col, 
                                       embedding_col)

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
    # FIX PATH
    np.save(f'out/history/{embedding_col}_{label_col}_history.npy', H.history)

    num_epochs = len(H.history['val_loss'])
    print(f"Model ran for {num_epochs} epochs")

    # save history plot in "plots" folder
    # FIX PATH
    save_plot_history(H, num_epochs, f'{embedding_col}_{feature_col}_history.png')

    # predict on test data
    predictions = model.predict(tf_ds_test)

    # find class with the highest probability
    predicted_classes = np.argmax(predictions, axis=1)

    # save predicted classes as .npy to be used for plotting
    np.save(f'out/y_pred/{embedding_col}_{feature_col}_y_pred.npy', predicted_classes)

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
    report = classification_report(test_data[feature_col],
                            predicted_classes, target_names = labels)
    
    # save classification report
    out_path = os.path.join("out", "classification_reports", f'{embedding_col}_{feature_col}_classification_report.txt')

    with open(out_path, 'w') as file:
                file.write(report)

#find better name..
def classify_all_features(successful_models, 
                          ds_splits, 
                          hidden_layer_size, 
                          batch_size, 
                          epochs):

    label_cols = ['genre', 'style', 'artist']

    # run classifier for each model
    for model_name in successful_models:
        for label_col in label_cols:
            predicted_classes = fit_and_predict(ds_splits, 
                                            hidden_layer_size, 
                                            model_name, 
                                            label_col,
                                            batch_size, 
                                            epochs)
            
            save_classification_report(ds_splits['test'], 
                                       label_col, 
                                       model_name, 
                                       predicted_classes)


def main():

    # parse arguments
    args = argument_parser()

    # load dataset
    ds = load_iter_hf_data(args['dataset'])

    # split dataset to train, test and validation
    ds_splits = split_data(ds, name, seed)

    # INSTEAD, GET SCRIPT TO WORK ON A LIST OF MODEL NAMES RATHER THAN LEADERBOARD CSV?

    # extract metadata from models
    models_metadata = get_model_names(leaderboard_csv, args['n_models'])

    # add embeddings from all models to columns in each dataset split
    ds_splits, succesful_models = embeddings_from_splits(ds_splits, models_metadata)

    # Now I should have all data needed for running classify.py scripts
    classify_all_features(succesful_models,
                          ds_splits, 
                          args['hidden_layer_size'], 
                          args['batch_size'], 
                          args['epochs'])

    if __name__ == '__main__':
        main()


        
