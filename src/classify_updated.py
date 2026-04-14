import torch 
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def build_classification_model(ds_splits, model_name, label, batch_size, device, inp_size):

    if label=='artist':

        hidden_layer_size = 2000
        decay=0.01
        lr = 0.001
        dropout_p = 0.3
    
    elif label=='style':
        
        hidden_layer_size = 1200
        decay=0.01
        lr = 0.0003
        dropout_p = 0.3
    
    else: # model-specific settings for genre
        hidden_layer_size = 500
        decay=0.01
        lr = 0.0001
        dropout_p = 0.4

    # get number of classes and embeddings dimensions
    num_classes = ds_splits['train'].features[label].num_classes

    model = nn.Sequential(
        nn.Linear(in_features=inp_size, out_features=hidden_layer_size),
        nn.ReLU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features=hidden_layer_size, out_features=num_classes)
            ).to(device) # use GPU if available

    # define class weights
    y_tensor = torch.tensor(ds_splits['train'][label])
    class_counts = torch.bincount(y_tensor)
    class_weights = 1.0 / class_counts.float() # weight the loss inversely proportional to class frequency
    class_weights /= class_weights.sum() # normalize weights so they sum to one

    # Define loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) 

    # define optimizer with learning rate scheduler and L2 regularization (==weight_decay - there's no direct kernel-regularizer keras equivalent in torch)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay) # apply to all hyperparameters of all layers

    # learning rate scheduler with exponential decays
    #steps_per_epoch = len(ds_splits['train']) // batch_size
    #decay_steps = steps_per_epoch * 2
    #decay_per_batch = 0.9 ** (1 / decay_steps)  # calc decay rate per batch

    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_per_batch)
    
    # test with per-epoch decay rather than per batch
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    model_inits = {'model': model,
                            'criterion': criterion,
                            'optimizer': optimizer,
                            'scheduler': scheduler}

    return model_inits

def create_dataloader(ds_splits, full_embedding_pt, label, split, batch_size, device):
    
    class EmbeddingsDataset(Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = embeddings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]

    # load full embedding and split based on correct indices
    split_indices = ds_splits[split]['old_emb_indices']
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
    plt.plot(np.arange(0, len(H["train_loss"])), H["train_loss"], label="train_loss")
    plt.plot(np.arange(0, len(H["val_loss"])), H["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, len(H["train_accuracy"])), H["train_accuracy"], label="train_acc")
    plt.plot(np.arange(0, len(H["val_accuracy"])), H["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()

    plots_dir_path = os.path.join('out', 'plots')
    os.makedirs(plots_dir_path, exist_ok=True)

    plt.savefig(os.path.join(plots_dir_path, name))

def build_training_loop(epochs, model_inits, dataloaders, model_name, label, device):

    # Early stopping setup
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

     # initialize empty lists to track loss and accuracy
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    criterion = model_inits['criterion']
    for epoch in range(epochs):

        ####################### TRAINING #######################
        # set model to training mode
        model_inits['model'].train()
        total_loss, correct, total = 0.0, 0, 0 # initialize counters to accumulate loss/accuracies

        for X_train, y_train in dataloaders['train_loader']: # iterate over embedding + labels in batches
            
            # move data to GPU if available
            X_train, y_train = X_train.to(device), y_train.to(device)

            # zero gradients for every batch
            model_inits['optimizer'].zero_grad()

            # make predictions for batch
            outputs = model_inits['model'](X_train)

            # calculate loss
            loss = criterion(outputs, y_train)
            loss.backward() # gradient backpropagation

            # update weights
            model_inits['optimizer'].step()

            # update learning rate according to schedule
            #model_inits['scheduler'].step()

            # calculate total loss for batch and add to total_loss variable to accumulate across batches
            total_loss += loss.item() * X_train.size(0)

            # calculate the number of correct predictions for batch
            correct += (outputs.argmax(1) == y_train).sum().item() # .argmax(1) gives predicted class for each sample 

            # count and add samples seen per batch
            total += y_train.size(0)

        # average loss across batches; average loss for epoch
        train_loss = total_loss / total

        # average accuracy across batches; average accuracy for epoch
        train_acc = correct / total

        # append to list gathering metrics for all epochs
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        ####################### VALIDATION #######################
        model_inits['model'].eval() # set model to evaluation mode/inference mode
        
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad(): # don't update gradients as we are only validating
            for X_val, y_val in dataloaders['val_loader']:

                # move data to GPU if available
                X_val, y_val = X_val.to(device), y_val.to(device)
                
                # make predictions for batch
                outputs = model_inits['model'](X_val)

                # compute loss - but no update!
                loss = criterion(outputs, y_val)

                # calculate total loss for batch and add to val_loss variable to accumulate across batches
                val_loss += loss.item()*X_val.size(0)

                # calculate the number of correct predictions for batch
                correct += (outputs.argmax(1) == y_val).sum().item()

                # count and add samples seen per batch
                total += y_val.size(0)

        # average loss across batches; average loss for epoch
        val_loss /= total

        # average accuracy across batches; average accuracy for epoch
        val_acc = correct / total

        # append to list gathering metrics for all epochs
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        model_inits['scheduler'].step()

        # prinnt metrics for epoch
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        ##### EARLY STOPPING
        if val_loss < best_val_loss: # < because we want the smallest lost possible
            best_val_loss = val_loss
            best_model_state = model_inits['model'].state_dict().copy() # save weights of the best model
            patience_counter = 0
        
        else: # if validation loss is not improving
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping: Model ran for {epoch} epochs")
                break
    
    ####################### TEST/INFERENCE #######################

    # Load weights of best model/replace parameters with best model's
    model_inits['model'].load_state_dict(best_model_state) 

    # set model to evaluation mode
    model_inits['model'].eval()

    predictions = []

    with torch.no_grad(): # disable gradient updates

        for X_test, y_test in dataloaders['test_loader']:
            X_test = X_test.to(device)
            outputs = model_inits['model'](X_test) # predict with model on test data
            preds = outputs.argmax(1).cpu().numpy() # get prediction across labels dimension
            predictions.extend(preds) # add to overall predictions to accumulate predictions across batches
    
    # save predictions to file
    y_pred_path = os.path.join('out', 'y_pred')
    os.makedirs(y_pred_path, exist_ok=True)
    np.save(os.path.join(y_pred_path, f'{model_name}_{label}_y_pred.npy'), predictions)

    # save history
    history = {'train_loss': train_losses,
               'val_loss': val_losses,
               'train_accuracy': train_accs,
               'val_accuracy': val_accs}

    # save model weights

    fitted_model_dir = os.path.join("out", "saved_models")
    os.makedirs(fitted_model_dir, exist_ok=True) 
    model_save_path = os.path.join(fitted_model_dir, f"{model_name}_{label}_best_model.pt")
    torch.save(model_inits['model'].state_dict(), model_save_path)

    return history, predictions

def save_classification_report(test_data, label_col, model_name, predicted_classes):

    '''
    Save classification report on predicted versus true data

    Args:
        - test_data: huggingface ds with test data
        - feature_col: label of dataset classified, e.g., 'genre'
        - embedding_col: name of column containing image embeddings
        - predicted_classes: predicted y labels

    '''
    
    # save the class labels
    #label_class = test_data.features[label_col]

    # save the number of classes
    #num_classes = test_data.features[label_col].num_classes

    # map integer values to class label strings
    #mapped_labels = {}

    #for i in range(num_classes):
     #  mapped_labels[i] = label_class.int2str(i)
    
    #labels = list(mapped_labels.values())

    labels = np.unique(test_data[label_col])
    target_names = [test_data.features[label_col].int2str(int(i)) for i in labels]

    # save classification report for y_true and y_pred
    report = classification_report(np.array(test_data[label_col]),
                           predicted_classes, target_names = target_names)
    
    # save classification report
    os.makedirs(os.path.join('out', 'classification_reports'), exist_ok=True)
    out_path = os.path.join("out", "classification_reports", f'{model_name}_{label_col}_classification_report.txt')

    with open(out_path, 'w') as file:
                file.write(report)

def fit_and_predict(ds_splits, model_name, label, batch_size, epochs, device):

    '''
    Model name as in only name, not 'google/siglip...' path

    This is the function we want to import to the other script
    '''
    print(f"Starting classification for {model_name}")

    # load embeddings for model

    full_embedding_pt = torch.load(os.path.join('data', 'filtered_embeddings_FINAL', model_name, f'{model_name}_all_splits.pt'))

    # define dataloaders
    train_loader, inp_size = create_dataloader(ds_splits, full_embedding_pt, label, 'train', batch_size, device)
    val_loader, _ = create_dataloader(ds_splits, full_embedding_pt, label, 'val', batch_size, device)
    test_loader, _ = create_dataloader(ds_splits, full_embedding_pt, label, 'test', batch_size, device)

    dataloaders = {'train_loader': train_loader,
                   'val_loader': val_loader,
                   'test_loader': test_loader}
    
    # build model
    model_inits = build_classification_model(ds_splits, model_name, label, batch_size, device, inp_size)

    # train loop
    history, predictions = build_training_loop(epochs, model_inits, dataloaders, model_name, label, device)
    
    # save history plot
    save_plot_history(history, epochs, f'{model_name}_{label}_history.png')

    # save classification report
    save_classification_report(ds_splits['test'], label, model_name, predictions)

    # model weights are saved in training loop function