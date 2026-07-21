"""
Augment images and extract embeddings from them for a subset of the wikiart dataset 
"""

import datasets
import os
from datasets import ClassLabel
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import timm
import torchvision.transforms as T
import mteb
from tqdm import tqdm
import cv2

cv2.setNumThreads(0)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='name of huggingface dataset')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_names', nargs='+', help='list of models to run classification task with')

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

def filter_data(ds: datasets.Dataset, label: str, subclassification_task: list) -> datasets.Dataset:

    """
    Filters a dataset to a subset of classes and remaps its ClassLabel feature.

    Selects only the rows whose label (as a string) is in the filtering parameter,
    then remaps the ClassLabel feature to reflect only the remaining classes.

    Args:
        ds: The full dataset to filter.
        label: The name of the ClassLabel column to filter on (e.g. 'artist' or 'genre').
        subclassification_task: The class names to keep.

    Returns:
        The filtered dataset with remapped ClassLabel and an added 'old_indices' column
        tracking each row's position in the original dataset.
    """

    # add old column to keep track of embedding indices
    ds = ds.add_column('old_indices', range(len(ds)))

    # find the rows that matches the subset criterion
    subclass_indices = [idx for idx, a in enumerate(ds[f'{label}_str']) if a in subclassification_task]
    ds_subset = ds.select(subclass_indices)

    # remap labels to fit to new number of classes for subclassification task
    ds_subset = remap_features(ds, ds_subset, label)

    return ds_subset

# create custom dataset class
class HFImageDataset(Dataset):

    """
    A PyTorch Dataset wrapper for a HF datasat with image data
    (Necessary to wrap data like this to match expected input to a DataLoader - Basically just converts a huggingface dataset with images in an 'image' column to a format that MTEB will accept)
    """
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

def extract_embeddings(dataset: datasets.Dataset, model, batch_size:int) -> torch.Tensor | None:

    """
    Extract embeddings using an MTEB-loaded model.
    For large datasets, it is better to input DataLoaders to get_image_embeddings function rather than a list.

    Args:
        dataset: HuggingFace Dataset with an 'image' column.
        model:   MTEB model.

    Returns:
        A tensor of shape (N, embedding_dim), or None if an error occurs.

    """

    # specify transforms; convert image to RGB 
    transforms = T.Compose([
        convert_to_rgb,
    ]) 
    
    # need to define custom data collater, create batch of list instead of stacking (not possible as input are not tensors)
    def pil_collate_fn(batch):
        return {"image": batch}

    # apply
    try:
        wrapped_dataset = HFImageDataset(hf_dataset=dataset, transform=transforms)

        # create dataloader with wrapped dataset
        dataloader = DataLoader(wrapped_dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                collate_fn=pil_collate_fn)
    
        # process images in batches from dataloader (function automatically applies model-specific preprocessing)
        embeddings = model.get_image_embeddings(dataloader)

        return embeddings
    
    except Exception as e:
        print(f'Error processing images with model: {model}, {e}')
        return None

def embeddings_w_eva(test_data: datasets.Dataset, batch_size: int) -> torch.Tensor:

    """
    Extract image embeddings using the EVA02 CLIP model from timm (PyTorch Image Models), as EVA is more easily implemented through this package.

    Args:
        ds: HuggingFace Dataset with an 'image' column.
        batch_size: batch size!

    Returns:
        A tensor of shape (N, embedding_dim) containing all image embeddings.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model from timm
    model = timm.create_model('eva02_large_patch14_clip_336.merged2b', pretrained=True, num_classes=0)
    model.eval() # turn on evaluation mode
    model.to(device)

    # save preprocessing information from the pretrained model
    data_config = timm.data.resolve_model_data_config(model)

    # use this information to transform the data
    transforms_model = timm.data.create_transform(**data_config, is_training=False)

    transforms_list = []

    # convert to RBG and apply model-specific transforms
    transforms_list.append(convert_to_rgb)
    transforms_list.extend(transforms_model.transforms)
    transforms = T.Compose(transforms_list)

    # create custom dataset for input to dataloaders
    wrapped_dataset = HFImageDataset(hf_dataset=test_data, transform=transforms)

    # create dataloader
    dataloader = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=False) # no need for custom collating because data is tensors for timm input

    # extract embeddings from batches:
    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            embeddings = model(batch)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def add_model_prefix(model_name: str) -> str | None:

    """
    Prepends the HuggingFace organization prefix to a model name to load the model via MTEB

    Args:
        model_name: The model name (e.g. 'dinov2-base').

    Returns:
        The model name with its organization prefix (e.g. 'facebook/dinov2-base'),
        or None if the model name doesn't match any known pattern.
    """

    name = model_name.lower()

    # OpenAI CLIP
    if name == "clip-vit-large-patch14":
        return f"openai/{model_name}"

    # LAION CLIP 
    if name.startswith("clip-vit"):
        return f"laion/{model_name}"

    # SigLIP
    if name.startswith("siglip"):
        return f"google/{model_name}"

    # DINOv2
    if name.startswith("dinov2"):
        return f"facebook/{model_name}"

def filter_test_from_sketches(ds_test: datasets.Dataset) -> Dataset:

    """ 
    Removes sketch and illustration samples from a dataset.

    Args:
        ds_test: A dataset with a 'genre_str' column.

    Returns:
        The dataset with sketch and study and illustration rows removed, and the genre ClassLabel remapped. 
    """

    # save classes containing illustrations
    illustrations_classes = ['sketch_and_study', 'illustrations']

    # select rows that do not contain illustrations
    no_illu_indices = [idx for idx, a in enumerate(ds_test['genre_str']) if a not in illustrations_classes]
    ds_test_filtered = ds_test.select(no_illu_indices)

    # remapping features (important; we are NOT resetting the indices! -> they still match the embedding indices)
    ds_test_filtered = remap_features(ds_test, ds_test_filtered, 'genre')

    return ds_test_filtered

class AugmentFn:
    def __init__(self, aug):
        self.aug = aug
    
    def __call__(self, example):
        import cv2
        img = example['image'].convert("RGB")
        img = self.aug(img)
        img = img.convert("RGB")
        return {'image': img}

class PILImageDataset:
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {'image': self.images[idx]}

def main():

    # import custom augmentation functions from script
    from custom_augmentations_new import AddLayeredFrame, JPEGCompression, AddVignette, AddGrain, AddLightArtifact, RelativeGaussianBlur, FixedContrast, CannySketch, PencilSketchCustom
    
    # import arguments
    args = argument_parser()

    # create parent save folder
    os.makedirs(os.path.join('data', 'aug_embeddings'), exist_ok=True)

    # define augmentations and their titles
    augmentations = {
        'strong_blur': RelativeGaussianBlur(strength=0.7, sigma=7),
        'grayscale': T.Grayscale(), # grayscale
        'contrast': FixedContrast(factor = 10), # changing contrasts
        'frame': AddLayeredFrame(border_sizes=(100, 100, 100)), # frame #
        'jpeg_compr': JPEGCompression(quality=6), # jpeg compression
        'vignette': AddVignette(strength = 0.9), # adding vignette
        'weak_grain': AddGrain(std=0.3), 
        'light_artifact': AddLightArtifact(max_intensity=0.4, max_radius_ratio=0.4),
        'Canny_sketch': CannySketch(), # edge detection algorithm to create a fake drawing
        'pencil_sketch': PencilSketchCustom() # other edge detection algorithm to create a fake drawing
    }

    # load data
    data_name = args['dataset']
    ds = datasets.load_from_disk(os.path.join('data', data_name), keep_in_memory=False)

    # define subset of artists
    subset = [
        "camille-pissarro",
        "paul-cezanne",
        "alfred-sisley",
        "edouard-manet",
        "eugene-boudin"
    ]

    # filter data for artists -> the filtered dataset still have 'old_indices' column that match to the rows->filtered_embeddings_FINAL mapping 
    ds_filtered = filter_data(ds, 'artist') 

    # remove sketches from dataset
    ds_filtered = filter_test_from_sketches(ds_filtered)

    # save the filtered dataframe to disk
    ds_filtered.save_to_disk(os.path.join('data', f"{data_name}_AUG_SUBSET"))

    # save batch size to re-use throughout
    batch_size = args['batch_size']

    for aug_name, aug in augmentations.items():

        try:
            augmented_images = []
            for idx in tqdm(range(len(ds_filtered)), desc=f"Augmenting [{aug_name}]"):
                img = ds_filtered[idx]['image'].convert("RGB")
                aug_img = aug(img).convert("RGB") # for grayscale images, we are not converting back to color - we give it three channels but it is still grayscaled
                augmented_images.append(aug_img)

                del img, aug_img

            # packaging the augmented images in a custom dataset class 
            ds_augmented = PILImageDataset(augmented_images)
            img_example = augmented_images[100]
            del augmented_images

            # sanity check image augmentation by saving sample image to disk
            aug_img_out_path = os.path.join('out', f'{data_name}_aug_img_ex')
            os.makedirs(aug_img_out_path, exist_ok=True)
            img_example.save(os.path.join(aug_img_out_path, f"{aug_name}.png"))

        except Exception as e:
            print(e)
            continue

        # create folder for saving the embeddings
        aug_folder_path = os.path.join('data', 'aug_embeddings', aug_name)
        os.makedirs(aug_folder_path, exist_ok=True)

        # then extract embeddings:
        try:
            for model_name in args['model_names']:
                if model_name == 'eva02_clip_336':
                    embeddings = embeddings_w_eva(ds_augmented, batch_size)  
                
                else:
                    model_path = add_model_prefix(model_name)
                    model_meta = mteb.get_model_meta(model_path)
                    print('LOADING MODEL...')
                    mteb_model = model_meta.load_model()
                    embeddings = extract_embeddings(ds_augmented, mteb_model, batch_size)
                    del mteb_model
                
        except Exception as e:
                print(f'Error processing model: {model_name}, {e}')
                continue # skip this model if error
            
        # save embeddings to disk
        aug_emb_model_path = os.path.join(aug_folder_path, f'{model_name}.pt')
        torch.save(embeddings, aug_emb_model_path)
        del embeddings 

        del ds_augmented
        import gc
        gc.collect()

if __name__ == '__main__':
    main()


