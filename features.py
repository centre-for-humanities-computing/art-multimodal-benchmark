import pandas as pd
import datasets
from datasets import Image as Image_ds # change name because of similar PIL module
from datasets import Dataset
import os
import requests 
from PIL import Image
from tqdm import tqdm
import torch
from transformers import pipeline
import diffusers
from datasets import load_dataset
from diffusers import AutoencoderKL
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

def test_diffusion_encoder(img, vae):
    # Example stats for standardization (ImageNet mean/std)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Compose preprocessing with standardization
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),             # Resize first (optional, adjust as needed)
        transforms.ToTensor(),                      # Convert PIL image to tensor (C, H, W) normalized to [0,1]
        transforms.Normalize(mean=mean, std=std)   # Standardize pixels per channel
    ])

    tensor = preprocess(img).unsqueeze(0)       # Add batch dimension -> (1, C, H, W)

    # Move tensor to same device as your model
    tensor = tensor.to(vae.device)

    # Run through encoder to get features
    features = vae.encoder(tensor)  # Assuming output shape: (1, 8, 64, 64) or similar

    # Adaptive average pooling to reduce spatial dims to 16x16
    adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
    pooled_features = adaptive_pool(features)    # (1, 8, 16, 16)

    # Flatten to get a feature vector of size 8*16*16 = 2048
    flattened = pooled_features.view(pooled_features.size(0), -1)  # (1, 2048)

    return flattened 

def main():

    ds = load_dataset("louisebrix/smk_canon_paintings", split="train") # all the data is in the 'train' split

    subset = ds.select(range(30))

    # 1. Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # initialize empty list
    embeddings = []

    # loop over each image in the dataset and extract feature embeddings
    for i in tqdm(range(len(subset)), desc="Extracting features from images"):
        try:
            img = subset[i]['image']
            feature = test_diffusion_encoder(img, vae)
            embeddings.append(feature.cpu().detach().numpy())
        
        except Exception as e:
            print(f"Error processing image {i}: {e}")

    np.save(os.path.join('data', 'features.npy'), embeddings, allow_pickle=True)

if __name__ == '__main__':
   main()