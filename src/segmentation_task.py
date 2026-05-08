"""
Augment images and apply tree segmentation framework
"""
import os
import pandas as pd 
import torch
import argparse
import random
import math
import torchvision.transforms as T
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import login
import sam3
from PIL import Image
from sam3.sam3.model_builder import build_sam3_image_model
from sam3.sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.sam3.model.sam3_image_processor import Sam3Processor
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm
import io
import ast

import sys
sys.path.insert(0, 'src')
from custom_augmentations import AddLayeredFrame, JPEGCompression, AddVignette, AddGrain, AddLightArtifact, RelativeGaussianBlur, FixedContrast, CannySketch, PencilSketchCustom
sys.path.pop(0)

def segment_image(model, img: Image.Image, prompt: str, confidence_threshold: float):
    
    """
    Segment an image using a SAM3 model guided by a text prompt.

    Args:
        model: Loaded SAM3 model instance used to initialize the processor.
        img: PIL image to segment.
        prompt: Text description of the target object to segment.
        confidence_threshold: Minimum confidence score for an object to be included in the output.

    Returns:
        Segmentation output containing masks and associated metadata.
    """
    # load Sam3 model with weights from folder
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)

    # pass the PIL image to be processed
    inference_state = processor.set_image(img)

    # prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    return output

def inference_on_df(df: pd.DataFrame, image_column: str, prompt: str, confidence_threshold: float, model, aug: Callable, aug_name: str) -> None:

    """
    Run SAM3 segmentation on all images in a DataFrame and save results to CSV.

    Iterates over each row, decodes the image bytes, applies an augmentation,
    and segments the result. Failures are caught per-row and recorded as pd.NA
    so the rest of the batch is unaffected. Results are written to
    data/segmentations/<aug_name>_segmented.csv.

    Args:
        df: DataFrame where each row represents one image sample.
        image_column: Name of the column containing image bytes stored as strings.
        prompt: Text description of the target object(s) to segment.
        confidence_threshold: Minimum confidence score for a segment to be included in the output.
        model: Loaded SAM3 model instance.
        aug: Augmentation callable that takes and returns a PIL image.
        aug_name: Name for the augmentation, used in output filenames and logged in results.
    """
    # initialize empty list to append results to
    output_lists = []

    # loop over data folder
    for idx, row in tqdm(df.iterrows(), desc = 'Segmenting images', total=len(df)):

        image_bytes = row[image_column]
        image_bytes = ast.literal_eval(image_bytes) # stored as strings instead of actual bytes

        try:
            # convert image bytes to PIL image
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # augment image
            augmented_img = aug(pil_image)

            if augmented_img.mode != "RGB":
                augmented_img = augmented_img.convert("RGB")

            # segment image with sam3
            output = segment_image(model, augmented_img, prompt, confidence_threshold)

            # save output in a pandas-friendly format
            output_dict = {'filename': row['filename'],
                            'aug': aug_name,
                            'scores': output['scores'].tolist(),
                            'original_height': output['original_height'],
                            'original_width': output['original_width'],
                            'boxes': output['boxes'].tolist(),
                            'count': len(output['scores'])}

        
        # if for some reason anything fails, just append pd.NA
        except Exception as e:
            print(f"Error processing row no. {idx}. Reason: {e}")

            output_dict = {'filename': row['filename'],
                            'aug': aug_name,
                            'scores': pd.NA,
                            'original_height': pd.NA,
                            'original_width': pd.NA,
                            'boxes': pd.NA,
                            'count': pd.NA}
            
        output_lists.append(output_dict)

        del pil_image
        del augmented_img

    # Convert list of dicts to DataFrame
    output_df = pd.DataFrame(output_lists)

    segmentation_path = os.path.join('data', 'segmentations')
    os.makedirs(segmentation_path, exist_ok=True)

    output_df.to_csv(os.path.join(segmentation_path, f'{aug_name}_segmented.csv'))

def main():
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # reads read HuggingFace token
    load_dotenv()  # reads .env file
    hf_token = os.getenv("hf_token")
    login()

    # load SAM3 model (if there's any issues with running the model, it's likely this path)
    bpe_path = os.path.join('sam3', 'sam3', 'assets', 'bpe_simple_vocab_16e6.txt.gz')

    model = build_sam3_image_model(bpe_path=bpe_path)

    df = pd.read_csv(os.path.join('data', 'sample_200_paintings.csv'))

    augmentations = {
       'weak_blur': RelativeGaussianBlur(strength=0.3, sigma=5), # weak blurring
        'strong_blur': RelativeGaussianBlur(strength=0.7, sigma=7), # stronger blurring
        'grayscale': T.Grayscale(), # grayscale
        'contrast': FixedContrast(factor = 10), # changing contrasts
        'frame': AddLayeredFrame(border_sizes=(100, 100, 100)), # frame
        'jpeg_compr': JPEGCompression(quality=6), # jpeg compression
        'vignette': AddVignette(strength = 0.9), # adding vignette
        'weak_grain': AddGrain(std=0.3),
        'strong_grain': AddGrain(std=0.7),
        'light_artifact': AddLightArtifact(max_intensity=0.4, max_radius_ratio=0.4),
        'Canny_sketch': CannySketch(),
        'pencil_sketch': PencilSketchCustom()
    }

    for aug_name, aug in augmentations.items():
        print(f'Augmentation + segmentation: {aug_name}')
        inference_on_df(df, 'image', 'tree', 0.4, model, aug, aug_name)

if __name__ == '__main__':
    main()