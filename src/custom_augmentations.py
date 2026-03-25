import datasets
import numpy as np
import os
from PIL import Image
#import torch
from tqdm import tqdm
from collections import Counter
import torchvision.transforms as T
import matplotlib.pyplot as plt
import random
import math
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import io
import cv2

class AddLayeredFrame:

    def __init__(self, border_sizes=(100, 100, 100), colors=None):

        self.border_sizes = border_sizes
        if colors is None:
            # default golden-brown variations (innermost → outermost)
            self.colors = [
                (205, 133, 63),  # innermost: golden-sienna
                (218, 165, 32),  # middle: goldenrod
                (184, 134, 11)   # outermost: dark golden brown
            ]
        else:
            self.colors = colors

    def __call__(self, img):
        w, h = img.size
        total_border = sum(self.border_sizes)  # total size needed for outermost layer

        # Create new canvas large enough for all layers
        new_w = w + 2 * total_border
        new_h = h + 2 * total_border
        canvas = Image.new("RGB", (new_w, new_h))

        # Draw all layers at once
        offset = 0
        draw = ImageDraw.Draw(canvas)
        for size, color in zip(self.border_sizes, self.colors):
            rect = [offset, offset, new_w - offset - 1, new_h - offset - 1]
            draw.rectangle(rect, fill=color)
            offset += size

        # Paste the original image in the center
        canvas.paste(img, (total_border, total_border))
        return canvas
# add JPEG-compression artifacts
class JPEGCompression:

    '''
    Add JPEG compression artifacts to image by re-encoding it at a low JPEG quality level.
    '''
    def __init__(self, quality=30):
        self.quality = quality

    def __call__(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality) # quality defines JPEG compression level (the lower the value, the worse the compression quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB") # return image with JPEG artifacts

# add vignette
class AddVignette:
    def __init__(self, strength=0.5, max_offset_ratio=0.1):
        """
        strength: 0-1, higher = darker edges
        max_offset_ratio: maximum fraction of width/height to shift the center randomly
        """
        self.strength = np.clip(strength, 0, 1)
        self.max_offset_ratio = max_offset_ratio

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32) / 255.0
        h, w = img_np.shape[:2]

        # Random center within ±max_offset_ratio of width/height
        max_dx = int(w * self.max_offset_ratio)
        max_dy = int(h * self.max_offset_ratio)
        center_x = w // 2 + np.random.randint(-max_dx, max_dx + 1)
        center_y = h // 2 + np.random.randint(-max_dy, max_dy + 1)

        # Compute radial distance mask
        y, x = np.ogrid[0:h, 0:w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        norm_distance = distance / max_dist

        # Apply quadratic falloff with fixed strength
        mask = 1 - self.strength * (norm_distance ** 2)

        # Apply mask to image
        if img_np.ndim == 3:
            img_np = img_np * mask[:, :, np.newaxis]
        else:
            img_np = img_np * mask

        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)

# add graining
class AddGrain:
    def __init__(self, mean=0.0, std=0.05):
        """
        mean: mean of Gaussian noise
        std: standard deviation, controls grain strength (0-1)
        """
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32) / 255.0

        # Generate Gaussian noise
        noise = np.random.normal(self.mean, self.std, img_np.shape)

        # Add noise and clip
        img_np = np.clip(img_np + noise, 0.0, 1.0)

        # Convert back to PIL
        img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)

# random light artifacts

class AddLightArtifact:
    def __init__(self, max_intensity=0.4, max_radius_ratio=0.4):
        """
        max_intensity: maximum brightness to add (0-1)
        max_radius_ratio: max size of light patch relative to image dimension
        """
        self.max_intensity = max_intensity
        self.max_radius_ratio = max_radius_ratio

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32) / 255.0
        h, w = img_np.shape[:2]

        # Random center of light
        center_x = np.random.randint(int(w * 0.2), int(w * 0.8))
        center_y = np.random.randint(int(h * 0.2), int(h * 0.8))

        # Random radius
        max_radius = int(min(h, w) * self.max_radius_ratio)
        radius = np.random.randint(max_radius // 2, max_radius)

        # Create mask
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse(
            (center_x - radius, center_y - radius,
             center_x + radius, center_y + radius),
            fill=int(255 * self.max_intensity)
        )
        mask = mask.filter(ImageFilter.GaussianBlur(radius // 2))
        mask_np = np.array(mask).astype(np.float32) / 255.0  # normalize

        # Apply mask as additive light
        if img_np.ndim == 3:
            img_np = np.clip(img_np + mask_np[:, :, np.newaxis], 0, 1)
        else:
            img_np = np.clip(img_np + mask_np, 0, 1)

        img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)

class RelativeGaussianBlur:
    def __init__(self, strength=0.02, sigma=None):
        """
        strength: fraction of min(width, height) for kernel size
        sigma: if None, use 0 (OpenCV auto)
        """
        self.strength = strength
        self.sigma = sigma

    def __call__(self, img):
        # Convert PIL to numpy
        img_np = np.array(img)

        h, w = img_np.shape[:2]

        # Compute kernel size relative to image
        k = int(self.strength * min(w, h))
        k = max(3, k)
        if k % 2 == 0:
            k += 1  # must be odd

        # Apply Gaussian blur using OpenCV (fast, even for large kernels)
        if img_np.ndim == 2:  # grayscale
            blurred = cv2.GaussianBlur(img_np, (k, k), sigmaX=self.sigma or 0)
        else:  # RGB
            blurred = cv2.GaussianBlur(img_np, (k, k), sigmaX=self.sigma or 0)

        return Image.fromarray(blurred)
    
