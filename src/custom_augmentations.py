import datasets
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt 
import torchvision.transforms as T
from PIL import ImageOps
import io
from PIL import Image, ImageDraw, ImageFilter

# add brown boarder, simulating a frame
class AddFrame:
    def __init__(self, border_size=150, color=(70, 55, 35)): # default settings
        self.border_size = border_size
        self.color = color

    def __call__(self, img): # (making class object behave like a function) (torchvision Compose expects callable objects)
        return ImageOps.expand(img, border=self.border_size, fill=self.color)

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
    
