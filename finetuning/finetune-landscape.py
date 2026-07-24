"""
Goal: given a frozen, pretrained vision backbone (CLIP/DINOv2/SigLIP/etc.),
train a small projection head on top of it so that images by the same artist
end up close together in embedding space (by cosine similarity), and images
by different artists end up far apart — a metric-learning setup, not a
classifier. Probed on both images from training classes and entire new classes as well.
Multiple models train in parallel via multiprocessing, one process per model,
throttled to one job per GPU at a time.


Pipeline, per model:
  1. load_hf_splits()        - pull train/probe rows from the HF dataset.
  2. load_model()             - load + freeze the pretrained backbone.
  3. build_descriptor_cache() - run every image through the frozen backbone
                                 once, cache the resulting descriptor.
  4. Per epoch:
       mine_triplets()  - pick triplets (anchor, positive, negative) using semi-hard mining
       train the head   - triplet_loss() on those triplets.
       evaluate()       - retrieval metrics (R@K) on the training gallery.
       probe_evaluate() - retrieval metrics on held-out probe images, split into
                            "known artist" (overfitting check)
                            "new artist" (generalisation check)
  5. Save the best-epoch head weights as a checkpoint.

"""

import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import traceback
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm


# --- CONFIG --- #

# Data source and paths

HF_REPO_ID       = "chcaa/finetuning-landscape-painting"
TRAIN_SPLIT_NAME = "train"
PROBE_SPLIT_NAME = "probe"

IMAGE_BYTES_COLUMN  = "image_bytes" # Column names in the parquet files
ARTIST_COLUMN       = "artist" # Column names in the parquet files

OUTPUT_DIR = "checkpoints"
CACHE_DIR  = "cache"
LOG_DIR    = "logs"
RUN_MODELS = ["DINOv2-base"] # add others from the list or keep empty to include all
CLEAR_CACHE = True  # delete descriptor cache and HF model weights after each job

# Checkpoint selection

CHECKPOINT_CRITERION = "r@1" # r@1: training R@1; probe_known_r@1: R@1 on probe images of known artists

# Training hyperparameters and architecture (defaults; per-model overrides below)
EPOCHS          = 20
PAIRS_PER_EPOCH = 1500  # semi-hard triplets sampled per epoch
BATCH_SIZE      = 32    # triplets per gradient step
MARGIN          = 0.5   # triplet loss margin
WARMUP_EPOCHS   = 0     # linear LR warmup epochs; 0 = no warmup

PROJ_DIM       = 512  # output dimension of the projection head
GRID_SIZE      = 4    # NxN grid of crops per image

SEED = 42             # NOTE: add a seed for reproducability. Note that the results might 
                      # still differ from GPU/CPU to GPU/CPU.

# Pooling

POOLING_METHOD = "stats"    # "attention" (scaled dot-product -> softmax -> weighted sum) or "stats" (mean, std, and mean-square)
PATCHES_SAMPLE = 64             # only used by "stats" pooling
TEMPERATURE    = 0.5            # only used by "attention" pooling; lower = sharper
POOLING_TAGS   = {"attention": "attnpool", "stats": "statspool"}
POOLING_TAG    = POOLING_TAGS[POOLING_METHOD]

# --- MODEL REGISTRY --- #

MODELS = [
    {"name": "CLIP-ViT-B-16-DataComp.XL",  "backend": "open_clip", "model_name": "ViT-B-16",                        "pretrained": "datacomp_xl_s13b_b90k", "lr": 1e-4, "epochs": 30},
    {"name": "CLIP-ViT-L-14-DataComp.XL",  "backend": "open_clip", "model_name": "ViT-L-14",                        "pretrained": "datacomp_xl_s13b_b90k", "lr": 5e-4, "epochs": 30},
    {"name": "CLIP-ViT-bigG-14-laion2B",   "backend": "open_clip", "model_name": "ViT-bigG-14",                     "pretrained": "laion2b_s39b_b160k",    "lr": 1e-4, "epochs": 30},
    {"name": "DINOv2-base",                 "backend": "dinov2",    "model_name": "facebook/dinov2-base",             "pretrained": None,                    "lr": 2e-5, "epochs": 20},
    {"name": "DINOv2-giant",                "backend": "dinov2",    "model_name": "facebook/dinov2-giant",            "pretrained": None,                    "lr": 2e-5, "epochs": 20},
    {"name": "EVA02-CLIP-L-14-336",         "backend": "open_clip", "model_name": "EVA02-L-14-336",                  "pretrained": "merged2b_s6b_b61k",     "lr": 5e-5, "epochs": 20},
    {"name": "SigLIP-B-patch16-224",        "backend": "open_clip", "model_name": "hf-hub:timm/ViT-B-16-SigLIP",    "pretrained": "",                      "lr": 7e-5, "epochs": 12, "pairs_per_epoch": 1200},
    {"name": "SigLIP-L-patch16-384",        "backend": "open_clip", "model_name": "hf-hub:timm/ViT-L-16-SigLIP-384","pretrained": "",                      "lr": 5e-5, "epochs": 14, "pairs_per_epoch": 1200},
    {"name": "SigLIP-SO400M-patch14-384",   "backend": "open_clip", "model_name": "ViT-SO400M-14-SigLIP-384",       "pretrained": "webli",                 "lr": 5e-5, "epochs": 20, "warmup_epochs": 1},
]

# --- FUNCTIONS --- #

# Dataset loading

def load_hf_splits():
    """Load the train and probe splits from the HF dataset. Datasets caches the
    parquet files on disk after the first download, so repeated calls are quicker."""
    hf_train = load_dataset(HF_REPO_ID, split=TRAIN_SPLIT_NAME)
    hf_probe = load_dataset(HF_REPO_ID, split=PROBE_SPLIT_NAME)
    return hf_train, hf_probe


# Utilities

def job_id(model_cfg: dict) -> str:
    """Filename stem shared by a model's descriptor cache and checkpoint —
    encodes the model name + POOLING_TAG."""
    return f"{model_cfg['name']}_patches_{POOLING_TAG}"


def build_a2i(hf_split, key_prefix: str, desc_cache: dict) -> dict[str, list[str]]:
    """Return {artist_name: [key, ...]} for all rows in a HF split, where each
    key is f"{key_prefix}::{row_index}"."""
    if hf_split is None:
        return {}
    a2i: dict[str, list[str]] = defaultdict(list)
    artists = hf_split[ARTIST_COLUMN]
    for i, artist in enumerate(artists):
        key = f"{key_prefix}::{i}"
        if key in desc_cache:
            a2i[artist].append(key)
    return dict(a2i)


# Logging

class JobLogger:
    """Writes log lines to stdout and to a per-job .log file simultaneously."""

    def __init__(self, job_name: str, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.job_name = job_name
        self._f = open(log_dir / f"{job_name}.log", "w", buffering=1)

    def write(self, msg: str):
        line = f"[{self.job_name}] {msg}"
        print(line, flush=True)
        self._f.write(line + "\n")

    def close(self):
        self._f.close()


# Model loading

def load_model(cfg: dict, device: str):
    """Load and freeze a pretrained backbone. Return (model, normalizer, image_size, input_dim).
    Input_dim is measured by a dummy forward pass through style_descriptor()."""
    if cfg["backend"] == "open_clip":
        import open_clip
        # Extracting the Normalize transform from the preprocess pipeline
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["model_name"], pretrained=cfg["pretrained"] or None, device=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        image_size = model.visual.image_size if hasattr(model.visual, "image_size") else 224
        if isinstance(image_size, (tuple, list)):
            image_size = image_size[0]
        normalizer = next(t for t in preprocess.transforms if isinstance(t, T.Normalize))

    elif cfg["backend"] == "dinov2":
        from transformers import AutoImageProcessor, AutoModel
        processor  = AutoImageProcessor.from_pretrained(cfg["model_name"])
        model      = AutoModel.from_pretrained(cfg["model_name"]).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        image_size = processor.size.get("height", 224)
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    else:
        raise ValueError(f"Unknown backend: {cfg['backend']}")

    dummy     = torch.zeros(1, 3, int(image_size), int(image_size), device=device)
    input_dim = style_descriptor(cfg["backend"], model, dummy, device).shape[-1]

    return model, normalizer, int(image_size), int(input_dim)


# Image loading

def load_image_from_bytes(image_bytes: bytes, image_size: int,
                          norm: T.Normalize, device: str = "cpu") -> torch.Tensor:
    """Decode raw image bytes (from the HF dataset's image_bytes column) and return
    (N_crops, C, H, W) normalised grid crops. The image is split into a
    GRID_SIZE×GRID_SIZE grid of non-overlapping square cells.
    Crops are bicubic-resized to image_size if needed."""
    img       = Image.open(BytesIO(image_bytes)).convert("RGB")
    cell_size = min(img.size) // GRID_SIZE          # largest square cell that fits
    img_t     = T.ToTensor()(img).to(device)
    _, H, W   = img_t.shape
    crops = torch.stack([
        img_t[:, r * cell_size:(r + 1) * cell_size,
                 c * cell_size:(c + 1) * cell_size]
        for r in range(H // cell_size) for c in range(W // cell_size)
    ])
    if cell_size != image_size:
        crops = F.interpolate(crops, size=(image_size, image_size),
                              mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor(norm.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(norm.std,  dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (crops - mean) / std


# Feature extraction

def _vit_tokens_open_clip(model, imgs: torch.Tensor) -> tuple:
    """Return (cls_tokens (B, D), patch_tokens (B, N, D)) from an open_clip ViT.
    Two variants are handled: conv1-style (standard ViT) and trunk-style (timm/SigLIP).
    Some timm-backed models (e.g. SigLIP) have no CLS token at all — for those,
    the mean of the patch tokens is used as the attention query instead."""
    visual = model.visual
    if hasattr(visual, "conv1"):
        # Patchify - prepend CLS - add positional embeddings - transformer
        x   = visual.conv1(imgs)
        x   = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x   = torch.cat([cls, x], dim=1) + visual.positional_embedding
        if hasattr(visual, "ln_pre"):
            x = visual.ln_pre(x)
        x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        return x[:, 0, :], x[:, 1:, :]
    if hasattr(visual, "trunk"):
        # timm-backed models expose forward_features directly
        x = visual.trunk.forward_features(imgs)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        has_cls = getattr(visual.trunk, "cls_token", None) is not None
        if has_cls and x.shape[1] > 1:
            return x[:, 0, :], x[:, 1:, :]
        return x.mean(dim=1), x  # no CLS token use mean as query
    raise RuntimeError(f"Unrecognised visual type '{type(visual).__name__}'")


def _get_cls_and_patch_tokens(backend: str, model, imgs: torch.Tensor) -> tuple:
    """Return (cls_tokens (B, D), patch_tokens (B, N, D)) for any supported backend."""
    if backend == "open_clip":
        return _vit_tokens_open_clip(model, imgs)
    # HuggingFace DINOv2: index 0 is the CLS token, 1: are patch tokens
    out = model(pixel_values=imgs).last_hidden_state
    return out[:, 0, :], out[:, 1:, :]


def _attention_pool_crop(cls: torch.Tensor, patch_tokens: torch.Tensor,
                         temperature: float) -> torch.Tensor:
    """cls: (D,)  patch_tokens: (N, D)  -> pooled (D,)
    Scaled dot-product attention: CLS token as query, patch tokens as key/value."""
    q = F.normalize(cls, dim=-1)
    k = F.normalize(patch_tokens, dim=-1)
    weights = F.softmax((k @ q) / temperature, dim=0)  # (N,)
    return (weights.unsqueeze(-1) * patch_tokens).sum(dim=0)


def _stats_pool(patch_tokens: torch.Tensor) -> torch.Tensor:
    """patch_tokens: (B, N, D) patch tokens across all crops of one image
    -> (3*D,) via mean/std/mean squared over a random subsample of PATCHES_SAMPLE*B
    patches."""
    flat = F.normalize(patch_tokens, dim=-1).reshape(-1, patch_tokens.shape[-1])
    n    = min(PATCHES_SAMPLE * patch_tokens.shape[0], flat.shape[0])
    idx  = torch.randperm(flat.shape[0], device=flat.device)[:n]
    flat = flat[idx]
    return torch.cat([flat.mean(0), flat.std(0), (flat**2).mean(0)])


@torch.no_grad()
def style_descriptor(backend: str, model, imgs: torch.Tensor, device: str) -> torch.Tensor:
    """Compute a (1, D) style descriptor from a batch of crops, using whichever."""
    cls_tokens, patch_tokens = _get_cls_and_patch_tokens(backend, model, imgs.to(device))

    if POOLING_METHOD == "attention":
        pooled = torch.stack([
            _attention_pool_crop(cls_tokens[i], patch_tokens[i], TEMPERATURE)
            for i in range(cls_tokens.shape[0])
        ]).mean(dim=0)  # (D,) — per-crop vectors averaged across crops
    elif POOLING_METHOD == "stats":
        pooled = _stats_pool(patch_tokens)  # (3*D,) — pooled jointly across crops
    else:
        raise ValueError(f"Unknown POOLING_METHOD: {POOLING_METHOD!r}")

    return F.normalize(pooled, dim=-1).unsqueeze(0)


# Descriptor cache

def _dataset_signature(hf_train, hf_probe) -> dict:
    """A fingerprint of the dataset to ensure it's the same (in case of updates on HF)."""
    return {
        "train_fingerprint": hf_train._fingerprint,
        "train_len":         len(hf_train),
        "probe_fingerprint": hf_probe._fingerprint,
        "probe_len":         len(hf_probe),
    }


def build_descriptor_cache(model_cfg: dict, hf_train, hf_probe,
                            model, normalizer, image_size: int,
                            device: str, cache_dir: Path, log) -> dict:
    """Pre-compute and save {key: descriptor} for all training and probe rows,
    where key is f"{split}::{row_index}". Caching avoids re-running the frozen
    backbone every epoch. Probe rows are included so probe_evaluate() can use
    them without reloading the backbone."""
    cache_path = cache_dir / f"{job_id(model_cfg)}.pt"
    current_sig = _dataset_signature(hf_train, hf_probe)

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        cached_sig = payload.get("signature")
        if cached_sig == current_sig:
            log.write(f"Loading cache from {cache_path} (dataset signature matches)")
            return payload["descriptors"]
        else:
            log.write(
                f"Cache at {cache_path} is STALE (dataset changed since it was built) "
                f"— rebuilding. cached={cached_sig}  current={current_sig}"
            )

    log.write("Building descriptor cache...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Both splits are cached together in one file per model, keyed by
    # "{split}::{row_index}" — e.g. "train::42", "probe::7". These keys are
    # opaque identifiers used everywhere else in this script (mine_triplets,
    # evaluate, embed_images, ...); this assumes row order is stable across
    # repeated load_dataset() calls for the same dataset version.
    splits = [(TRAIN_SPLIT_NAME, hf_train), (PROBE_SPLIT_NAME, hf_probe)]

    desc_cache = {}
    for prefix, hf_split in splits:
        for i in tqdm(range(len(hf_split)), desc=f"  Caching ({prefix})", unit="img", leave=False):
            image_bytes = hf_split[i][IMAGE_BYTES_COLUMN]
            feat = style_descriptor(
                model_cfg["backend"], model,
                load_image_from_bytes(image_bytes, image_size, normalizer, device=device),
                device,
            )
            desc_cache[f"{prefix}::{i}"] = feat.squeeze(0).cpu()  # store as (D*3,) on CPU

    torch.save({"signature": current_sig, "descriptors": desc_cache}, cache_path)
    log.write(f"Cached {len(desc_cache)} images - {cache_path}")
    return desc_cache


# Embedding helper

@torch.no_grad()
def embed_images(keys: list[str], desc_cache: dict,
                 head: nn.Module, device: str) -> torch.Tensor:
    """Project cached descriptors through the head. Returns (N, PROJ_DIM) on CPU."""
    head.eval()
    return torch.stack([
        head(desc_cache[k].to(device).unsqueeze(0)).squeeze(0).cpu()
        for k in keys
    ])


# Semi-hard negative mining

def mine_triplets(hf_train, desc_cache: dict, head: nn.Module,
                  device: str, pairs: int) -> list:
    """Return `pairs` (anchor, positive, negative) key triplets via semi-hard mining.
    Semi-hard negatives are from a different artist but still easier than the
    positive: sim(a, neg) < sim(a, pos). This prevents trivially easy negatives
    and avoids the instability of always-hardest negatives.
    Falls back to the hardest negative when no semi-hard candidate exists."""
    a2i         = build_a2i(hf_train, TRAIN_SPLIT_NAME, desc_cache)
    artists     = list(a2i.keys())
    train_keys  = [k for keys in a2i.values() for k in keys]
    embs        = embed_images(train_keys, desc_cache, head, device).to(device)
    key_to_idx  = {k: i for i, k in enumerate(train_keys)}
    artist_ids  = [a for a, keys in a2i.items() for k in keys]  # label per key

    sim_matrix  = embs @ embs.T                     # (N, N) full pairwise cosine similarities
    # NOTE: this same_artist mask is built with a pure-Python double loop —
    # O(N^2) comparisons in Python, not vectorised.
    same_artist = torch.tensor(
        [[artist_ids[i] == artist_ids[j] for j in range(len(train_keys))]
         for i in range(len(train_keys))],
        dtype=torch.bool, device=device,
    )

    triplets = []
    for _ in range(pairs):
        anc_artist        = random.choice(artists)
        anc_key, pos_key  = random.sample(a2i[anc_artist], 2)
        anc_idx           = key_to_idx[anc_key]

        neg_sims = sim_matrix[anc_idx].clone()
        neg_sims[same_artist[anc_idx]] = -2.0       # mask self + all same-artist images

        # Keep only negatives that are easier than the positive (semi-hard zone)
        sim_pos   = sim_matrix[anc_idx, key_to_idx[pos_key]].item()
        semi_mask = (neg_sims < sim_pos) & (neg_sims > -2.0)
        if semi_mask.any():
            neg_sims[~semi_mask] = -2.0             # suppress everything outside the zone

        triplets.append((anc_key, pos_key, train_keys[neg_sims.argmax().item()]))

    return triplets


# Triplet Dataset

class TripletDataset(Dataset):
    """Serves pre-built (anchor, positive, negative) triplets from the descriptor cache."""
    def __init__(self, triplets: list, desc_cache: dict):
        self.triplets = triplets
        self.cache    = desc_cache

    def __len__(self): return len(self.triplets)

    def __getitem__(self, i):
        return tuple(self.cache[k] for k in self.triplets[i])


# Projection head & loss

class StyleProjectionHead(nn.Module):
    """Two-layer MLP: Linear - GELU - LayerNorm - Linear - L2-normalise.
    Maps frozen backbone descriptors to a compact PROJ_DIM embedding space.
    LayerNorm stabilises training across models with different descriptor scales.
    L2-normalisation makes cosine similarity equivalent to dot product."""
    def __init__(self, input_dim: int, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim), nn.GELU(),
            nn.LayerNorm(proj_dim), nn.Linear(proj_dim, proj_dim),
        )
    def forward(self, x): return F.normalize(self.net(x), dim=-1)


def triplet_loss(a, p, n, margin: float = 0.5):
    """Cosine-distance triplet loss: max(0, dist(a,p) - dist(a,n) + margin)."""
    return F.relu(1 - F.cosine_similarity(a, p)
                  - (1 - F.cosine_similarity(a, n)) + margin).mean()


# LR scheduler

class WarmupCosineLR(_LRScheduler):
    """Cosine LR decay with optional linear warm-up.
    Warm-up ramps LR from 0-  base_lr over the first warmup_epochs epochs,
    then cosine-decays from base_lr - min_lr for the remaining epochs."""
    def __init__(self, optimizer, total_epochs, warmup_epochs=0, min_lr=1e-6, last_epoch=-1):
        self.total_epochs  = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr        = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch + 1
        return [
            base_lr * e / self.warmup_epochs                           # linear warm-up
            if self.warmup_epochs > 0 and e <= self.warmup_epochs else
            self.min_lr + 0.5 * (base_lr - self.min_lr) *             # cosine decay
            (1 + math.cos(math.pi * max(e - self.warmup_epochs, 0) /
                          max(self.total_epochs - self.warmup_epochs, 1)))
            for base_lr in self.base_lrs
        ]


# --- EVALUATION --- #

RECALL_KS       = (1, 5, 10)  # K values for retrieval metrics
N_EVAL_TRIPLETS = 1000        # random triplets used to estimate triplet accuracy


def recall_at_k(query_emb: torch.Tensor, gallery_embs: torch.Tensor,
                gallery_labels: list, true_label, ks: tuple,
                exclude_idx: int = None) -> dict:
    """Cosine-similarity top-k retrieval for one query against a gallery.
    Returns {k: True/False} — whether true_label appears among the top-k gallery labels by similarity."""
    sims = F.cosine_similarity(query_emb.unsqueeze(0), gallery_embs)
    if exclude_idx is not None:
        sims[exclude_idx] = -2.0
    max_k = min(max(ks), len(gallery_labels) - (1 if exclude_idx is not None else 0))
    top   = [gallery_labels[j] for j in torch.topk(sims, max_k).indices]
    return {k: true_label in top[:k] for k in ks}


@torch.no_grad()
def evaluate(head: nn.Module, desc_cache: dict, hf_train, device: str) -> dict:
    """Compute training-set metrics: triplet accuracy, similarity gap, and R@K.
    Triplet accuracy: fraction of random (a, p, n) where sim(a,p) > sim(a,n).
    sim_gap: mean_pos_sim - mean_neg_sim — the key separation signal.
    R@K: for each image, fraction of queries where >=1 same-artist image appears in top K."""
    head.eval()
    a2i = build_a2i(hf_train, TRAIN_SPLIT_NAME, desc_cache)
    if len(a2i) < 2:
        return {}

    all_keys    = [k for keys in a2i.values() for k in keys]
    all_artists = [a for a, keys in a2i.items() for k in keys]  # label per embedding
    embs        = embed_images(all_keys, desc_cache, head, device)  # (N, PROJ_DIM)
    emb_map     = dict(zip(all_keys, embs))
    artists     = list(a2i.keys())

    # Triplet accuracy & similarity statistics
    correct = 0
    pos_sims, neg_sims = [], []
    for _ in range(N_EVAL_TRIPLETS):
        anc    = random.choice(artists)
        a_imgs = a2i[anc]
        a_p, p_p = (random.sample(a_imgs, 2) if len(a_imgs) >= 2 else (a_imgs[0], a_imgs[0]))
        n_p    = random.choice(a2i[random.choice([x for x in artists if x != anc])])
        sp = F.cosine_similarity(emb_map[a_p].unsqueeze(0), emb_map[p_p].unsqueeze(0)).item()
        sn = F.cosine_similarity(emb_map[a_p].unsqueeze(0), emb_map[n_p].unsqueeze(0)).item()
        pos_sims.append(sp); neg_sims.append(sn)
        correct += sp > sn

    mean_pos = sum(pos_sims) / len(pos_sims)
    mean_neg = sum(neg_sims) / len(neg_sims)

    # R@K retrieval
    hits = {k: 0 for k in RECALL_KS}
    for i, label in enumerate(all_artists):
        for k, hit in recall_at_k(embs[i], embs, all_artists, label,
                                  RECALL_KS, exclude_idx=i).items():
            hits[k] += hit

    return {
        "triplet_acc":  correct / N_EVAL_TRIPLETS,
        "mean_pos_sim": mean_pos,
        "mean_neg_sim": mean_neg,
        "sim_gap":      mean_pos - mean_neg,
        **{f"r@{k}": hits[k] / len(all_artists) for k in RECALL_KS},
    }


@torch.no_grad()
def probe_evaluate(head: nn.Module, desc_cache: dict,
                   hf_train, hf_probe, device: str) -> dict:
    """Evaluate on held-out probe rows.

    Known artists (probe artist matches a training artist):
      Each probe image queries the full training gallery. Measures overfitting.

    New artists (probe artist not in training set):
      Measures zero-shot generalisation.
    """
    head.eval()

    probe_a2i = build_a2i(hf_probe, PROBE_SPLIT_NAME, desc_cache)
    if not probe_a2i:
        return {}

    # Build the full training gallery matrix once
    train_a2i       = build_a2i(hf_train, TRAIN_SPLIT_NAME, desc_cache)
    gallery_keys    = [k for keys in train_a2i.values() for k in keys]
    gallery_artists = [a for a, keys in train_a2i.items() for k in keys]
    gallery_mat     = embed_images(gallery_keys, desc_cache, head, device).to(device)

    train_names = set(train_a2i.keys())
    known_a2i   = {a: v for a, v in probe_a2i.items() if a in train_names}
    new_a2i     = {a: v for a, v in probe_a2i.items() if a not in train_names}
    results     = {}

    # Known artists
    if known_a2i:
        hits, n_q = {k: 0 for k in RECALL_KS}, 0
        for artist, keys in known_a2i.items():
            for q_key in keys:
                q_emb = head(desc_cache[q_key].to(device).unsqueeze(0)).squeeze(0)
                for k, hit in recall_at_k(q_emb, gallery_mat, gallery_artists,
                                          artist, RECALL_KS).items():
                    hits[k] += hit
                n_q += 1
        if n_q:
            results.update({f"probe_known_r@{k}": hits[k] / n_q for k in RECALL_KS})

    # New artists
    if new_a2i and sum(len(v) for v in new_a2i.values()) > 1:
        all_probe_keys    = [k for keys in new_a2i.values() for k in keys]
        all_probe_artists = [a for a, keys in new_a2i.items() for k in keys]
        probe_embs        = embed_images(all_probe_keys, desc_cache, head, device)
        hits, n_q = {k: 0 for k in RECALL_KS}, 0
        for i, (q_key, q_artist) in enumerate(zip(all_probe_keys, all_probe_artists)):
            if sum(1 for a in all_probe_artists if a == q_artist) < 2:
                continue
            for k, hit in recall_at_k(probe_embs[i], probe_embs, all_probe_artists,
                                      q_artist, RECALL_KS, exclude_idx=i).items():
                hits[k] += hit
            n_q += 1
        if n_q:
            results.update({f"probe_new_r@{k}": hits[k] / n_q for k in RECALL_KS})

    return results


# HuggingFace backbone cache cleanup

def clear_hf_cache(model_cfg: dict, log):
    """Delete downloaded backbone weights to reclaim disk space."""
    cache_root = Path(os.environ.get(
        "HUGGINGFACE_HUB_CACHE",
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")))
    name = model_cfg["model_name"]

    if model_cfg["backend"] == "dinov2":
        candidates = [cache_root / ("models--" + name.replace("/", "--"))]
    elif name.startswith("hf-hub:"):
        candidates = [cache_root / ("models--" + name.replace("hf-hub:", "").replace("/", "--"))]
    else:
        oc_cache   = Path(os.environ.get("OPEN_CLIP_CACHE", cache_root))
        candidates = list(oc_cache.glob(f"*{name.replace('/', '--').replace(' ', '-')}*"))

    # shutil.rmtree returns None on success; collect paths of what was deleted
    deleted = [str(c) for c in candidates if c.exists() and shutil.rmtree(c) is None]
    log.write(f"Cleared HF model cache: {deleted}" if deleted
              else f"HF model cache: nothing found for {name}")


# Worker

def worker(job_args: tuple):
    """Multiprocessing entry point. Acquires the per-GPU semaphore before training
    so at most one job runs per GPU at a time, then releases it on exit."""
    model_cfg, output_dir, cache_dir, log_dir, \
        gpu_id, clear_cache, ckpt_criterion, gpu_sem = job_args

    gpu_sem.acquire()
    try:
        return _worker_inner(
            model_cfg, output_dir, cache_dir, log_dir,
            gpu_id, clear_cache, ckpt_criterion)
    finally:
        gpu_sem.release()


def _worker_inner(model_cfg, output_dir, cache_dir, log_dir,
                  gpu_id, clear_cache, ckpt_criterion):
    """Full training loop for one model"""
    # Each worker is a separate spawned process, so it must seed itself —
    # a seed set in the parent process would not carry over here.
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device      = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    # Scope this process to one GPU; gpu_id is then always 0 inside CUDA calls
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    job_name    = job_id(model_cfg)
    log         = JobLogger(job_name, log_dir)
    output_path = output_dir / f"{job_name}.pt"

    if output_path.exists():
        log.write("Skipping — checkpoint already exists")
        log.close()
        return job_name, 0.0, None

    try:
        t0 = time.time()
        log.write(f"GPU {gpu_id}  pooling={POOLING_METHOD}")

        log.write(f"Loading HF dataset: {HF_REPO_ID}")
        hf_train, hf_probe = load_hf_splits()

        # Per-model keys override the global defaults when present
        epochs          = model_cfg.get("epochs",          EPOCHS)
        warmup_epochs   = model_cfg.get("warmup_epochs",   WARMUP_EPOCHS)
        pairs_per_epoch = model_cfg.get("pairs_per_epoch", PAIRS_PER_EPOCH)

        model, normalizer, image_size, input_dim = load_model(model_cfg, device)
        log.write(f"image_size={image_size}  input_dim={input_dim}  "
                  f"epochs={epochs}  pairs={pairs_per_epoch}  "
                  f"batch={BATCH_SIZE}  margin={MARGIN}")

        desc_cache = build_descriptor_cache(
            model_cfg, hf_train, hf_probe, model, normalizer,
            image_size, device, cache_dir, log)
        # Backbone no longer needed — free GPU memory before allocating the head
        del model; torch.cuda.empty_cache()

        head      = StyleProjectionHead(input_dim, PROJ_DIM).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=model_cfg["lr"])
        scheduler = WarmupCosineLR(optimizer, total_epochs=epochs,
                                   warmup_epochs=warmup_epochs, min_lr=1e-6)

        best_score, best_state, best_metrics = -1.0, None, {}

        for epoch in range(epochs):
            # Re-mine every epoch so negatives stay challenging as the head improves
            triplets = mine_triplets(hf_train, desc_cache, head, device, pairs_per_epoch)
            loader   = DataLoader(TripletDataset(triplets, desc_cache),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

            head.train()
            total_loss = 0.0
            bar = tqdm(loader,
                       desc=f"GPU{gpu_id} {model_cfg['name'][:20]} ep{epoch+1}/{epochs}",
                       position=gpu_id, leave=False, unit="batch", file=sys.stdout)

            for a_d, p_d, n_d in bar:
                loss = triplet_loss(head(a_d.to(device)),
                                    head(p_d.to(device)),
                                    head(n_d.to(device)),
                                    margin=MARGIN)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()
                bar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            avg_loss = total_loss / len(loader)

            metrics = evaluate(head, desc_cache, hf_train, device)
            metrics.update(probe_evaluate(head, desc_cache, hf_train, hf_probe, device))

            # Select checkpoint by the chosen criterion; fall back to r@1 if unavailable
            score = metrics.get(ckpt_criterion)
            if score is None:
                if ckpt_criterion != "r@1":
                    log.write(f"checkpoint_criterion '{ckpt_criterion}' not available "
                              f"this epoch — using r@1")
                score = metrics.get("r@1", 0.0)
            is_best = score > best_score

            probe_parts = []
            if "probe_known_r@1" in metrics:
                probe_parts.append(f"probe_known_R@1={metrics['probe_known_r@1']:.3f}")
            if "probe_new_r@1" in metrics:
                probe_parts.append(f"probe_new_R@1={metrics['probe_new_r@1']:.3f}")
            probe_str = ("  " + "  ".join(probe_parts)) if probe_parts else ""
            log.write(
                f"ep {epoch+1}/{epochs}  loss={avg_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"acc={metrics.get('triplet_acc', 0):.3f}  "
                f"R@1={metrics.get('r@1', 0):.3f}  "
                f"R@5={metrics.get('r@5', 0):.3f}  "
                f"R@10={metrics.get('r@10', 0):.3f}"
                f"{probe_str}  "
                f"gap={metrics.get('sim_gap', 0):.3f}  "
                f"pos={metrics.get('mean_pos_sim', 0):.3f}  "
                f"neg={metrics.get('mean_neg_sim', 0):.3f}"
                + (f"  [best:{ckpt_criterion}={score:.3f}]" if is_best else "")
            )

            if is_best:
                best_score   = score
                best_metrics = metrics
                best_state   = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        elapsed = time.time() - t0
        torch.save({
            # Metadata needed to reconstruct the head at inference time
            "name":         model_cfg["name"],     "backend":    model_cfg["backend"],
            "model_name":   model_cfg["model_name"],
            "pretrained":   model_cfg.get("pretrained"),
            "image_size":   image_size,            "input_dim":  input_dim,
            "proj_dim":     PROJ_DIM,
            "pooling":      POOLING_METHOD,
            # Use best epoch weights; fall back to final epoch if no improvement seen
            "state_dict":   best_state or head.state_dict(),
            "best_metrics": best_metrics,
        }, output_path)

        crit_val = best_metrics.get(ckpt_criterion, best_metrics.get("r@1", 0))
        log.write(
            f"Saved {output_path}  ({elapsed/60:.1f} min)  "
            f"criterion={ckpt_criterion}={crit_val:.3f}  "
            f"R@1={best_metrics.get('r@1', 0):.3f}  "
            f"R@5={best_metrics.get('r@5', 0):.3f}  "
            f"R@10={best_metrics.get('r@10', 0):.3f}  "
            f"acc={best_metrics.get('triplet_acc', 0):.3f}"
        )

        if clear_cache:
            cache_file = cache_dir / f"{job_id(model_cfg)}.pt"
            if cache_file.exists():
                cache_file.unlink()
                log.write(f"Deleted cache: {cache_file}")
            clear_hf_cache(model_cfg, log)

        log.close()
        return job_name, elapsed, None

    except Exception as e:
        log.write(f"ERROR: {e}\n{traceback.format_exc()}")
        log.close()
        return job_name, 0.0, str(e)


# --- MAIN --- #

if __name__ == "__main__":
    # spawn is required for CUDA in subprocesses; fork would copy the CUDA context
    mp.set_start_method("spawn", force=True)

    output_dir = Path(OUTPUT_DIR)
    cache_dir  = Path(CACHE_DIR)
    log_dir    = Path(LOG_DIR)
    for d in (output_dir, cache_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_gpus   = torch.cuda.device_count()
    gpu_pool = list(range(n_gpus)) if n_gpus else [0]  # index 0 = CPU fallback
    if not n_gpus:
        print("No CUDA GPUs found — running on CPU")

    if RUN_MODELS:
        names         = set(RUN_MODELS)
        models_to_run = [m for m in MODELS if m["name"] in names]
        if not models_to_run:
            raise ValueError(f"No models matched. Available: {[m['name'] for m in MODELS]}")
    else:
        models_to_run = MODELS

    # Jobs whose checkpoint already exists are skipped; allows safe re-run
    pending_jobs = [m for m in models_to_run
                    if not (output_dir / f"{job_id(m)}.pt").exists()]

    print(f"\n{'='*65}")
    print(f"  GPUs      : {n_gpus}  {gpu_pool}")
    print(f"  Pooling   : {POOLING_METHOD}  (tag={POOLING_TAG})")
    print(f"  Jobs      : {len(models_to_run)} total  ({len(pending_jobs)} pending)")
    for m in models_to_run:
        print(f"    {m['name']}")
    print(f"  HF dataset: {HF_REPO_ID}  (train='{TRAIN_SPLIT_NAME}', probe='{PROBE_SPLIT_NAME}')")

    # Loads the dataset once in the main process just to print this.
    # This try/except is just for a nicer error message
    try:
        _preview_train, _preview_probe = load_hf_splits()
        train_artist_set = set(_preview_train[ARTIST_COLUMN])
        probe_artist_set = set(_preview_probe[ARTIST_COLUMN])
        known_count = sum(1 for a in probe_artist_set if a in train_artist_set)
        print(f"  Train     : {len(_preview_train)} images, {len(train_artist_set)} artists")
        print(f"  Probe     : {len(_preview_probe)} images, {len(probe_artist_set)} artists "
              f"({known_count} known, {len(probe_artist_set) - known_count} new)")
    except Exception as e:
        print(f"  Could not preview dataset: {e}")

    print(f"  Criterion : {CHECKPOINT_CRITERION}")
    print(f"  Output    : {output_dir}  Cache: {cache_dir}  Logs: {log_dir}")
    print(f"{'='*65}\n")

    if not pending_jobs:
        print("All jobs already complete.")
        sys.exit(0)

    manager = mp.Manager()
    t_start = time.time()
    failed, elapsed = [], {}

    try:
        # One semaphore per GPU ensures at most one job runs per GPU at a time
        gpu_sems    = {gpu_id: manager.Semaphore(1) for gpu_id in gpu_pool}
        # Round-robin GPU assignment; semaphore serialises jobs on the same GPU
        worker_args = [
            (m, output_dir, cache_dir, log_dir,
             gpu_pool[i % len(gpu_pool)], CLEAR_CACHE,
             CHECKPOINT_CRITERION,
             gpu_sems[gpu_pool[i % len(gpu_pool)]])
            for i, m in enumerate(pending_jobs)
        ]

        with mp.Pool(processes=len(pending_jobs)) as pool:
            bar = tqdm(pool.imap_unordered(worker, worker_args),
                       total=len(worker_args), desc="Jobs", unit="job",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}  [{elapsed}<{remaining}]")
            try:
                for job_name, job_elapsed, error in bar:
                    if error:
                        failed.append(job_name)
                        bar.write(f"  ✗ FAILED  {job_name}: {error}")
                    else:
                        elapsed[job_name] = job_elapsed
                        if job_elapsed > 0:
                            bar.write(f"  ✓ done    {job_name}  ({job_elapsed/60:.1f} min)")
            except KeyboardInterrupt:
                print("\n  Interrupted — terminating workers...")
                pool.terminate()
                pool.join()

    finally:
        manager.shutdown()

    wall = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  Completed : {len(elapsed)}/{len(pending_jobs)}")
    print(f"  Wall time : {wall/3600:.1f}h  ({len(gpu_pool)} GPU(s))")
    if elapsed:
        print(f"  GPU-hours : {sum(elapsed.values())/3600:.1f}h")
        slowest = max(elapsed, key=elapsed.get)
        print(f"  Slowest   : {slowest}  ({elapsed[slowest]/60:.1f} min)")
    if failed:
        print(f"  Failed ({len(failed)}): {', '.join(failed)}")
        print("  Re-run to retry — completed jobs are skipped automatically.")
    print(f"  Logs -> {log_dir}/")
    print(f"{'='*65}")
