""" SugarCBAN — Asym-CBAM (1×5 / 5×1)  3-Seed """

import os, sys, time, json, random, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, models
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    matthews_corrcoef, classification_report,
    confusion_matrix, roc_curve, auc,
    precision_score, recall_score,
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL_CLASSES = ["Healthy", "Mosaic", "Red Rot", "Rust", "Yellow"]
NUM_CLASSES       = 5
IMAGE_SIZE        = 224
MEAN              = [0.485, 0.456, 0.406]
STD               = [0.229, 0.224, 0.225]
IMG_EXTENSIONS    = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
                     "*.JPG", "*.JPEG", "*.PNG")
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10

KERNEL_SIZE = 5          # ← best kernel from ablation
SEEDS       = [42, 0, 1]

CLASS_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
def _normalise(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


def detect_class_folders(data_root: str) -> Dict[str, str]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {root}")
    actual_dirs    = {d.name: d for d in root.iterdir() if d.is_dir()}
    norm_to_actual = {_normalise(k): k for k in actual_dirs}
    mapping, missing = {}, []
    for canon in CANONICAL_CLASSES:
        key = _normalise(canon)
        if key in norm_to_actual:
            mapping[canon] = norm_to_actual[key]
        else:
            missing.append(canon)
    if missing:
        raise ValueError(f"Could not match: {missing}. Found: {sorted(actual_dirs.keys())}")
    print("  📂  Class → Folder mapping:")
    for canon, actual in mapping.items():
        n = sum(len(list(Path(data_root, actual).glob(ext))) for ext in IMG_EXTENSIONS)
        print(f"       {canon:<10} → '{actual}'  ({n} images)")
    return mapping


def get_transforms(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0),
                ratio=(0.75, 1.333),
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=0, shear=(-10, 10, -10, 10)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.20),
                                     ratio=(0.3, 3.3), value="random"),
        ])
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


class SugarcaneDataset(Dataset):
    def __init__(self, data_root, class_map, split="train",
                 indices: Optional[List[int]] = None):
        self.transform = get_transforms(split)
        self.samples: List[Tuple[Path, int]] = []
        root = Path(data_root)
        for idx, canon in enumerate(CANONICAL_CLASSES):
            cls_dir = root / class_map[canon]
            for ext in IMG_EXTENSIONS:
                for p in sorted(cls_dir.glob(ext)):
                    self.samples.append((p, idx))
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


def make_dataloaders(data_root, class_map, batch_size=32, num_workers=2, seed=42):
    full   = SugarcaneDataset(data_root, class_map, split="train")
    labels = [s[1] for s in full.samples]
    idx    = list(range(len(full)))
    tr_val_idx, te_idx, tr_val_lbl, _ = train_test_split(
        idx, labels, test_size=TEST_RATIO, stratify=labels, random_state=seed)
    val_frac = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    tr_idx, val_idx = train_test_split(
        tr_val_idx, test_size=val_frac, stratify=tr_val_lbl, random_state=seed)
    train_ds = SugarcaneDataset(data_root, class_map, "train", tr_idx)
    val_ds   = SugarcaneDataset(data_root, class_map, "val",   val_idx)
    test_ds  = SugarcaneDataset(data_root, class_map, "test",  te_idx)
    print(f"  Split  Train:{len(train_ds)}  Val:{len(val_ds)}  Test:{len(test_ds)}")
    pin = num_workers > 0
    kw  = dict(batch_size=batch_size, num_workers=num_workers,
               pin_memory=pin, persistent_workers=pin,
               prefetch_factor=2 if pin else None,
               multiprocessing_context="fork" if pin else None)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(val_ds,   shuffle=False, **kw),
            DataLoader(test_ds,  shuffle=False, **kw),
            test_ds)


# ══════════════════════════════════════════════════════════════════════════════
# Model — Asym-CBAM (1×5 / 5×1)
# ══════════════════════════════════════════════════════════════════════════════

class StandardChannelAttention(nn.Module):
    def __init__(self, channels: int = 768, reduction: int = 16):
        super().__init__()
        r = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(r, channels, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x).flatten(1))
        mx  = self.mlp(self.max_pool(x).flatten(1))
        return x * self.sig(avg + mx).unsqueeze(-1).unsqueeze(-1)


class AsymmetricSpatialAttention(nn.Module):
    """
    Asym-CBAM spatial gate: 1×k + k×1 asymmetric kernel decomposition.
    This experiment uses k=5.
    Biological motivation: sugarcane leaf veins run vertically;
    1×5 captures horizontal (cross-vein) disease spread,
    5×1 captures vertical (along-vein) disease spread.
    """
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        pad_h = (0, kernel_size // 2)
        pad_v = (kernel_size // 2, 0)
        self.conv_h = nn.Conv2d(2, 1, (1, kernel_size), padding=pad_h, bias=False)
        self.conv_v = nn.Conv2d(2, 1, (kernel_size, 1), padding=pad_v, bias=False)
        self.bn     = nn.BatchNorm2d(1)
        self.sig    = nn.Sigmoid()

    def forward(self, x):
        avg  = x.mean(dim=1, keepdim=True)
        mx   = x.max(dim=1, keepdim=True).values
        desc = torch.cat([avg, mx], dim=1)
        gate = self.sig(self.bn(self.conv_h(desc) + self.conv_v(desc)))
        return x * gate


class AsymCBAM_1x5(nn.Module):
    """Proposed Asym-CBAM with 1×5 / 5×1 asymmetric spatial kernels."""
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        backbone      = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.features     = backbone.features
        self.channel_attn = StandardChannelAttention(channels=768)
        self.spatial_attn = AsymmetricSpatialAttention(kernel_size=KERNEL_SIZE)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.channel_attn(feat)
        feat = self.spatial_attn(feat)
        return self.head(feat)

    def get_gradcam_target_layer(self):
        """Depthwise convolution in the final ConvNeXt block of Stage 4."""
        return self.features[7][2].block[0]


# ══════════════════════════════════════════════════════════════════════════════
# GradCAM
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Hooks into the target layer, runs a forward + backward pass,
    and produces a spatial heatmap showing which regions drove the prediction.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self.activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, input_tensor: torch.Tensor,
                 class_idx: Optional[int] = None) -> np.ndarray:
        """
        Returns a (H, W) heatmap in [0, 1] for the given class.
        If class_idx is None, uses the predicted class.
        """
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Global average pool the gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                                mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam, int(class_idx)


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised tensor to a displayable (H, W, 3) uint8 array."""
    mean = np.array(MEAN)
    std  = np.array(STD)
    img  = tensor.cpu().permute(1, 2, 0).numpy()
    img  = img * std + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def overlay_heatmap(img_np: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """Overlay a GradCAM heatmap on the original image."""
    heatmap = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    overlay = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    return overlay


def save_gradcam_grid(model: nn.Module, test_ds: SugarcaneDataset,
                      device: torch.device, save_dir: Path,
                      n_samples: int = 10, seed: int = 42):
    """
    Saves a grid of n_samples GradCAM visualisations from the test set.
    Each row: [original image | GradCAM overlay | confidence bar]
    """
    random.seed(seed)
    indices = random.sample(range(len(test_ds)), min(n_samples, len(test_ds)))

    target_layer = model.get_gradcam_target_layer()
    gradcam      = GradCAM(model, target_layer)

    val_tf = get_transforms("val")

    n_cols = 3
    n_rows = len(indices)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        path, true_label = test_ds.samples[idx]
        img_pil   = Image.open(path).convert("RGB")
        img_tensor = val_tf(img_pil).to(device)
        img_np     = denormalise(img_tensor)

        cam, pred_label = gradcam.generate(img_tensor)
        overlay         = overlay_heatmap(img_np, cam)

        # Col 0 — original
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"True: {CANONICAL_CLASSES[true_label]}",
                               fontsize=8)
        axes[row, 0].axis("off")

        # Col 1 — GradCAM overlay
        axes[row, 1].imshow(overlay)
        correct = "✓" if pred_label == true_label else "✗"
        axes[row, 1].set_title(
            f"Pred: {CANONICAL_CLASSES[pred_label]} {correct}", fontsize=8)
        axes[row, 1].axis("off")

        # Col 2 — softmax confidence bar
        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0))
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        ax_bar = axes[row, 2]
        bars   = ax_bar.barh(CANONICAL_CLASSES, probs,
                             color=CLASS_COLORS, alpha=0.85)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel("Confidence", fontsize=7)
        ax_bar.tick_params(labelsize=7)
        ax_bar.axvline(0.5, color="grey", linewidth=0.8, linestyle="--")
        for bar, p in zip(bars, probs):
            ax_bar.text(min(p + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                        f"{p:.2f}", va="center", fontsize=7)

    plt.suptitle(f"GradCAM — Asym-CBAM (1×5 / 5×1)  |  Seed {seed}",
                 fontsize=11, y=1.002)
    plt.tight_layout()
    path_out = save_dir / "gradcam_grid.png"
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path_out}")


def save_gradcam_per_class(model: nn.Module, test_ds: SugarcaneDataset,
                           device: torch.device, save_dir: Path, seed: int = 42):
    """
    One correctly-classified GradCAM sample per disease class.
    Useful for the paper's qualitative results section.
    """
    random.seed(seed)
    target_layer = model.get_gradcam_target_layer()
    gradcam      = GradCAM(model, target_layer)
    val_tf       = get_transforms("val")

    # Collect one correct sample per class
    class_samples: Dict[int, Tuple] = {}
    shuffled = list(range(len(test_ds)))
    random.shuffle(shuffled)

    model.eval()
    for idx in shuffled:
        if len(class_samples) == NUM_CLASSES:
            break
        path, true_label = test_ds.samples[idx]
        if true_label in class_samples:
            continue
        img_tensor = val_tf(Image.open(path).convert("RGB")).to(device)
        with torch.no_grad():
            pred = model(img_tensor.unsqueeze(0)).argmax(dim=1).item()
        if pred == true_label:
            class_samples[true_label] = (path, img_tensor)

    if len(class_samples) < NUM_CLASSES:
        print(f"  ⚠ Only found {len(class_samples)}/{NUM_CLASSES} "
              f"correctly classified samples for GradCAM per-class plot.")

    fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(NUM_CLASSES * 3.5, 7))
    for col, cls_idx in enumerate(sorted(class_samples.keys())):
        path, img_tensor = class_samples[cls_idx]
        img_np  = denormalise(img_tensor)
        cam, _  = gradcam.generate(img_tensor, class_idx=cls_idx)
        overlay = overlay_heatmap(img_np, cam)

        axes[0, col].imshow(img_np)
        axes[0, col].set_title(CANONICAL_CLASSES[cls_idx], fontsize=10,
                               fontweight="bold")
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay)
        axes[1, col].set_title("GradCAM", fontsize=9)
        axes[1, col].axis("off")

    plt.suptitle("GradCAM per disease class — Asym-CBAM (1×5 / 5×1)",
                 fontsize=12)
    plt.tight_layout()
    path_out = save_dir / "gradcam_per_class.png"
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Eval helpers
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            preds  = logits.argmax(1)
            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels))


def evaluate_with_probs(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


def full_metrics(y_true, y_pred) -> Dict:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1_macro":  float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred,    average="macro", zero_division=0)),
        "kappa":     float(cohen_kappa_score(y_true, y_pred)),
        "mcc":       float(matthews_corrcoef(y_true, y_pred)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# All publication plots
# ─────────────────────────────────────────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, save_dir: Path, title: str):
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_pct = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Blues",
                xticklabels=CANONICAL_CLASSES,
                yticklabels=CANONICAL_CLASSES, ax=axes[0])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].set_title(f"{title} — Counts")
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=CANONICAL_CLASSES,
                yticklabels=CANONICAL_CLASSES,
                ax=axes[1], vmin=0, vmax=100)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].set_title(f"{title} — Row %")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved: {save_dir / 'confusion_matrix.png'}")


def save_roc_curves(y_true, y_probs, save_dir: Path, title: str):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (cls, col) in enumerate(zip(CANONICAL_CLASSES, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, color=col, linewidth=2,
                label=f"{cls}  (AUC={auc(fpr, tpr):.3f})")
    fpr_m, tpr_m, _ = roc_curve(y_bin.ravel(), y_probs.ravel())
    ax.plot(fpr_m, tpr_m, "k--", linewidth=1.5,
            label=f"Micro-avg  (AUC={auc(fpr_m, tpr_m):.3f})")
    ax.plot([0, 1], [0, 1], "grey", linestyle=":", linewidth=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC — {title}")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / "roc_curves.png", dpi=150)
    plt.close()
    print(f"  Saved: {save_dir / 'roc_curves.png'}")


def save_training_curves(history: Dict, save_dir: Path, seed: int):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=1.8)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   linewidth=1.8)
    axes[0].set_title(f"Loss — Seed {seed}")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], label="Train",  linewidth=1.8)
    axes[1].plot(epochs, history["val_acc"],   label="Val",    linewidth=1.8)
    axes[1].plot(epochs, history["val_f1"],    label="Val F1",
                 linewidth=1.8, linestyle="--")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"Accuracy — Seed {seed}")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"  Saved: {save_dir / 'training_curves.png'}")


def save_per_class_metrics(y_true, y_pred, save_dir: Path, title: str):
    report = classification_report(y_true, y_pred,
             target_names=CANONICAL_CLASSES, zero_division=0, output_dict=True)
    prec = [report[c]["precision"] for c in CANONICAL_CLASSES]
    rec  = [report[c]["recall"]    for c in CANONICAL_CLASSES]
    f1s  = [report[c]["f1-score"]  for c in CANONICAL_CLASSES]
    x = np.arange(NUM_CLASSES); w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w, prec, w, label="Precision", color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x,     rec,  w, label="Recall",    color="#55A868", alpha=0.85)
    b3 = ax.bar(x + w, f1s,  w, label="F1",        color="#C44E52", alpha=0.85)
    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(CANONICAL_CLASSES, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title(f"Per-Class Metrics — {title}")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "per_class_metrics.png", dpi=150)
    plt.close()
    print(f"  Saved: {save_dir / 'per_class_metrics.png'}")


def save_confidence_histogram(y_true, y_probs, y_pred,
                              save_dir: Path, title: str):
    max_conf = y_probs.max(axis=1)
    correct  = y_true == y_pred
    fig, ax  = plt.subplots(figsize=(8, 5))
    ax.hist(max_conf[correct],  bins=25, alpha=0.6,
            color="#55A868", label=f"Correct  (n={correct.sum()})")
    ax.hist(max_conf[~correct], bins=25, alpha=0.6,
            color="#C44E52", label=f"Wrong  (n={(~correct).sum()})")
    ax.set_xlabel("Max Softmax Confidence"); ax.set_ylabel("Count")
    ax.set_title(f"Confidence Distribution — {title}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "confidence_histogram.png", dpi=150)
    plt.close()
    print(f"  Saved: {save_dir / 'confidence_histogram.png'}")


def save_kappa_mcc(y_true, y_pred, save_dir: Path, title: str):
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Cohen's Kappa", "MCC"], [kappa, mcc],
                  color=["#4C72B0", "#55A868"], alpha=0.85)
    for bar, val in zip(bars, [kappa, mcc]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005, f"{val:.4f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title(f"Kappa & MCC — {title}"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "kappa_mcc.png", dpi=150)
    plt.close()
    print(f"  Saved: {save_dir / 'kappa_mcc.png'}")


def save_precision_recall_curve(y_true, y_probs, save_dir: Path, title: str):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (cls, col) in enumerate(zip(CANONICAL_CLASSES, CLASS_COLORS)):
        prec_c, rec_c, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        ax.plot(rec_c, prec_c, color=col, linewidth=2,
                label=f"{cls}  (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curves — {title}")
    ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / "precision_recall_curves.png", dpi=150)
    plt.close()
    print(f"  Saved: {save_dir / 'precision_recall_curves.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed summary
# ─────────────────────────────────────────────────────────────────────────────
def build_summary(all_metrics: Dict[int, dict], save_dir: Path):
    metrics_keys = ["accuracy", "f1_macro", "kappa", "mcc"]
    print(f"\n{'═'*62}")
    print(f"  Asym-CBAM (1×5) — Multi-Seed Summary  seeds={SEEDS}")
    print(f"{'═'*62}")
    print(f"  {'Metric':<14}", end="")
    for s in SEEDS:
        print(f"{'Seed'+str(s):>10}", end="")
    print(f"{'Mean':>10} {'±Std':>8}")
    print("  " + "─" * 58)

    summary = {}
    for mk in metrics_keys:
        vals = [all_metrics[s][mk] for s in SEEDS]
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        summary[mk] = {"seeds": {str(s): v for s, v in zip(SEEDS, vals)},
                       "mean": mean, "std": std}
        print(f"  {mk:<14}", end="")
        for v in vals:
            print(f"{v:>10.4f}", end="")
        print(f"{mean:>10.4f} {std:>7.4f}")
    print(f"{'═'*62}\n")

    # Save JSON
    out_json = save_dir / "multiseed_summary.json"
    with open(out_json, "w") as f:
        json.dump({"variant": "asym_cbam_1x5", "kernel_size": KERNEL_SIZE,
                   "seeds": SEEDS, "summary": summary}, f, indent=2)
    print(f"  Saved: {out_json}")

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(metrics_keys))
    means = [summary[mk]["mean"] for mk in metrics_keys]
    stds  = [summary[mk]["std"]  for mk in metrics_keys]
    bars  = ax.bar(x, means, color="#009E73", alpha=0.85,
                   width=0.45, label="Mean ± Std")
    ax.errorbar(x, means, yerr=stds, fmt="none",
                color="black", capsize=8, linewidth=2)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.002,
                f"{mean:.4f}\n±{std:.4f}",
                ha="center", va="bottom", fontsize=9)
    seed_markers = ["o", "s", "^"]
    seed_colors  = ["#4C72B0", "#C44E52", "#8172B2"]
    for si, (seed, mk_col) in enumerate(zip(SEEDS, seed_colors)):
        vals = [summary[mk]["seeds"][str(seed)] for mk in metrics_keys]
        ax.scatter(x, vals, color=mk_col, zorder=5, s=70,
                   marker=seed_markers[si], label=f"Seed {seed}")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "F1 Macro", "Kappa", "MCC"], fontsize=11)
    ax.set_ylim(min(means) - 0.015, 1.02)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Asym-CBAM (1×5 / 5×1) — 3-Seed Robustness  (ConvNeXt-Tiny)",
                 fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "multiseed_summary.png", dpi=150)
    plt.close()
    print(f"  Saved: {save_dir / 'multiseed_summary.png'}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed training
# ─────────────────────────────────────────────────────────────────────────────
def train_one_seed(seed, data_root, class_map, epochs=50, batch_size=32,
                   lr=5e-5, weight_decay=0.01, num_workers=2,
                   save_dir="outputs_asym1x5_3seed"):
    set_seed(seed)
    out = Path(save_dir) / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'═'*70}")
    print(f"  Asym-CBAM (1×{KERNEL_SIZE}) | Seed: {seed} | Device: {device}")
    print(f"  Output: {out}")
    print(f"{'═'*70}")

    result_path = out / "results.json"
    if result_path.exists():
        print(f"  [SKIP] Seed {seed} already completed.")
        with open(result_path) as f:
            r = json.load(f)
        m = r["metrics"]
        print(f"  Acc:{m['accuracy']:.4f}  F1:{m['f1_macro']:.4f}  "
              f"Kappa:{m['kappa']:.4f}  MCC:{m['mcc']:.4f}")
        return m

    train_loader, val_loader, test_loader, test_ds = make_dataloaders(
        data_root, class_map, batch_size=batch_size,
        num_workers=num_workers, seed=seed)

    model     = AsymCBAM_1x5().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  [], "val_f1": []}
    best_val_acc = 0.0
    best_ckpt    = out / "best_model.pth"
    epoch_ckpt   = out / "epoch_ckpt.pth"

    # Resume
    start_epoch = 1
    if epoch_ckpt.exists():
        print(f"  [RESUME] {epoch_ckpt}")
        ck = torch.load(epoch_ckpt, map_location=device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        history      = ck["history"]
        best_val_acc = ck["best_val_acc"]
        start_epoch  = ck["epoch"] + 1

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            preds       = model(imgs).argmax(1) if False else \
                          criterion.__class__.__name__ and \
                          imgs.new_zeros(1) or \
                          model(imgs).detach().argmax(1)
            # recompute preds cleanly
            with torch.no_grad():
                preds = model(imgs).argmax(1)
            tr_loss    += loss.item() * imgs.size(0)
            tr_correct += (preds == labels).sum().item()
            tr_total   += imgs.size(0)
        scheduler.step()

        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device)
        val_f1 = f1_score(val_labels, val_preds,
                          average="macro", zero_division=0)

        history["train_loss"].append(tr_loss / tr_total)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_correct / tr_total)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": val_acc}, best_ckpt)

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "history": history, "best_val_acc": best_val_acc},
                   epoch_ckpt)

        print(f"  Ep[{epoch:3d}/{epochs}] "
              f"TrainAcc:{tr_correct/tr_total:.4f} "
              f"ValAcc:{val_acc:.4f} F1:{val_f1:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    # Load best and evaluate
    print(f"\n  Loading best checkpoint (val_acc={best_val_acc:.4f})")
    model.load_state_dict(torch.load(best_ckpt, map_location=device)["model_state"])
    if epoch_ckpt.exists():
        epoch_ckpt.unlink()

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device)
    test_probs, _, _ = evaluate_with_probs(model, test_loader, device)

    metrics = full_metrics(test_labels, test_preds)
    metrics["test_loss"] = float(test_loss)

    title = f"Asym-CBAM (1×{KERNEL_SIZE}) Seed {seed}"
    print(f"\n  ── Test Results ─────────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<14} : {v:.4f}" if isinstance(v, float) else
              f"  {k:<14} : {v}")
    print()
    print(classification_report(test_labels, test_preds,
                                target_names=CANONICAL_CLASSES, zero_division=0))

    # ── Save all plots ───────────────────────────────────────────────────────
    save_confusion_matrix(test_labels, test_preds, out, title)
    save_roc_curves(test_labels, test_probs, out, title)
    save_training_curves(history, out, seed)
    save_per_class_metrics(test_labels, test_preds, out, title)
    save_confidence_histogram(test_labels, test_probs, test_preds, out, title)
    save_kappa_mcc(test_labels, test_preds, out, title)
    save_precision_recall_curve(test_labels, test_probs, out, title)

    # ── GradCAM ─────────────────────────────────────────────────────────────
    print(f"  Generating GradCAM visualisations …")
    save_gradcam_grid(model, test_ds, device, out, n_samples=10, seed=seed)
    save_gradcam_per_class(model, test_ds, device, out, seed=seed)

    # ── Save results.json ────────────────────────────────────────────────────
    with open(result_path, "w") as f:
        json.dump({
            "variant": f"asym_cbam_1x{KERNEL_SIZE}", "seed": seed,
            "metrics": metrics,
            "history": {k: [float(v) for v in vals]
                        for k, vals in history.items()},
        }, f, indent=2)
    print(f"  Saved: {result_path}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Config + Entry
# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    try:
        get_ipython()
        class Args:
            data_root    = ("/root/sugar/data/Sugarcane Leaf Disease_preprocess/"
                            "Sugarcane Leaf Disease Dataset")
            epochs       = 50
            batch_size   = 32
            lr           = 5e-5
            weight_decay = 0.01
            num_workers  = 2
            save_dir     = "outputs_asym1x5_3seed"
        return Args()
    except NameError:
        pass
    import argparse
    p = argparse.ArgumentParser(description="Asym-CBAM 1x5 — 3-seed experiment")
    p.add_argument("--data_root",    type=str, required=True)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--save_dir",     type=str,   default="outputs_asym1x5_3seed")
    return p.parse_args()


def main():
    args     = get_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Scanning dataset: {args.data_root}")
    class_map = detect_class_folders(args.data_root)

    common = dict(
        data_root=args.data_root, class_map=class_map,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        num_workers=args.num_workers, save_dir=args.save_dir,
    )

    all_metrics: Dict[int, Dict] = {}
    for seed in SEEDS:
        all_metrics[seed] = train_one_seed(seed=seed, **common)

    summary = build_summary(all_metrics, save_dir)

    print(f"\n  ✓  All 3 seeds complete.")
    print(f"  Results  → {args.save_dir}/")
    print(f"\n  Mean Accuracy : {summary['accuracy']['mean']:.4f} "
          f"± {summary['accuracy']['std']:.4f}")
    print(f"  Mean F1 Macro : {summary['f1_macro']['mean']:.4f} "
          f"± {summary['f1_macro']['std']:.4f}")
    print(f"  Mean Kappa    : {summary['kappa']['mean']:.4f} "
          f"± {summary['kappa']['std']:.4f}")
    print(f"  Mean MCC      : {summary['mcc']['mean']:.4f} "
          f"± {summary['mcc']['std']:.4f}")


if __name__ == "__main__":
    main()
