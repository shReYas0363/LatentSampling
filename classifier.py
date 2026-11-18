#!/usr/bin/env python3

import os
import sys
import random
import logging
import argparse
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
    logging.warning("tensorboard is not installed. TensorBoard logging disabled.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to: {seed}")


# ============================================================
# Augmentation
# ============================================================

def _shift(img: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate by zero-padding and cropping. No interpolation, so the
    sprite stays exactly binary. Content shifted off-canvas is lost,
    unlike torch.roll which would wrap a sprite around the border."""
    _, H, W = img.shape
    out = torch.zeros_like(img)
    xs_src = slice(max(0, -dx), min(W, W - dx))
    xs_dst = slice(max(0, dx), min(W, W + dx))
    ys_src = slice(max(0, -dy), min(H, H - dy))
    ys_dst = slice(max(0, dy), min(H, H + dy))
    out[:, ys_dst, xs_dst] = img[:, ys_src, xs_src]
    return out


def _rescale(img: torch.Tensor, factor: float) -> torch.Tensor:
    """Nearest-neighbour resize, centre crop/pad to 64x64, re-binarise."""
    _, H, W = img.shape
    new = max(8, int(round(H * factor)))
    r = F.interpolate(img.unsqueeze(0), size=(new, new), mode="nearest").squeeze(0)
    out = torch.zeros_like(img)
    if new >= H:
        o = (new - H) // 2
        out = r[:, o:o + H, o:o + W]
    else:
        o = (H - new) // 2
        out[:, o:o + new, o:o + new] = r
    return (out > 0.5).float()


def augment(img: torch.Tensor,
            translate: bool = False, scale: bool = False,
            blur: bool = True, noise: bool = True,
            max_shift: int = 12, scale_range: tuple = (0.7, 1.3),
            blur_sigma_max: float = 0.8, noise_std: float = 0.05) -> torch.Tensor:
    """Geometry first (stays binary); blur/noise last (deliberately not
    binary — a domain-shift correction, since at sampling time the
    classifier sees soft decoder outputs, not crisp dataset sprites)."""
    if translate:
        img = _shift(img, int(np.random.randint(-max_shift, max_shift + 1)),
                          int(np.random.randint(-max_shift, max_shift + 1)))
    if scale:
        img = _rescale(img, float(np.random.uniform(*scale_range)))

    if blur and np.random.rand() < 0.5:
        sigma = float(np.random.uniform(0.1, blur_sigma_max))
        k = 5
        ax = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
        img = F.conv2d(img.unsqueeze(0), g.view(1, 1, 1, k), padding=(0, k // 2))
        img = F.conv2d(img, g.view(1, 1, k, 1), padding=(k // 2, 0)).squeeze(0)
    if noise:
        img = img + torch.randn_like(img) * noise_std

    return img.clamp(0.0, 1.0)


# ============================================================
# Networks
# ============================================================

class ShapeNet(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


class ScaleNet(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


class PositionNet(nn.Module):
    """Flatten retained deliberately: this network needs spatial
    information, so pooling would destroy its signal. For the same
    reason it receives no translation augmentation."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# Dataset
# ============================================================

NUM_CLASSES = {
    'shape': 3,
    'scale': 3,
    'position': 2,
    'position_y': 2,
}


def _bin3(v: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """0 below lo, 1 in [lo, hi), 2 at or above hi."""
    out = np.ones(len(v), dtype=np.int64)
    out[v < lo] = 0
    out[v >= hi] = 2
    return out


class DSpritesSeparateFactorDataset(Dataset):
    """dSprites with per-factor labels and per-classifier augmentation.

    Args:
        path: path to the dSprites .npz.
        augment_for: which classifier this loader feeds; controls which
            geometric augmentations are applied. None = blur/noise only.
        train: False disables all augmentation, for clean validation.
    """

    def __init__(self, path: str,
                 augment_for: Optional[str] = None,
                 subsample_ratio: float = 1.0,
                 train: bool = True):
        super().__init__()
        self.augment_for = augment_for
        self.train = train

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"dSprites not found at {path!r}. Refusing to fall back to "
                "random data, which would silently produce meaningless results.")

        logging.info(f"Loading dataset from: {path}")
        data = np.load(path, allow_pickle=True, encoding='bytes')
        imgs = data['imgs']
        lat = data['latents_classes']
        shape, scale, px, py = lat[:, 1], lat[:, 2], lat[:, 4], lat[:, 5]

        logging.info(f"  Full dataset: {len(imgs)} samples")

        if subsample_ratio < 1.0:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(imgs), int(len(imgs) * subsample_ratio),
                             replace=False)
            imgs, shape, scale, px, py = (imgs[idx], shape[idx], scale[idx],
                                          px[idx], py[idx])
            logging.info(f"  Subsampled to {len(imgs)} samples "
                         f"(ratio={subsample_ratio})")

        self.images = imgs
        self.shape_labels = shape.astype(np.int64)
        self.scale_labels = _bin3(scale, 2, 4)                  # 3 classes
        self.position_labels = (px >= 16).astype(np.int64)      # 2 classes
        self.position_y_labels = (py >= 16).astype(np.int64)    # 2 classes

        for name, lab in (('shape', self.shape_labels),
                          ('scale', self.scale_labels),
                          ('position', self.position_labels),
                          ('position_y', self.position_y_labels)):
            counts = np.bincount(lab, minlength=NUM_CLASSES[name])
            logging.info(f"  {name} distribution: {counts.tolist()}")

        # Each classifier is made invariant to every factor except its own.
        self._policy = {
            'shape':      dict(translate=True,  scale=True),
            'scale':      dict(translate=True,  scale=False),
            'position':   dict(translate=False, scale=True),
            'position_y': dict(translate=False, scale=True),
        }

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = torch.from_numpy(self.images[idx].astype(np.float32)).unsqueeze(0)
        if self.train:
            policy = self._policy.get(self.augment_for,
                                      dict(translate=False, scale=False))
            img = augment(img, **policy)
        return {
            "image": img,
            "shape": torch.tensor(self.shape_labels[idx]),
            "scale": torch.tensor(self.scale_labels[idx]),
            "position": torch.tensor(self.position_labels[idx]),
            "position_y": torch.tensor(self.position_y_labels[idx]),
        }


# ============================================================
# Training
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="dSprites attribute classifiers")
    p.add_argument('--model-type', type=str, default='all',
                   choices=['shape', 'position', 'position_y', 'scale', 'all'])
    p.add_argument('--data-path', type=str,
                   default='dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    p.add_argument('--subsample-ratio', type=float, default=1.0)
    p.add_argument('--val-split', type=float, default=0.1,
                   help='Fraction for validation (default 0.1)')
    p.add_argument('--test-split', type=float, default=0.1,
                   help='Fraction for test (default 0.1). Train = 1 - val - test.')
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint-dir', type=str, default='.')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--log-interval', type=int, default=50)
    p.add_argument('--no-tensorboard', action='store_true')
    p.add_argument('--tb-log-dir', type=str, default='runs/classifier')
    return p.parse_args()


def train_one_epoch(model, device, dataloader, optimizer, criterion,
                    target_key, epoch, log_interval, writer, tag,
                    global_step) -> Tuple[float, float, int]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        targets = batch[target_key].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if writer is not None:
            writer.add_scalar(f'{tag}/Step/Loss', loss.item(), global_step)
        global_step += 1

        if batch_idx % log_interval == 0:
            logging.info(f"[{tag} Train] Epoch {epoch} "
                         f"[{batch_idx}/{len(dataloader)}] "
                         f"Loss: {loss.item():.4f} Acc: {100. * correct / total:.2f}%")

    return total_loss / len(dataloader), 100. * correct / total, global_step


def evaluate(model, device, dataloader, criterion, target_key) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch[target_key].to(device)
            outputs = model(images)
            total_loss += criterion(outputs, targets).item() * images.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    return total_loss / total, 100. * correct / total


def run_training(tag, model, target_key, args, device,
                 train_loader, val_loader, writer) -> None:
    logging.info(f"\n--- Training {tag} ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    ckpt = os.path.join(args.checkpoint_dir, f"{tag}.pth")
    best_metric, start_epoch, global_step = 0.0, 1, 0

    if args.resume and os.path.exists(ckpt):
        logging.info(f"Resuming from {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        best_metric = state.get('best_metric', 0.0)
        global_step = state.get('global_step', 0)
        start_epoch = state.get('epoch', 0) + 1
        logging.info(f"Resumed at epoch {start_epoch} (best {best_metric:.2f}%)")
    elif args.resume:
        logging.warning(f"No checkpoint at {ckpt}; training from scratch.")

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_acc, global_step = train_one_epoch(
            model, device, train_loader, optimizer, criterion, target_key,
            epoch, args.log_interval, writer, tag, global_step)
        va_loss, va_acc = evaluate(model, device, val_loader, criterion, target_key)

        logging.info(f"[{tag} Epoch {epoch}] train {tr_acc:.2f}% "
                     f"(loss {tr_loss:.4f}) | val {va_acc:.2f}% (loss {va_loss:.4f})")

        if writer is not None:
            writer.add_scalar(f'{tag}/Epoch/Train_Accuracy', tr_acc, epoch)
            writer.add_scalar(f'{tag}/Epoch/Val_Accuracy', va_acc, epoch)

        if va_acc > best_metric:
            best_metric = va_acc
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_metric': best_metric,
                        'global_step': global_step,
                        'args': vars(args)}, ckpt)
            logging.info(f" => saved {ckpt} (best {best_metric:.2f}%)")


def build_model(name: str, n_classes: Optional[int] = None) -> nn.Module:
    if n_classes is None:
        n_classes = NUM_CLASSES[name]
    if name == 'shape':
        return ShapeNet(num_classes=n_classes)
    if name == 'scale':
        return ScaleNet(num_classes=n_classes)
    if name in ('position', 'position_y'):
        return PositionNet(num_classes=n_classes)
    raise ValueError(name)


def test_and_display(model, device, test_loader, target_key, tag, n_classes):
    """Evaluate the best checkpoint on the held-out test set and print results."""
    model.eval()
    correct = 0
    total = 0
    per_class_correct = [0] * n_classes
    per_class_total = [0] * n_classes

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            targets = batch[target_key].to(device)
            preds = model(images).argmax(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            for c in range(n_classes):
                mask = targets == c
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += preds[mask].eq(targets[mask]).sum().item()

    acc = 100.0 * correct / total if total else 0.0
    logging.info(f"")
    logging.info(f"{'=' * 55}")
    logging.info(f"  TEST RESULTS: {tag.upper()}")
    logging.info(f"{'=' * 55}")
    logging.info(f"  Overall accuracy: {acc:.2f}%  ({correct}/{total})")
    logging.info(f"  {'Class':<10} {'Correct':>10} {'Total':>10} {'Accuracy':>10}")
    logging.info(f"  {'-' * 42}")
    for c in range(n_classes):
        c_acc = 100.0 * per_class_correct[c] / per_class_total[c] if per_class_total[c] else 0.0
        logging.info(f"  {c:<10} {per_class_correct[c]:>10} {per_class_total[c]:>10} {c_acc:>9.2f}%")
    logging.info(f"{'=' * 55}\n")
    return acc


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(f"Split: train={1 - args.val_split - args.test_split:.0%} / "
                 f"val={args.val_split:.0%} / test={args.test_split:.0%}")

    writer = None
    if not args.no_tensorboard and SummaryWriter is not None:
        os.makedirs(args.tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        writer.add_text("Hyperparameters", str(vars(args)))

    targets = (['shape', 'position', 'position_y', 'scale']
               if args.model_type == 'all' else [args.model_type])

    test_results = {}

    try:
        for name in targets:
            n_classes = NUM_CLASSES[name]
            tag = name

       
            train_ds = DSpritesSeparateFactorDataset(
                path=args.data_path, augment_for=name,
                subsample_ratio=args.subsample_ratio, train=True)
            eval_ds = DSpritesSeparateFactorDataset(
                path=args.data_path, augment_for=name,
                subsample_ratio=args.subsample_ratio, train=False)

            # 80 / 10 / 10 split
            n_total = len(train_ds)
            n_val = int(n_total * args.val_split)
            n_test = int(n_total * args.test_split)
            n_train = n_total - n_val - n_test

            g = torch.Generator().manual_seed(args.seed)
            perm = torch.randperm(n_total, generator=g).tolist()
            train_idx = perm[:n_train]
            val_idx = perm[n_train:n_train + n_val]
            test_idx = perm[n_train + n_val:]

            logging.info(f"  {name}: train={n_train}, val={n_val}, test={n_test}")

            train_loader = DataLoader(
                torch.utils.data.Subset(train_ds, train_idx),
                batch_size=args.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(
                torch.utils.data.Subset(eval_ds, val_idx),
                batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(
                torch.utils.data.Subset(eval_ds, test_idx),
                batch_size=args.batch_size, shuffle=False)

            model = build_model(name, n_classes).to(device)
            run_training(tag, model, name, args, device,
                         train_loader, val_loader, writer)

            # ── Reload best checkpoint and evaluate on test set ──────
            ckpt_path = os.path.join(args.checkpoint_dir, f"{tag}.pth")
            if os.path.exists(ckpt_path):
                best_state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(best_state['model_state_dict'])
                logging.info(f"Loaded best checkpoint for {tag} "
                             f"(val {best_state.get('best_metric', '?'):.2f}%)")
            else:
                logging.warning(f"No checkpoint found at {ckpt_path}; "
                                f"testing with last-epoch weights.")

            acc = test_and_display(model, device, test_loader, name, tag, n_classes)
            test_results[name] = acc

        # ── Final summary ────────────────────────────────────────────
        logging.info("")
        logging.info("=" * 55)
        logging.info("            FINAL TEST SUMMARY (all classifiers)")
        logging.info("=" * 55)
        for name, acc in test_results.items():
            logging.info(f"  {name:<16s} : {acc:.2f}%")
        logging.info("=" * 55)
        logging.info("All requested classifier training and testing completed.\n")

    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()



