#!/usr/bin/env python3

import os
import sys
import math
import random
import logging
import argparse
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    logging.warning("matplotlib is not installed. Visual grids will not be saved locally.")

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
    logging.warning("tensorboard (or tensorflow/tensorboard) is not installed. TensorBoard logging will be disabled.")


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to: {seed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible Training Script for Beta-TCVAE on dSprites")
    
    # Dataset Parameters
    parser.add_argument('--data-path', type=str, default='dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                        help='Path to dSprites .npz archive file')
    parser.add_argument('--subsample-ratio', type=float, default=1.0,
                        help='Dataset subsampling ratio (between 0.0 and 1.0)')
    parser.add_argument('--no-exclude-zero-shot', dest='exclude_zero_shot', action='store_false',
                        help='Disable zero-shot class training sample exclusions')
    parser.set_defaults(exclude_zero_shot=True)
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Fraction of dataset reserved for validation split metrics')
    
    # Model/Training Hyperparameters
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Training batch size')
    parser.add_argument('--latent-dim', type=int, default=10,
                        help='Dimensionality of target latent vector z')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Adam optimizer learning rate')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of optimization epochs to train')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deterministic reproducibility')
    
    # Beta-TCVAE Loss Parameters
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for Index-Code MI')
    parser.add_argument('--beta', type=float, default=6.0,
                        help='Weight for Total Correlation')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Weight for Dimension-wise KL')
    parser.add_argument('--steps-to-max-capacity', type=int, default=100000,
                        help='Total number of steps for linear warm-up to reach max_capacity')
    parser.add_argument('--recon-loss-type', type=str, default='BCE+Dice',
                        choices=['BCE', 'BCE+Dice', 'BCE+IoU'],
                        help='Type of reconstruction loss (BCE, BCE+Dice, BCE+IoU)')
    
    # Logging, Checkpointing and Visualizations
    parser.add_argument('--checkpoint-path', type=str, default='best_beta_vae_dsprites_leave.pth',
                        help='Output filepath to save the best model checkpoint dictionary')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the checkpoint specified in --checkpoint-path if it exists')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Console log output frequency in terms of mini-batch iterations')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Explicitly disable logging tensorboard stats')
    parser.add_argument('--tb-log-dir', type=str, default='runs/beta_vae',
                        help='Tensorboard logging base run directory path')
    parser.add_argument('--no-visual-grid', action='store_true',
                        help='Disable saving reconstruction visual quality validation grids')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory path where comparison image grids are saved')

    return parser.parse_args()


class DSpritesDataset(Dataset):
    """Dataset wrapper for loading and filtering dSprites dataset images."""
    def __init__(self, path: str, subsample_ratio: float = 1.0, exclude: bool = True):
        super(DSpritesDataset, self).__init__()
        
        if os.path.exists(path):
            logging.info(f"Loading dataset from: {path}")
            data = np.load(path, allow_pickle=True, encoding='bytes')
            raw_images = data['imgs']
            
           
            if exclude:
                shape_classes = data['latents_classes'][:, 1] 
                scale_classes = data['latents_classes'][:, 2]  
                pos_x_classes = data['latents_classes'][:, 4]  
                pos_y_classes = data['latents_classes'][:, 5]  
                #Top right square
                zero_shot_mask = ((scale_classes==2) | (scale_classes==3)) & (shape_classes==0)
                valid_train_mask = ~zero_shot_mask
                self.images = raw_images[valid_train_mask]
                logging.info(f"Excluded {np.sum(zero_shot_mask)} zero-shot samples from training pool "
                             f"(medium scale square).")
            else:
                self.images = raw_images

            if subsample_ratio < 1.0:
                np.random.seed(42)  
                total_samples = len(self.images)
                keep_count = int(total_samples * subsample_ratio)
                indices = np.random.choice(total_samples, keep_count, replace=False)
                self.images = self.images[indices]
                logging.info(f"Subsampled dataset by ratio {subsample_ratio}. Size: {len(self.images)}.")
                
            logging.info(f"Final VAE Dataset Size: {len(self.images)} samples.")
        else:
            
            logging.warning(f"Dataset path '{path}' not found. Initializing dummy noise dataset.")
            self.images = np.random.randint(0, 2, (1000, 64, 64), dtype=np.uint8)
            
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img = torch.from_numpy(self.images[idx]).float().unsqueeze(0) 
        return img


class Encoder(nn.Module):
    def __init__(self, img_size: Tuple[int, int, int] = (1, 64, 64), latent_dim: int = 10):
        super(Encoder, self).__init__()

        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
    
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.lin1 = nn.Linear(np.prod(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, img_size: Tuple[int, int, int] = (1, 64, 64), latent_dim: int = 10):
        super(Decoder, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.prod(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))

        return self.convT3(x)


class BetaTCVAE(nn.Module):
    """
    Beta-TCVAE architecture mapping 64x64 input images to a low-dimensional latent space.
    """
    def __init__(self, latent_dim: int = 10, img_size: Tuple[int, int, int] = (1, 64, 64)):
        super(BetaTCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(img_size=img_size, latent_dim=latent_dim)
        self.decoder = Decoder(img_size=img_size, latent_dim=latent_dim)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

def iou_loss(recon_x: torch.Tensor, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the soft Intersection over Union (IoU) loss.
    Both inputs should have values in range [0, 1].
    
    Args:
        recon_x: Sigmoid reconstruction output, shape [B, C, H, W]
        x: Target binary images, shape [B, C, H, W]
    """
    # Flatten spatial and channel dimensions to compute per-sample IoU
    recon_x_flat = recon_x.view(recon_x.size(0), -1)
    x_flat = x.view(x.size(0), -1)
    
    intersection = (recon_x_flat * x_flat).sum(dim=1)
    union = recon_x_flat.sum(dim=1) + x_flat.sum(dim=1) - intersection
    
    iou = (intersection + eps) / (union + eps)
    # IoU loss is 1 - IoU, averaged over the batch
    return (1.0 - iou).mean()


def dice_loss(recon_x: torch.Tensor, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the soft Dice loss.
    Both inputs should have values in range [0, 1].
    
    Args:
        recon_x: Sigmoid reconstruction output, shape [B, C, H, W]
        x: Target binary images, shape [B, C, H, W]
    """
    # Flatten spatial and channel dimensions to compute per-sample Dice
    recon_x_flat = recon_x.view(recon_x.size(0), -1)
    x_flat = x.view(x.size(0), -1)
    
    intersection = (recon_x_flat * x_flat).sum(dim=1)
    union = recon_x_flat.sum(dim=1) + x_flat.sum(dim=1)
    
    dice = (2.0 * intersection + eps) / (union + eps)
    # Dice loss is 1 - Dice, averaged over the batch
    return (1.0 - dice).mean()


def beta_tcvae_loss(
    recon_x: torch.Tensor, 
    x: torch.Tensor, 
    z: torch.Tensor,
    mu: torch.Tensor, 
    logvar: torch.Tensor, 
    dataset_size: int,
    alpha: float = 1.0,
    beta: float = 6.0,
    gamma: float = 1.0,
    recon_loss_type: str = 'BCE+Dice'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Beta-TCVAE loss using the Minibatch Weighted Sampling estimator.
    
    Args:
        recon_x: Reconstructed images, tensor of shape [B, C, H, W]
        x: Ground truth images, tensor of shape [B, C, H, W]
        z: Sampled latent vectors, tensor of shape [B, D]
        mu: Mean latent vectors, tensor of shape [B, D]
        logvar: Log-variance latent vectors, tensor of shape [B, D]
        dataset_size: Total number of samples in the entire dataset (N)
        alpha: Weight for Index-Code MI
        beta: Weight for Total Correlation (typically set high, e.g., 6.0)
        gamma: Weight for Dimension-wise KL
        recon_loss_type: Type of reconstruction loss ('BCE', 'BCE+Dice', 'BCE+IoU')
    """
    batch_size, latent_dim = z.size()
    
    # Compute BCE loss to provide pixel-wise gradients and avoid posterior collapse
    bce_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum') / batch_size
    
    if recon_loss_type == 'BCE':
        recon_loss = bce_loss
    elif recon_loss_type == 'BCE+Dice':
        # Apply sigmoid to raw logits of reconstruction for Dice computation
        recon_x_sigmoid = torch.sigmoid(recon_x)
        # Compute soft Dice loss and scale it by spatial dimensions
        _, _, H, W = recon_x.size()
        dice_val = dice_loss(recon_x_sigmoid, x) * (H * W)
        recon_loss = bce_loss + dice_val
    elif recon_loss_type == 'BCE+IoU':
        # Apply sigmoid to raw logits of reconstruction for IoU computation
        recon_x_sigmoid = torch.sigmoid(recon_x)
        # Compute soft IoU loss and scale it by spatial dimensions
        _, _, H, W = recon_x.size()
        iou_val = iou_loss(recon_x_sigmoid, x) * (H * W)
        recon_loss = bce_loss + iou_val
    else:
        raise ValueError(f"Unknown reconstruction loss type: {recon_loss_type}")

    log_q_z_per_dim = -0.5 * (
        math.log(2 * math.pi) + 
        logvar.unsqueeze(0) + 
        (z.unsqueeze(1) - mu.unsqueeze(0)).pow(2) / logvar.unsqueeze(0).exp()
    )
    
    log_q_z_given_x = log_q_z_per_dim.sum(dim=2) # log q(z_i | x_i) -> Shape: [B, B]
    log_q_z_given_x = torch.diagonal(log_q_z_given_x) # Shape: [B]
 
    log_dataset_size = math.log(dataset_size)
    log_batch_size = math.log(batch_size)
    
    log_prod_q_z = torch.logsumexp(log_q_z_per_dim, dim=1) - log_batch_size
    log_prod_q_z = log_prod_q_z.sum(dim=1) 
    
    log_q_z = torch.logsumexp(log_q_z_per_dim.sum(dim=2), dim=1) - log_batch_size
    log_p_z = (-0.5 * (math.log(2 * math.pi) + z.pow(2))).sum(dim=1) # Shape: [B]
 
    mi_loss = torch.mean(log_q_z_given_x - log_q_z)
    tc_loss = torch.mean(log_q_z - log_prod_q_z)
    dw_kl_loss = torch.mean(log_prod_q_z - log_p_z)
 
    total_loss = recon_loss + (alpha * mi_loss) + (beta * tc_loss) + (gamma * dw_kl_loss)
    
    return total_loss, recon_loss, mi_loss, tc_loss, dw_kl_loss


def save_reconstruction_grid(
    model: nn.Module, 
    device: torch.device, 
    dataloader: DataLoader, 
    epoch: int, 
    filepath: str
) -> Optional[Any]:
    """Generates and saves a grid showing original images alongside their VAE reconstructions."""
    if plt is None:
        return None
        
    model.eval()
    with torch.no_grad():
        # Retrieve one batch of test/val data
        for inputs in dataloader:
            inputs = inputs.to(device)
            break
            
        num_samples = min(inputs.size(0), 8)
        sample_inputs = inputs[:num_samples]
        recon_outputs, _, _, _ = model(sample_inputs)
        
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 1.5, 3.2))
        

        if num_samples == 1:
            axes = np.expand_dims(axes, axis=1)

        for i in range(num_samples):
            # Row 0: Original image
            axes[0, i].imshow(sample_inputs[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title("Original", fontsize=10, fontweight='bold')
                
            # Row 1: VAE Reconstruction image
            axes[1, i].imshow(torch.sigmoid(recon_outputs[i, 0]).cpu().numpy(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title("Reconstructed", fontsize=10, fontweight='bold')
                
        plt.tight_layout()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)
        logging.info(f"Visual quality grid saved to: {filepath}")
        return fig


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    writer: Optional[SummaryWriter]
) -> Tuple[float, float, float, float, float, int]:
    """Trains the model for one epoch."""
    model.train()
    total_loss, total_recon, total_mi, total_tc, total_dw_kl = 0.0, 0.0, 0.0, 0.0, 0.0
    total_samples = len(dataloader.dataset)

    for batch_idx, inputs in enumerate(dataloader):
        inputs = inputs.to(device)
        optimizer.zero_grad()

        # Linear warm-up coefficient for beta-tcvae loss weights
        anneal_coef = min(1.0, global_step / args.steps_to_max_capacity)

        recon_batch, z, mu, logvar = model(inputs)
        loss, recon, mi, tc, dw_kl = beta_tcvae_loss(
            recon_batch, inputs, z, mu, logvar,
            dataset_size=total_samples,
            alpha=args.alpha * anneal_coef,
            beta=args.beta * anneal_coef,
            gamma=args.gamma * anneal_coef,
            recon_loss_type=args.recon_loss_type
        )

        loss.backward()
        optimizer.step()

        # Track batch statistics
        total_loss += loss.item() * inputs.size(0)
        total_recon += recon.item() * inputs.size(0)
        total_mi += mi.item() * inputs.size(0)
        total_tc += tc.item() * inputs.size(0)
        total_dw_kl += dw_kl.item() * inputs.size(0)

        # Log step metric updates to TensorBoard
        if writer is not None:
            writer.add_scalar('Step/Loss', loss.item(), global_step)
            writer.add_scalar('Step/Recon', recon.item(), global_step)
            writer.add_scalar('Step/MI', mi.item(), global_step)
            writer.add_scalar('Step/TC', tc.item(), global_step)
            writer.add_scalar('Step/DW_KL', dw_kl.item(), global_step)

        global_step += 1

        if batch_idx % args.log_interval == 0:
            logging.info(
                f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{total_samples} "
                f"({100. * batch_idx / len(dataloader):.0f}%)]\t"
                f"Loss: {loss.item():.4f}\tRecon: {recon.item():.4f}\t"
                f"MI: {mi.item():.4f}\tTC: {tc.item():.4f}\tDW_KL: {dw_kl.item():.4f}"
            )

    avg_loss = total_loss / total_samples
    avg_recon = total_recon / total_samples
    avg_mi = total_mi / total_samples
    avg_tc = total_tc / total_samples
    avg_dw_kl = total_dw_kl / total_samples

    return avg_loss, avg_recon, avg_mi, avg_tc, avg_dw_kl, global_step


def evaluate(
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    args: argparse.Namespace,
    train_dataset_size: int
) -> Tuple[float, float, float, float, float,]:
    """Evaluates reconstruction loss on the validation dataset."""
    model.eval()
    total_loss, total_recon, total_mi, total_tc, total_dw_kl = 0.0, 0.0, 0.0, 0.0, 0.0
    total_samples = len(dataloader.dataset)

    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            recon_batch, z, mu, logvar = model(inputs)
            loss, recon, mi, tc, dw_kl = beta_tcvae_loss(
                recon_batch, inputs, z, mu, logvar,
                dataset_size=train_dataset_size,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                recon_loss_type=args.recon_loss_type
            )
            
            total_loss += loss.item() * inputs.size(0)
            total_recon += recon.item() * inputs.size(0)
            total_mi += mi.item() * inputs.size(0)
            total_tc += tc.item() * inputs.size(0)
            total_dw_kl += dw_kl.item() * inputs.size(0)

    avg_loss = total_loss / total_samples
    avg_recon = total_recon / total_samples
    avg_mi = total_mi / total_samples
    avg_tc = total_tc / total_samples
    avg_dw_kl = total_dw_kl / total_samples
    
    return avg_loss, avg_recon, avg_mi, avg_tc, avg_dw_kl


def main() -> None:
    args = parse_args()

    # Apply Random Seed Settings
    set_seed(args.seed)

    # Initialize Hardware Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Target execution hardware device detected: {device}")

    # Load Full dataset
    full_dataset = DSpritesDataset(
        path=args.data_path, 
        subsample_ratio=args.subsample_ratio, 
        exclude=args.exclude_zero_shot
    )

    # Split into train/validation loaders
    if args.val_split > 0.0:
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        # Fix generator seed for reproducible splitting
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        logging.info(f"Validation split enabled: {train_size} train samples, {val_size} val samples.")
    else:
        train_dataset = full_dataset
        val_dataset = None
        logging.info("Validation split disabled. Training on complete dataset pool.")

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset else None

    # Model and Optimizer setup
    model = BetaTCVAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup TensorBoard SummaryWriter
    writer = None
    if not args.no_tensorboard and SummaryWriter is not None:
        os.makedirs(args.tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        logging.info(f"TensorBoard logging activated at: {args.tb_log_dir}")
        # Log training hyperparameters
        writer.add_text("Hyperparameters", str(vars(args)))

    best_loss = float('inf')
    global_step = 0
    start_epoch = 1

    if args.resume:
        if os.path.exists(args.checkpoint_path):
            logging.info(f"Loading checkpoint from {args.checkpoint_path} to resume training...")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint.get('best_loss', float('inf'))
            global_step = checkpoint.get('global_step', 0)
            start_epoch = checkpoint.get('epoch', 0) + 1
            logging.info(f"Resumed from epoch {start_epoch - 1} with best loss {best_loss:.4f} and global step {global_step}.")
        else:
            logging.warning(f"Checkpoint file '{args.checkpoint_path}' not found. Training from scratch.")

    logging.info("Starting Optimization loop...")
    
    try:
        train_dataset_size = len(train_loader.dataset)
        end_epoch = start_epoch + args.epochs - 1
        for epoch in range(start_epoch, end_epoch + 1):
            # 1. Training Phase
            train_loss, train_recon, train_mi, train_tc, train_dw_kl, global_step = train_one_epoch(
                model=model,
                device=device,
                dataloader=train_loader,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                args=args,
                writer=writer
            )

            # 2. Validation Phase
            val_loss, val_recon, val_mi, val_tc, val_dw_kl = 0.0, 0.0, 0.0, 0.0, 0.0
            if val_loader is not None:
                val_loss, val_recon, val_mi, val_tc, val_dw_kl = evaluate(
                    model=model,
                    device=device,
                    dataloader=val_loader,
                    args=args,
                    train_dataset_size=train_dataset_size
                )
                
            # Logger Console Output
            if val_loader is not None:
                logging.info(
                    f"Epoch [{epoch}/{end_epoch}] complete -> "
                    f"Avg Train Loss: {train_loss:.2f} (Recon: {train_recon:.2f}, TC: {train_tc:.2f}) | "
                    f"Avg Val Loss: {val_loss:.2f} (Recon: {val_recon:.2f}, TC: {val_tc:.2f})"
                )
            else:
                logging.info(
                    f"Epoch [{epoch}/{end_epoch}] complete -> "
                    f"Avg Train Loss: {train_loss:.2f} (Recon: {train_recon:.2f}, TC: {train_tc:.2f})"
                )

            # Log epoch metric summaries to TensorBoard
            if writer is not None:
                writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
                writer.add_scalar('Epoch/Train_Recon', train_recon, epoch)
                writer.add_scalar('Epoch/Train_MI', train_mi, epoch)
                writer.add_scalar('Epoch/Train_TC', train_tc, epoch)
                writer.add_scalar('Epoch/Train_DW_KL', train_dw_kl, epoch)
                if val_loader is not None:
                    writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
                    writer.add_scalar('Epoch/Val_Recon', val_recon, epoch)
                    writer.add_scalar('Epoch/Val_MI', val_mi, epoch)
                    writer.add_scalar('Epoch/Val_TC', val_tc, epoch)
                    writer.add_scalar('Epoch/Val_DW_KL', val_dw_kl, epoch)

            # 3. Save Visual Quality Comparison Grid
            if not args.no_visual_grid and plt is not None:
                grid_filepath = os.path.join(args.results_dir, f"epoch_{epoch}.png")
                # Use val loader for grids if available, fallback to train loader
                grid_loader = val_loader if val_loader is not None else train_loader
                fig = save_reconstruction_grid(model, device, grid_loader, epoch, grid_filepath)
                if writer is not None and fig is not None:
                    writer.add_figure('reconstructions/epoch_grid', fig, global_step=epoch)

            # 4. Checkpoint Check & Save
            # Decide validation metric to monitor
            monitored_loss = val_loss if val_loader is not None else train_loss
            if monitored_loss < best_loss:
                best_loss = monitored_loss
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'global_step': global_step,
                    'args': vars(args)
                }
                
                # Secure directory existence
                checkpoint_dir = os.path.dirname(args.checkpoint_path)
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                
                torch.save(checkpoint_dict, args.checkpoint_path)
                logging.info(f"New best validation model checkpoint saved: {args.checkpoint_path} (Loss: {best_loss:.4f})")

        logging.info("Training complete.")

    except KeyboardInterrupt:
        logging.warning("Training interrupted by KeyboardInterrupt. Attempting to save emergency checkpoint...")
        emergency_path = "emergency_vae_checkpoint.pth"
        torch.save({
            'epoch': 'interrupted',
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'args': vars(args)
        }, emergency_path)
        logging.info(f"Emergency state saved to: {emergency_path}")

    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()
