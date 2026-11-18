import os 
import sys
import argparse
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from gmm import GaussianMixture
from gmm_save_load import save_gmm
from train_vae import *
import matplotlib.pyplot as plt


class DSpritesDataset(Dataset):
    """Dataset Wrapper for GMM on Latent Space with VAE zero-shot condition filtering"""
    def __init__(self, path, subsample_ratio=1.0, exclude_zero_shot=True):
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True, encoding='bytes')
            all_images = data['imgs'] 
            
            # if exclude_zero_shot:
            #     latents_classes = data['latents_classes']
            #     shape_classes = latents_classes[:, 1] 
            #     pos_x_classes = latents_classes[:, 4]  
                
            #     # Exclude hearts on the left (matches VAE training exclusion)
            #     zero_shot_mask = (shape_classes == 2) & (pos_x_classes < 16)
            #     valid_train_mask = ~zero_shot_mask
                
            #     all_images = all_images[valid_train_mask]
            #     print(f"Excluded {np.sum(zero_shot_mask)} zero-shot samples from training pool.")
            
            if subsample_ratio < 1.0:
                np.random.seed(42)
                total_samples = len(all_images)
                keep_count = int(total_samples * subsample_ratio)
                indices = np.random.choice(total_samples, keep_count, replace=False)
                all_images = all_images[indices]
                print(f"Subsampled dataset by ratio {subsample_ratio}. Size: {len(all_images)}.")

            self.images = all_images
            print(f"Final GMM Dataset Size: {len(self.images)} samples.")
        else:
            print(f"Dataset path '{path}' not found. Initializing dummy dataset.")
            self.images = np.random.randint(0, 2, (1000, 64, 64), dtype=np.uint8)
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).float().unsqueeze(0) 
        return img

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
    print(f"Random seed set to: {seed}")


def extract_latent_means(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    gmm_samples: int = None
) -> torch.Tensor:
    """Encode full dataset through VAE and optionally subsample latent means for GMM fitting."""
    print("Extracting latent means from VAE...")
    model.eval()
    model.to(device)
    
    latent_means = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            latent_means.append(mu.cpu())
    
    X_all = torch.vstack(latent_means)
    print(f"Encoded full dataset: {X_all.shape[0]} latent vectors.")
    
    # Subsample for GMM fitting if dataset is larger than gmm_samples
    if gmm_samples is not None and gmm_samples < X_all.shape[0]:
        indices = torch.randperm(X_all.shape[0])[:gmm_samples]
        X_train = X_all[indices]
        print(f"Subsampled {gmm_samples} latent vectors for GMM fitting.")
    else:
        X_train = X_all
    
    return X_train.to(device)


def train_and_evaluate_gmm(
    n_components: int, 
    device: str,
    X_train: torch.Tensor,
    covariance_type: str = "full",
    init_params: str = "kmeans",
    delta: float = 1e-4,
    n_iter: int = 100,
) -> GaussianMixture:
    n_samples, n_features = X_train.shape
    print(f"GMM training data shape: {X_train.shape}")

    print(f"Fitting GMM with {n_components} components ({covariance_type} covariance) on {device}...")
    gmm = GaussianMixture(
        n_components=n_components, 
        n_features=n_features, 
        covariance_type=covariance_type, 
        init_params=init_params
    ).to(device)
    
    gmm.fit(X_train, delta=delta, n_iter=n_iter)
    print("GMM fitting complete.")
        
    return gmm

def sample_results(gmm: GaussianMixture, model: torch.nn.Module, num_samples: int, save_path: str, save_name: str) -> None:
    print(f"\nSampling {num_samples} vectors from the GMM...")
    samples, _ = gmm.sample(num_samples)
    
    model.eval()
    with torch.no_grad():
        recons = torch.sigmoid(model.decode(samples))
    recons = recons.squeeze(1)
    recons = recons.cpu().numpy()
    
    # Save a grid of images
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    axes = axes.flatten()
    for i in range(min(num_samples, len(axes))):
        axes[i].imshow(recons[i], cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, save_name)
    plt.savefig(full_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Sampled reconstruction grid saved to: {full_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible Training Script for GMM on Latent Space")
    
    # Dataset Parameters
    parser.add_argument('--data-path', type=str, default='/home/gandalf/objects/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                        help='Path to dSprites .npz archive file')
    parser.add_argument('--subsample-ratio', type=float, default=1.0,
                        help='Dataset subsampling ratio (between 0.0 and 1.0, matches VAE training)')
    parser.add_argument('--no-exclude-zero-shot', dest='exclude_zero_shot', action='store_false',
                        help='Disable zero-shot class training sample exclusions')
    parser.set_defaults(exclude_zero_shot=True)
    
    # Model/Training Hyperparameters
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Training batch size')
    parser.add_argument('--latent-dim', type=int, default=12 ,
                        help='Dimensionality of target latent vector z')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deterministic reproducibility')
    
    # GMM Parameters
    parser.add_argument('--num-components', type=int, default=20,
                        help='Number of Gaussian components')
    parser.add_argument('--component-list', type=int, nargs='+', default=None,
                        help='List of component values to perform experimental training on (min BIC selects the best)')
    parser.add_argument('--gmm-samples', type=int, default=60000,
                        help='Number of latent vectors to subsample for GMM fitting (avoids GPU OOM)')
    parser.add_argument('--covariance-type', type=str, default='full', choices=['full', 'diag', 'spherical'],
                        help='Type of covariance to use in the GMM model')
    parser.add_argument('--init-params', type=str, default='kmeans', choices=['kmeans', 'random'],
                        help='Method used to initialize the weights')
    parser.add_argument('--delta', type=float, default=1e-4,
                        help='EM convergence threshold')
    parser.add_argument('--n-iter', type=int, default=100,
                        help='Max EM iteration steps')
    
    # Checkpointing / Saving Paths
    parser.add_argument('--vae-checkpoint', type=str, default='best_beta_vae_dsprites_leave.pth',
                        help='Path to the trained BetaVAE model weights')
    parser.add_argument('--gmm-checkpoint', type=str, default='gmm_checkpoint.pth',
                        help='Path to save GMM checkpoint')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory where generated sample images are saved')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(args.seed)

    # Initialize fair-sampled dataset with zero-shot condition
    dataset = DSpritesDataset(args.data_path, subsample_ratio=args.subsample_ratio, exclude_zero_shot=args.exclude_zero_shot)
    print(f"Loaded dataset with {len(dataset)} samples.")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load trained Beta-VAE model
    if not os.path.exists(args.vae_checkpoint):
        print(f"Error: VAE checkpoint '{args.vae_checkpoint}' does not exist! Please train the VAE first.")
        sys.exit(1)
        
    checkpoint = torch.load(args.vae_checkpoint, map_location=device)
    latent_dim = checkpoint.get('args', {}).get('latent_dim', args.latent_dim)
    print(f"Detected VAE latent dimension: {latent_dim}")
    vae = BetaTCVAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    print("Beta-TCVAE successfully loaded.")

    # Extract and subsample latent means (encode full dataset, subsample for GMM)
    X_train = extract_latent_means(vae, dataloader, device, gmm_samples=args.gmm_samples)

    # Train GMM
    if args.component_list is not None and len(args.component_list) > 0:
        print(f"Starting experimental GMM training over components: {args.component_list}")

        best_bic = float('inf')
        best_gmm = None
        best_k = None

        for k in args.component_list:
            # Set seed before each run for reproducibility
            set_seed(args.seed)
            gmm_k = train_and_evaluate_gmm(
                n_components=k,
                device=device,
                X_train=X_train,
                covariance_type=args.covariance_type,
                init_params=args.init_params,
                delta=args.delta,
                n_iter=args.n_iter,
            )
            # Calculate BIC using the BIC function in gmm.py
            bic_val = gmm_k.bic(X_train).item()
            print(f"GMM with k={k} components has BIC = {bic_val:.4f}")
            if bic_val < best_bic:
                best_bic = bic_val
                if best_gmm is not None:
                    del best_gmm
                best_gmm = gmm_k
                best_k = k
            else:
                del gmm_k
        
        print(f"\nBest model selected: k={best_k} components with minimized BIC = {best_bic:.4f}")
        gmm = best_gmm
    else:
        # Train single GMM
        gmm = train_and_evaluate_gmm(
            n_components=args.num_components,
            device=device,
            X_train=X_train,
            covariance_type=args.covariance_type,
            init_params=args.init_params,
            delta=args.delta,
            n_iter=args.n_iter
        )

    # Save GMM model
    save_gmm(gmm, args.gmm_checkpoint)

    # Generate and save a grid of sampled reconstruction images
    sample_results(gmm, vae, num_samples=32, save_path=args.results_dir, save_name="gmm_sampled_images.png")


if __name__ == '__main__':
    main()