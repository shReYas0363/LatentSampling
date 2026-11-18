
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys

from train_vae import BetaTCVAE


FACTOR_NAMES = ['Shape', 'Scale', 'Orientation', 'Pos X', 'Pos Y']
FACTOR_INDICES = [1, 2, 3, 4, 5]


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_dsprites_data(path: str, subsample_size: int = 10000) -> tuple:
    if not os.path.exists(path):
        raise FileNotFoundError(f"dSprites dataset not found at {path}")
    
    print(f"Loading dSprites dataset from {path}...")
    data = np.load(path, allow_pickle=True, encoding='bytes')
    imgs = data['imgs']
    latents = data['latents_classes']
    
    np.random.seed(42)
    indices = np.random.choice(len(imgs), subsample_size, replace=False)
    return imgs[indices], latents[indices]


def extract_latent_representations(model, imgs, device, batch_size=256):
    model.eval()
    mu_list = []
    
    print("Extracting latent representations (mu) from the VAE encoder...")
    num_samples = len(imgs)
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_imgs = imgs[i:i+batch_size]
            # Convert to float and add channel dimension: [B, 1, 64, 64]
            batch_tensor = torch.from_numpy(batch_imgs).float().unsqueeze(1).to(device)
            mu, _ = model.encode(batch_tensor)
            mu_list.append(mu.cpu().numpy())
            
    return np.concatenate(mu_list, axis=0)


def compute_mi(z: np.ndarray, y: np.ndarray, num_bins: int = 20) -> float:
    """Computes Mutual Information between binned continuous z and discrete y."""
    y_min, y_max = int(y.min()), int(y.max())
    y_bins = y_max - y_min + 1
    
    hist_joint, _, _ = np.histogram2d(z, y, bins=[num_bins, y_bins])
    p_joint = hist_joint / np.sum(hist_joint)
    
    p_z = np.sum(p_joint, axis=1, keepdims=True)
    p_y = np.sum(p_joint, axis=0, keepdims=True)
    
    mi = 0.0
    for i in range(p_joint.shape[0]):
        for j in range(p_joint.shape[1]):
            if p_joint[i, j] > 0:
                mi += p_joint[i, j] * np.log2(p_joint[i, j] / (p_z[i, 0] * p_y[0, j] + 1e-12))
    return mi


def compute_entropy(y: np.ndarray) -> float:
    """Computes entropy H(y) of discrete variable y."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p + 1e-12))


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    args = sys.argv
    checkpoint_path = args[1]
    data_path = '/home/gandalf/objects/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    
    # 1. Load VAE model
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Trained VAE checkpoint not found at {checkpoint_path}")
        
    print(f"Loading VAE checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get('args', {})
    latent_dim = checkpoint_args.get('latent_dim', 10)
    
    model = BetaTCVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model successfully loaded. Latent dimension: {latent_dim}")
    
    # 2. Load dataset
    imgs, latents = load_dsprites_data(data_path, subsample_size=10000)
    
    # 3. Get latent representations
    mu_np = extract_latent_representations(model, imgs, device)
    
    # 4. PCA to 2 dimensions
    print("Performing PCA down to 2 dimensions...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    mu_pca = pca.fit_transform(mu_np)
    
    # Plot PCA colored by each factor
    fig, axes = plt.subplots(1, len(FACTOR_INDICES), figsize=(22, 4))
    for i, factor_idx in enumerate(FACTOR_INDICES):
        ax = axes[i]
        scatter = ax.scatter(mu_pca[:, 0], mu_pca[:, 1], c=latents[:, factor_idx], cmap='viridis', s=2, alpha=0.6)
        fig.colorbar(scatter, ax=ax)
        ax.set_title(f'PCA colored by {FACTOR_NAMES[i]}')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
    plt.tight_layout()
    pca_plot_path = 'pca_latent_space.png'
    plt.savefig(pca_plot_path, dpi=150)
    plt.close()
    print(f"Saved 2D PCA plots to {pca_plot_path}")
    
    # 5. Spearman Rank Correlation
    print("Computing Spearman Rank Correlation...")
    from scipy.stats import spearmanr
    corr_matrix = np.zeros((latent_dim, len(FACTOR_INDICES)))
    for j in range(latent_dim):
        for k, factor_idx in enumerate(FACTOR_INDICES):
            corr, _ = spearmanr(mu_np[:, j], latents[:, factor_idx])
            corr_matrix[j, k] = abs(corr)

    
    # 6. Mutual Information and Mutual Information Gap (MIG)
    print("Computing Mutual Information and Mutual Information Gap (MIG)...")
    mi_matrix = np.zeros((latent_dim, len(FACTOR_INDICES)))
    mig_scores = {}
    
    print("\n--- Disentanglement Metrics (Mutual Information Gap) ---")
    for k, factor_idx in enumerate(FACTOR_INDICES):
        y = latents[:, factor_idx]
        entropy_y = compute_entropy(y)
        
        mis = []
        for j in range(latent_dim):
            mi = compute_mi(mu_np[:, j], y, num_bins=20)
            mi_matrix[j, k] = mi
            mis.append((mi, j))
            
        mis.sort(reverse=True)
        if entropy_y > 0:
            mig = (mis[0][0] - mis[1][0]) / entropy_y
        else:
            mig = 0.0
            
        mig_scores[FACTOR_NAMES[k]] = mig
        print(f"MIG ({FACTOR_NAMES[k]}): {mig:.4f} (Highest MI with z_{mis[0][1]}: {mis[0][0]:.4f}, Second: z_{mis[1][1]}: {mis[1][0]:.4f})")
        
    avg_mig = np.mean(list(mig_scores.values()))
    print(f"Average Mutual Information Gap (MIG): {avg_mig:.4f}")
    print("------------------------------------------------------\n")
    

    
    # 7. Latent Traversals
    print("Generating latent traversals for the most disentangled dimensions...")
    traversal_values = np.linspace(-3, 3, 9)
    num_factors = len(FACTOR_NAMES)
    
    # Identify the best latent dimension for each ground truth factor
    best_latents = [np.argmax(mi_matrix[:, k]) for k in range(len(FACTOR_INDICES))]
    
    fig, axes = plt.subplots(num_factors, len(traversal_values), figsize=(len(traversal_values) * 1.5, num_factors * 1.5))
    
    with torch.no_grad():
        for r, (factor_name, j_best) in enumerate(zip(FACTOR_NAMES, best_latents)):
            for c, val in enumerate(traversal_values):
                z_sample = torch.zeros(1, latent_dim).to(device)
                z_sample[0, j_best] = val
                recon_img = torch.sigmoid(model.decode(z_sample))
                recon_img_np = recon_img[0, 0].cpu().numpy()
                
                ax = axes[r, c]
                ax.imshow(recon_img_np, cmap='gray')
                ax.axis('off')
                
                if c == 0:
                    ax.set_ylabel(f'{factor_name}\n(z_{j_best})', fontsize=10, rotation=0, labelpad=40, va='center')
                    ax.axis('on')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                if r == 0:
                    ax.set_title(f'{val:+.2f}', fontsize=8)
                    
    plt.tight_layout()
    traversal_plot_path = 'latent_traversals.png'
    plt.savefig(traversal_plot_path, dpi=150)
    plt.close()
    print(f"Saved latent traversal grid to {traversal_plot_path}")
    
    # 8. Reconstruction / Decoding by Encoding
    print("Generating examples of decoding by encoding (reconstruction)...")
    num_reconstruct_examples = 8
    # Select deterministic indices spread across the dataset to get a variety of shapes/factors
    recon_indices = np.linspace(0, len(imgs) - 1, num_reconstruct_examples, dtype=int)
    
    fig, axes = plt.subplots(2, num_reconstruct_examples, figsize=(num_reconstruct_examples * 1.5, 3))
    
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(recon_indices):
            orig_img = imgs[idx]
            # Convert to tensor: [1, 1, 64, 64]
            img_tensor = torch.from_numpy(orig_img).float().unsqueeze(0).unsqueeze(0).to(device)
            mu, _ = model.encode(img_tensor)
            recon_img = torch.sigmoid(model.decode(mu))
            recon_img_np = recon_img[0, 0].cpu().numpy()
            
            # Plot Original
            ax_orig = axes[0, i]
            ax_orig.imshow(orig_img, cmap='gray')
            ax_orig.axis('off')
            if i == 0:
                ax_orig.set_ylabel('Original', fontsize=12, labelpad=10)
                ax_orig.axis('on')
                ax_orig.set_xticks([])
                ax_orig.set_yticks([])
                for spine in ax_orig.spines.values():
                    spine.set_visible(False)
            
            # Plot Reconstructed
            ax_recon = axes[1, i]
            ax_recon.imshow(recon_img_np, cmap='gray')
            ax_recon.axis('off')
            if i == 0:
                ax_recon.set_ylabel('Reconstructed', fontsize=12, labelpad=10)
                ax_recon.axis('on')
                ax_recon.set_xticks([])
                ax_recon.set_yticks([])
                for spine in ax_recon.spines.values():
                    spine.set_visible(False)
                    
    plt.tight_layout()
    recon_plot_path = 'reconstructions.png'
    plt.savefig(recon_plot_path, dpi=150)
    plt.close()
    print(f"Saved reconstruction comparison (decoding by encoding) to {recon_plot_path}")
    
    print("Latent space analysis completed successfully.")


if __name__ == '__main__':
    main()

