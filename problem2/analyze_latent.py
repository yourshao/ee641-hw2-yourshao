"""
Latent space analysis tools for hierarchical VAE.
"""
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

SAVE_DIR = os.path.join("results", "visualize")
os.makedirs(SAVE_DIR, exist_ok=True)


def visualize_latent_hierarchy(model, data_loader, device='cuda'):
    """
    Visualize the two-level latent space structure.
    
    TODO:
    1. Encode all data to get z_high and z_low
    2. Use t-SNE to visualize z_high (colored by genre)
    3. For each z_high cluster, show z_low variations
    4. Create hierarchical visualization
    """
    model.eval()
    z_high_all, z_low_all, labels_all = [], [], []

    with torch.no_grad():
        for patterns, styles, _ in data_loader:
            patterns = patterns.to(device)
            mu_low, logvar_low, z_low, mu_high, logvar_high, z_high = model.encode_hierarchy(patterns)
            z_high_all.append(z_high.cpu().numpy())
            z_low_all.append(z_low.cpu().numpy())
            labels_all.append(styles.numpy())

    z_high_all = np.concatenate(z_high_all, axis=0)
    z_low_all = np.concatenate(z_low_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    # --- t-SNE on high-level latent (style) ---
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_high_2d = tsne.fit_transform(z_high_all)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_high_2d[:, 0], z_high_2d[:, 1], c=labels_all, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, label='Style label')
    plt.title('t-SNE of High-level Latent (z_high)')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "tsne_z_high.png"), dpi=200)
    plt.close()

    # --- PCA on low-level latent (variation) ---
    pca = PCA(n_components=2)
    z_low_2d = pca.fit_transform(z_low_all)
    plt.figure(figsize=(8, 6))
    plt.scatter(z_low_2d[:, 0], z_low_2d[:, 1], c=labels_all, cmap='tab10', s=10, alpha=0.7)
    plt.title('PCA of Low-level Latent (z_low)')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "pca_z_low.png"), dpi=200)
    plt.close()



def interpolate_styles(model, pattern1, pattern2, n_steps=10, device='cuda'):
    """
    Interpolate between two drum patterns at both latent levels.
    
    TODO:
    1. Encode both patterns to get latents
    2. Interpolate z_high (style transition)
    3. Interpolate z_low (variation transition)
    4. Decode and visualize both paths
    5. Compare smooth vs abrupt transitions
    """
    model.eval()
    pattern1 = pattern1.unsqueeze(0).to(device)
    pattern2 = pattern2.unsqueeze(0).to(device)

    with torch.no_grad():
        mu_low1, logvar_low1, z_low1, mu_high1, logvar_high1, z_high1 = model.encode_hierarchy(pattern1)
        mu_low2, logvar_low2, z_low2, mu_high2, logvar_high2, z_high2 = model.encode_hierarchy(pattern2)

        z_high_list, z_low_list = [], []
        for alpha in np.linspace(0, 1, n_steps):
            z_high_interp = (1 - alpha) * z_high1 + alpha * z_high2
            z_low_interp = (1 - alpha) * z_low1 + alpha * z_low2
            z_high_list.append(z_high_interp)
            z_low_list.append(z_low_interp)

        z_high_cat = torch.cat(z_high_list, dim=0)
        z_low_cat = torch.cat(z_low_list, dim=0)
        logits = model.decode_hierarchy(z_high_cat, z_low_cat, temperature=1.0)
        probs = torch.sigmoid(logits).cpu().numpy()

    # --- Plot ---
    fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 3))
    for i in range(n_steps):
        axes[i].imshow(probs[i], aspect='auto', cmap='Blues', vmin=0, vmax=1)
        axes[i].set_title(f"α={i/(n_steps-1):.2f}")
        axes[i].axis('off')
    plt.suptitle('Interpolation Between Two Styles')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "interpolation.png"), dpi=200)
    plt.close()
    print(f"[Saved] Interpolation figure → {SAVE_DIR}/interpolation.png")


def measure_disentanglement(model, data_loader, device='cuda'):
    """
    Measure how well the hierarchy disentangles style from variation.
    
    TODO:
    1. Group patterns by genre
    2. Compute z_high variance within vs across genres
    3. Compute z_low variance for same genre
    4. Return disentanglement metrics
    """
    model.eval()
    z_high_by_style, z_low_by_style = {}, {}

    with torch.no_grad():
        for patterns, styles, _ in data_loader:
            patterns = patterns.to(device)
            mu_low, logvar_low, z_low, mu_high, logvar_high, z_high = model.encode_hierarchy(patterns)
            for s in torch.unique(styles):
                s = s.item()
                mask = (styles == s)
                if s not in z_high_by_style:
                    z_high_by_style[s] = []
                    z_low_by_style[s] = []
                z_high_by_style[s].append(z_high[mask].cpu().numpy())
                z_low_by_style[s].append(z_low[mask].cpu().numpy())

    style_means = []
    within_var_high, within_var_low = [], []
    for s in z_high_by_style:
        z_high_s = np.concatenate(z_high_by_style[s], axis=0)
        z_low_s = np.concatenate(z_low_by_style[s], axis=0)
        style_means.append(z_high_s.mean(axis=0))
        within_var_high.append(np.mean(np.var(z_high_s, axis=0)))
        within_var_low.append(np.mean(np.var(z_low_s, axis=0)))

    within_high = np.mean(within_var_high)
    within_low = np.mean(within_var_low)
    between_high = np.mean(np.var(np.stack(style_means, axis=0), axis=0))

    disentanglement_score = between_high / (within_high + 1e-6)

    return {
        'within_var_high': within_high,
        'within_var_low': within_low,
        'between_var_high': between_high,
        'disentanglement_score': disentanglement_score
    }

def controllable_generation(model, genre_labels, device='cuda'):
    """
    Test controllable generation using the hierarchy.
    
    TODO:
    1. Learn genre embeddings in z_high space
    2. Generate patterns with specified genre
    3. Control complexity via z_low sampling temperature
    4. Evaluate genre classification accuracy
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for genre_id in genre_labels:
            z_high = torch.zeros(1, model.z_high_dim, device=device)
            z_high[:, 0] = genre_id / len(genre_labels) * 2 - 1  # 简单编码风格方向
            patterns_per_temp = []

            for temp in [1.5, 1.0, 0.7, 0.5]:
                logits = model.decode_hierarchy(z_high, z_low=None, temperature=temp)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                patterns_per_temp.append(probs)

            results[genre_id] = patterns_per_temp

    # --- Plot ---
    n_genres = len(results)
    n_temps = len(next(iter(results.values())))
    fig, axes = plt.subplots(n_genres, n_temps, figsize=(2.5 * n_temps, 2 * n_genres))
    for i, genre_id in enumerate(results):
        for j, pattern in enumerate(results[genre_id]):
            ax = axes[i, j] if n_genres > 1 else axes[j]
            ax.imshow(pattern, aspect='auto', cmap='Blues', vmin=0, vmax=1)
            ax.set_title(f"Genre {genre_id}, T={ [1.5,1.0,0.7,0.5][j] }")
            ax.axis('off')
    plt.suptitle('Controllable Generation (Style × Complexity)')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "controllable_generation.png"), dpi=200)
    plt.close()
    print(f"[Saved] Controllable generation → {SAVE_DIR}/controllable_generation.png")


    return results