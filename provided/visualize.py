"""
Provided visualization utilities for HW2.
These functions help students visualize their results consistently.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_alphabet_grid(generator, device='cuda', z_dim=100, seed=None):
    """
    Generate and plot a grid of all 26 letters.
    
    Args:
        generator: Trained generator model
        device: Device to run on
        z_dim: Dimension of latent space
        seed: Random seed for reproducibility
    
    Returns:
        figure: Matplotlib figure object
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    generator.eval()
    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(4, 7, figure=fig)
    
    with torch.no_grad():
        for i in range(26):
            ax = fig.add_subplot(gs[i // 7, i % 7])
            
            # Generate random z
            z = torch.randn(1, z_dim).to(device)
            
            # Generate image
            if hasattr(generator, 'conditional') and generator.conditional:
                # Conditional GAN - provide letter label
                label = torch.zeros(1, 26).to(device)
                label[0, i] = 1
                fake_img = generator(z, label).squeeze().cpu()
            else:
                # Unconditional - just generate
                fake_img = generator(z).squeeze().cpu()
            
            # Convert from [-1, 1] to [0, 1] for display
            fake_img = (fake_img + 1) / 2
            
            ax.imshow(fake_img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(chr(65 + i), fontsize=12)
            ax.axis('off')
    
    plt.suptitle('Generated Alphabet', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig

def plot_training_history(history, save_path=None):
    """
    Plot training curves for GAN.
    
    Args:
        history: Dict with 'd_loss', 'g_loss', 'mode_coverage', etc.
        save_path: Optional path to save figure
    
    Returns:
        figure: Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Discriminator and Generator losses
    ax = axes[0, 0]
    if 'd_loss' in history:
        ax.plot(history.get('epoch', range(len(history['d_loss']))), 
                history['d_loss'], label='D Loss', alpha=0.7)
    if 'g_loss' in history:
        ax.plot(history.get('epoch', range(len(history['g_loss']))), 
                history['g_loss'], label='G Loss', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mode coverage
    ax = axes[0, 1]
    if 'mode_coverage' in history:
        coverage = history['mode_coverage']
        epochs = np.linspace(0, len(coverage) * 10, len(coverage))
        ax.plot(epochs, coverage, marker='o', linewidth=2, markersize=6)
        ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect (26/26)')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% (13/26)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mode Coverage')
        ax.set_title('Letter Coverage Score')
        ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Loss ratio
    ax = axes[1, 0]
    if 'd_loss' in history and 'g_loss' in history:
        ratio = np.array(history['d_loss']) / (np.array(history['g_loss']) + 1e-8)
        ax.plot(history.get('epoch', range(len(ratio))), ratio, alpha=0.7)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('D Loss / G Loss')
        ax.set_title('Loss Balance')
        ax.set_ylim([0, 5])
        ax.grid(True, alpha=0.3)
    
    # Gradient norms (if available)
    ax = axes[1, 1]
    if 'g_grad_norm' in history:
        ax.plot(history.get('epoch', range(len(history['g_grad_norm']))), 
                history['g_grad_norm'], label='G gradients', alpha=0.7)
    if 'd_grad_norm' in history:
        ax.plot(history.get('epoch', range(len(history['d_grad_norm']))), 
                history['d_grad_norm'], label='D gradients', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Magnitudes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('GAN Training Dynamics', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_drum_pattern(pattern, title='Drum Pattern'):
    """
    Visualize a drum pattern as a piano roll.
    
    Args:
        pattern: Binary array [16, 9] or tensor
        title: Title for the plot
    
    Returns:
        figure: Matplotlib figure object
    """
    if torch.is_tensor(pattern):
        pattern = pattern.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    instruments = ['Kick', 'Snare', 'Hi-hat C', 'Hi-hat O', 
                  'Tom L', 'Tom H', 'Crash', 'Ride', 'Clap']
    
    # Create piano roll visualization
    for i in range(9):
        for j in range(16):
            if pattern[j, i] > 0.5:  # Threshold for binary
                ax.add_patch(plt.Rectangle((j, i), 1, 1, 
                                          facecolor='blue', 
                                          edgecolor='black',
                                          linewidth=0.5))
    
    # Grid
    for i in range(17):
        ax.axvline(i, color='gray', linewidth=0.5, alpha=0.5)
        if i % 4 == 0:
            ax.axvline(i, color='black', linewidth=1.5, alpha=0.7)
    
    for i in range(10):
        ax.axhline(i, color='gray', linewidth=0.5, alpha=0.5)
    
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_xticks(range(16))
    ax.set_xticklabels([str(i+1) for i in range(16)])
    ax.set_yticks(range(9))
    ax.set_yticklabels(instruments)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Instrument')
    ax.set_title(title)
    ax.invert_yaxis()
    
    return fig

def plot_latent_space_2d(latent_codes, labels=None, title='Latent Space'):
    """
    Plot 2D visualization of latent space.
    
    Args:
        latent_codes: Array of latent codes [N, D]
        labels: Optional labels for coloring
        title: Title for plot
    
    Returns:
        figure: Matplotlib figure object
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    if latent_codes.shape[1] > 2:
        # Use t-SNE for dimensionality reduction
        if latent_codes.shape[0] > 1000:
            # Subsample for efficiency
            indices = np.random.choice(latent_codes.shape[0], 1000, replace=False)
            latent_codes = latent_codes[indices]
            if labels is not None:
                labels = labels[indices]
        
        # First reduce with PCA if very high dimensional
        if latent_codes.shape[1] > 50:
            pca = PCA(n_components=50)
            latent_codes = pca.fit_transform(latent_codes)
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        coords = tsne.fit_transform(latent_codes)
    else:
        coords = latent_codes
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                           c=labels, cmap='tab10', 
                           alpha=0.7, s=30)
        plt.colorbar(scatter, ax=ax, label='Class/Style')
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=30)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig