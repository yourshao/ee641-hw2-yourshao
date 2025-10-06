"""
Training implementations for hierarchical VAE with posterior collapse prevention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


def kl_annealing_schedule(epoch):
    """
    TODO: Implement KL annealing schedule
    Start with beta ≈ 0, gradually increase to 1.0
    Consider cyclical annealing for better results
    """
    period = 20
    pos = epoch % period
    return pos / 10.0 if pos <= 10 else 1.0

def temperature_annealing_schedule(epoch):
    hi, lo, decay = 2.0, 0.7, 0.02
    val = hi * (1.0 / (1.0 + decay * epoch))
    return max(lo, val)


def train_hierarchical_vae(model, data_loader, num_epochs=100, device='cuda'):
    """
    Train hierarchical VAE with KL annealing and other tricks.
    
    Implements several techniques to prevent posterior collapse:
    1. KL annealing (gradual beta increase)
    2. Free bits (minimum KL per dimension)
    3. Temperature annealing for discrete outputs
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # KL annealing schedule

    # Free bits threshold
    free_bits = 0.5  # Minimum nats per latent dimension
    
    history = defaultdict(list)
    
    for epoch in range(num_epochs):
        beta = kl_annealing_schedule(epoch)
        model.train()

        temperature = temperature_annealing_schedule(epoch)

        epoch_loss = 0
        epoch_recon = 0
        epoch_kl_low = 0
        epoch_kl_high = 0
        n_samples = 0

        for batch_idx, patterns in enumerate(data_loader):
            patterns = patterns.to(device)
            
            # TODO: Implement training step
            # 1. Forward pass through hierarchical VAE
            # 2. Compute reconstruction loss
            # 3. Compute KL divergences (both levels)
            # 4. Apply free bits to prevent collapse
            # 5. Total loss = recon_loss + beta * kl_loss
            # 6. Backward and optimize

            optimizer.zero_grad()

            recon, mu_low, logvar_low, mu_high, logvar_high = model(patterns)

            recon_loss = F.binary_cross_entropy_with_logits(
                recon.view(-1), patterns.view(-1), reduction='sum'
            )

            # KL(q(z)|p(z)) for both levels
            kl_low_vec  = -0.5 * (1 + logvar_low - mu_low.pow(2) - logvar_low.exp())   # [B, D_low]
            kl_high_vec = -0.5 * (1 + logvar_high - mu_high.pow(2) - logvar_high.exp()) # [B, D_high]

            kl_low  = torch.clamp(kl_low_vec,  min=free_bits).sum()
            kl_high = torch.clamp(kl_high_vec, min=free_bits).sum()

            loss = recon_loss + beta * (kl_low + kl_high)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 累积统计
            bs = patterns.size(0)
            n_samples += bs
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl_low += kl_low.item()
            epoch_kl_high += kl_high.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch:3d} [{batch_idx:3d}/{len(data_loader)}] "
                      f"Loss={loss.item()/bs:.4f} Beta={beta:.2f} Temp={temperature:.2f}")

        history['epoch'].append(epoch)
        history['total_loss'].append(epoch_loss / n_samples)
        history['recon_loss'].append(epoch_recon / n_samples)
        history['kl_low'].append(epoch_kl_low / n_samples)
        history['kl_high'].append(epoch_kl_high / n_samples)
        history['beta'].append(beta)
        history['temperature'].append(temperature)

        print(f"[Epoch {epoch}] total={history['total_loss'][-1]:.4f}, "
              f"recon={history['recon_loss'][-1]:.4f}, "
              f"kl_low={history['kl_low'][-1]:.4f}, "
              f"kl_high={history['kl_high'][-1]:.4f}")

    return history


def sample_diverse_patterns(model, n_styles=5, n_variations=10, device='cuda'):
    """
    Generate diverse drum patterns using the hierarchy.
    
    TODO:
    1. Sample n_styles from z_high prior
    2. For each style, sample n_variations from conditional p(z_low|z_high)
    3. Decode to patterns
    4. Organize in grid showing style consistency
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        z_high = torch.randn(n_styles, model.z_high_dim, device=device)
        all_patterns = []

        for i in range(n_styles):
            z_high_i = z_high[i].unsqueeze(0).repeat(n_variations, 1)
            logits = model.decode_hierarchy(z_high_i, z_low=None, temperature=0.9)
            probs = torch.sigmoid(logits)
            patterns = (probs > 0.5).float()
            all_patterns.append(patterns.cpu())

        return torch.stack(all_patterns, dim=0)  # [n_styles, n_variations, 16, 9]



def analyze_posterior_collapse(model, data_loader, device='cuda'):
    """
    Diagnose which latent dimensions are being used.
    
    TODO:
    1. Encode validation data
    2. Compute KL divergence per dimension
    3. Identify collapsed dimensions (KL ≈ 0)
    4. Return utilization statistics
    """
    model.eval()
    model.to(device)

    kl_low_sum = None
    kl_high_sum = None
    n = 0

    with torch.no_grad():
        for patterns, _, _ in data_loader:
            patterns = patterns.to(device)
            mu_low, logvar_low, z_low, mu_high, logvar_high, z_high = model.encode_hierarchy(patterns)

            kl_low_vec  = -0.5 * (1 + logvar_low - mu_low.pow(2) - logvar_low.exp())   # [B, D_low]
            kl_high_vec = -0.5 * (1 + logvar_high - mu_high.pow(2) - logvar_high.exp())# [B, D_high]

            if kl_low_sum is None:
                kl_low_sum  = kl_low_vec.sum(dim=0)
                kl_high_sum = kl_high_vec.sum(dim=0)
            else:
                kl_low_sum  += kl_low_vec.sum(dim=0)
                kl_high_sum += kl_high_vec.sum(dim=0)

            n += patterns.size(0)

    kl_low_mean  = (kl_low_sum  / n).cpu().numpy()
    kl_high_mean = (kl_high_sum / n).cpu().numpy()

    collapsed_low_idx  = np.where(kl_low_mean  < 1e-2)[0].tolist()
    collapsed_high_idx = np.where(kl_high_mean < 1e-2)[0].tolist()

    return {
        "kl_low_mean": kl_low_mean.tolist(),
        "kl_high_mean": kl_high_mean.tolist(),
        "collapsed_low_idx": collapsed_low_idx,
        "collapsed_high_idx": collapsed_high_idx
    }