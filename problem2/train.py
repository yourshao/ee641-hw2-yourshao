"""
Main training script for hierarchical VAE experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from pathlib import Path
import numpy as np

from dataset import DrumPatternDataset
from hierarchical_vae import HierarchicalDrumVAE
from training_utils import kl_annealing_schedule, temperature_annealing_schedule

def compute_hierarchical_elbo(recon_x, x, mu_low, logvar_low, mu_high, logvar_high, beta=1.0):
    """
    Compute Evidence Lower Bound (ELBO) for hierarchical VAE.
    
    ELBO = E[log p(x|z_low)] - beta * KL(q(z_low|x) || p(z_low|z_high)) 
           - beta * KL(q(z_high|z_low) || p(z_high))
    
    Args:
        recon_x: Reconstructed pattern logits [batch, 16, 9]
        x: Original patterns [batch, 16, 9]
        mu_low, logvar_low: Low-level latent parameters
        mu_high, logvar_high: High-level latent parameters
        beta: KL weight for beta-VAE
        
    Returns:
        loss: Total loss
        recon_loss: Reconstruction component
        kl_low: KL divergence for low-level latent
        kl_high: KL divergence for high-level latent
    """
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy_with_logits(
        recon_x.view(-1), x.view(-1), reduction='sum'
    )
    
    # KL divergence for high-level latent: KL(q(z_high) || p(z_high))
    # where p(z_high) = N(0, I)
    kl_high = -0.5 * torch.sum(1 + logvar_high - mu_high.pow(2) - logvar_high.exp())
    
    # KL divergence for low-level latent: KL(q(z_low) || p(z_low|z_high))
    # For simplicity, we use standard KL with N(0, I) prior
    # In practice, you might want to implement conditional prior p(z_low|z_high)
    kl_low = -0.5 * torch.sum(1 + logvar_low - mu_low.pow(2) - logvar_low.exp())
    
    # Total loss
    total_loss = recon_loss + beta * (kl_low + kl_high)
    
    return total_loss, recon_loss, kl_low, kl_high

def train_epoch(model, data_loader, optimizer, epoch, device, config):
    """
    Train model for one epoch with annealing schedules.
    
    Returns:
        Dictionary of average metrics for the epoch
    """
    model.train()
    
    # Metrics tracking
    metrics = {
        'total_loss': 0,
        'recon_loss': 0,
        'kl_low': 0,
        'kl_high': 0
    }
    
    # Get annealing parameters for this epoch
    beta = kl_annealing_schedule(epoch, method=config['kl_anneal_method'])
    temperature = temperature_annealing_schedule(epoch)
    
    for batch_idx, (patterns, styles, densities) in enumerate(data_loader):
        patterns = patterns.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon, mu_low, logvar_low, mu_high, logvar_high = model(patterns, beta=beta)
        
        # Compute loss
        loss, recon_loss, kl_low, kl_high = compute_hierarchical_elbo(
            recon, patterns, mu_low, logvar_low, mu_high, logvar_high, beta
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        metrics['total_loss'] += loss.item()
        metrics['recon_loss'] += recon_loss.item()
        metrics['kl_low'] += kl_low.item()
        metrics['kl_high'] += kl_high.item()
        
        # Log progress
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch:3d} [{batch_idx:3d}/{len(data_loader)}] '
                  f'Loss: {loss.item()/len(patterns):.4f} '
                  f'Beta: {beta:.3f} Temp: {temperature:.2f}')
    
    # Average metrics
    n_samples = len(data_loader.dataset)
    for key in metrics:
        metrics[key] /= n_samples
    
    return metrics

def main():
    """
    Main training entry point for hierarchical VAE experiments.
    """
    # Configuration
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'z_high_dim': 4,
        'z_low_dim': 12,
        'kl_anneal_method': 'cyclical',  # 'linear', 'cyclical', or 'sigmoid'
        'data_dir': 'data/drums',
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results'
    }
    
    # Create directories
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset and dataloader
    train_dataset = DrumPatternDataset(config['data_dir'], split='train')
    val_dataset = DrumPatternDataset(config['data_dir'], split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model and optimizer
    model = HierarchicalDrumVAE(
        z_high_dim=config['z_high_dim'],
        z_low_dim=config['z_low_dim']
    ).to(config['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training history
    history = {
        'train': [],
        'val': [],
        'config': config
    }
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, epoch, 
            config['device'], config
        )
        history['train'].append(train_metrics)
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            val_metrics = {
                'total_loss': 0,
                'recon_loss': 0,
                'kl_low': 0,
                'kl_high': 0
            }
            
            with torch.no_grad():
                for patterns, styles, densities in val_loader:
                    patterns = patterns.to(config['device'])
                    recon, mu_low, logvar_low, mu_high, logvar_high = model(patterns)
                    loss, recon_loss, kl_low, kl_high = compute_hierarchical_elbo(
                        recon, patterns, mu_low, logvar_low, mu_high, logvar_high
                    )
                    
                    val_metrics['total_loss'] += loss.item()
                    val_metrics['recon_loss'] += recon_loss.item()
                    val_metrics['kl_low'] += kl_low.item()
                    val_metrics['kl_high'] += kl_high.item()
            
            # Average validation metrics
            n_val = len(val_dataset)
            for key in val_metrics:
                val_metrics[key] /= n_val
            
            history['val'].append(val_metrics)
            
            print(f"Epoch {epoch:3d} Validation - "
                  f"Loss: {val_metrics['total_loss']:.4f} "
                  f"KL_high: {val_metrics['kl_high']:.4f} "
                  f"KL_low: {val_metrics['kl_low']:.4f}")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch+1}.pth")
    
    # Save final model and history
    torch.save(model.state_dict(), f"{config['results_dir']}/best_model.pth")
    
    with open(f"{config['results_dir']}/training_log.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training complete. Results saved to {config['results_dir']}/")

if __name__ == '__main__':
    main()