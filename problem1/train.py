"""
Main training script for GAN experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from pathlib import Path

from dataset import FontDataset
from models import Generator, Discriminator
from training_dynamics import train_gan, analyze_mode_coverage
from fixes import train_gan_with_fix

def main():
    """
    Main training entry point for GAN experiments.
    """
    # Configuration
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 64,
        'num_epochs': 100,
        'z_dim': 100,
        'learning_rate': 0.0002,
        'data_dir': 'data/fonts',
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results',
        'experiment': 'vanilla',  # 'vanilla' or 'fixed'
        'fix_type': 'feature_matching'  # Used if experiment='fixed'
    }
    
    # Create directories
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset and dataloader
    train_dataset = FontDataset(config['data_dir'], split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    # Initialize models
    generator = Generator(z_dim=config['z_dim']).to(config['device'])
    discriminator = Discriminator().to(config['device'])
    
    # Train model
    if config['experiment'] == 'vanilla':
        print("Training vanilla GAN (expect mode collapse)...")
        history = train_gan(
            generator, 
            discriminator, 
            train_loader,
            num_epochs=config['num_epochs'],
            device=config['device']
        )
    else:
        print(f"Training GAN with {config['fix_type']} fix...")
        history = train_gan_with_fix(
            generator,
            discriminator, 
            train_loader,
            num_epochs=config['num_epochs'],
            fix_type=config['fix_type']
        )
    
    # Save results
    # TODO: Save training history to JSON
    with open(f"{config['results_dir']}/training_log.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # TODO: Save final model checkpoint
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'config': config,
        'final_epoch': config['num_epochs']
    }, f"{config['results_dir']}/best_generator.pth")
    
    print(f"Training complete. Results saved to {config['results_dir']}/")

if __name__ == '__main__':
    main()