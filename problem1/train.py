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
from training_dynamics import train_gan, analyze_mode_coverage, visualize_mode_collapse
from fixes import train_gan_with_fix

from evaluate import interpolation_experiment, style_consistency_experiment, mode_recovery_experiment
from torchvision.utils import make_grid, save_image

def main():
    """
    Main training entry point for GAN experiments.
    """
    # Configuration
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'mps'),
        'batch_size': 64,
        'num_epochs': 10,
        'z_dim': 100,
        'learning_rate': 0.001,
        'data_dir': 'data/fonts',
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results',
        'experiment': 'vanilla',  # 'vanilla' or 'fixed'
        'fix_type': 'default'  # Used if experiment='fixed'
    }

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    config['data_dir'] = str((PROJECT_ROOT / config['data_dir']).resolve())

    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    (Path(config['results_dir']) / "visualizations").mkdir(parents=True, exist_ok=True)

    train_dataset = FontDataset(config['data_dir'], split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

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
        run_dir = Path(config['results_dir']) / 'vanilla'
    else:
        print(f"Training GAN with {config['fix_type']} fix...")
        history = train_gan_with_fix(
            generator,
            discriminator,
            train_loader,
            num_epochs=config['num_epochs'],
            fix_type=config['fix_type']
        )
        run_dir = Path(config['results_dir']) / config['fix_type']

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    # --- Save training history ---
    # defaultdict
    hist_json = {k: list(v) for k, v in history.items()}
    with open(run_dir / "training_log.json", 'w') as f:
        json.dump(hist_json, f, indent=2)


    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'config': config,
        'final_epoch': config['num_epochs']
    }, run_dir / "checkpoint_final.pth")
    torch.save(generator.state_dict(), run_dir / "best_generator.pth")  # 评测脚本常用 G-only

    # ---------------- EVALUATION & VISUALIZATIONS ----------------
    with torch.no_grad():
        z_fixed = torch.randn(100, config['z_dim'], device=config['device'])
        samples = generator(z_fixed).clamp(-1, 1)
        samples01 = (samples + 1) / 2.0
        grid = make_grid(samples01, nrow=10, padding=2)
        save_image(grid, run_dir / "visualizations" / "samples_grid.png")

    # 2)loss & mode coverage
    visualize_mode_collapse(history, save_path=str(run_dir / "mode_collapse_analysis.png"))

    # 3) latent plot
    interpolation_experiment(generator, device=config['device'])

    if getattr(generator, "conditional", False):
        style_consistency_experiment(generator, device=config['device'])

    mode_recovery_experiment([("final", generator)])

    print(f"Training complete. Results saved to {run_dir}/")

if __name__ == '__main__':
    main()
