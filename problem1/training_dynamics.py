"""
GAN training implementation with mode collapse analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from provided.metrics import mode_coverage_score


def train_gan(generator, discriminator, data_loader, num_epochs=100, device='cuda'):
    """
    Standard GAN training implementation.

    Uses vanilla GAN objective which typically exhibits mode collapse.

    Args:
        generator: Generator network
        discriminator: Discriminator network
        data_loader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device for computation

    Returns:
        dict: Training history and metrics
    """
    device = torch.device(device)
    generator = generator.to(device).train()
    discriminator = discriminator.to(device).train()

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    # Training history
    history = defaultdict(list)
    z_dim = getattr(generator, 'z_dim', 100)

    for epoch in range(num_epochs):
        for batch_idx, (real_images, _labels) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)  # [B,1,28,28] in [0,1]

            # Labels for loss computation
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ========== Train Discriminator ==========
            d_optimizer.zero_grad(set_to_none=True)

            # 1) Real
            logits_real = discriminator(real_images)                  # logits
            prob_real = torch.sigmoid(logits_real)                    # [0,1]
            d_loss_real = criterion(prob_real, real_labels)

            # 2) Fake (detach)
            z = torch.randn(batch_size, z_dim, device=device)
            with torch.no_grad():
                fake_images_m11 = generator(z)                        # [-1,1]
            logits_fake = discriminator((fake_images_m11 + 1) / 2.0)
            prob_fake = torch.sigmoid(logits_fake)
            d_loss_fake = criterion(prob_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # ========== Train Generator ==========
            g_optimizer.zero_grad(set_to_none=True)
            z = torch.randn(batch_size, z_dim, device=device)
            fake_images_m11 = generator(z)                            # [-1,1]
            logits_fake_for_g = discriminator((fake_images_m11 + 1) / 2.0)
            prob_fake_for_g = torch.sigmoid(logits_fake_for_g)
            g_loss = criterion(prob_fake_for_g, real_labels)
            g_loss.backward()
            g_optimizer.step()

            # Log metrics
            if batch_idx % 10 == 0:
                history['d_loss'].append(float(d_loss.item()))
                history['g_loss'].append(float(g_loss.item()))
                history['epoch'].append(epoch + batch_idx/len(data_loader))

        # Analyze mode collapse every 10 epochs
        if epoch % 10 == 0:
            generator.eval()
            with torch.no_grad():
                mode_coverage = analyze_mode_coverage(generator, device)
            generator.train()
            history['mode_coverage'].append((epoch, float(mode_coverage)))
            print(f"Epoch {epoch}: Mode coverage = {mode_coverage:.2f}")

    return history


def analyze_mode_coverage(generator, device, n_samples=1000):
    """
    Measure mode coverage by counting unique letters in generated samples.

    Args:
        generator: Trained generator network
        device: Device for computation
        n_samples: Number of samples to generate

    Returns:
        float: Coverage score (unique letters / 26)
    """
    device = torch.device(device)
    z_dim = getattr(generator, 'z_dim', 100)
    generator = generator.to(device).eval()
    with torch.no_grad():
        z = torch.randn(n_samples, z_dim, device=device)
        imgs_m11 = generator(z).clamp(-1, 1)        # [-1,1]
        imgs_01 = (imgs_m11 + 1) / 2.0              # → [0,1]，metrics.py

        result = mode_coverage_score(imgs_01.cpu(), classifier_fn=None)
        return float(result['coverage_score'])


def visualize_mode_collapse(history, save_path):
    """
    Visualize mode collapse progression over training.

    Args:
        history: Training metrics dictionary
        save_path: Output path for visualization
    """
    epochs = history.get('epoch', [])
    d_loss = history.get('d_loss', [])
    g_loss = history.get('g_loss', [])
    cov_pairs = history.get('mode_coverage', [])

    plt.figure(figsize=(10, 5))


    plt.subplot(1, 2, 1)
    if len(epochs) and len(d_loss) and len(g_loss):
        plt.plot(epochs, d_loss, label='D loss')
        plt.plot(epochs, g_loss, label='G loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()


    plt.subplot(1, 2, 2)
    if cov_pairs:
        xs = [e for e, _ in cov_pairs]
        ys = [c for _, c in cov_pairs]
        plt.plot(xs, ys, marker='o')
        plt.ylim(0, 1.05)
        plt.xlabel('Epoch (every 10)')
        plt.ylabel('Mode Coverage (0..1)')
        plt.title('Mode Coverage over Time')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
