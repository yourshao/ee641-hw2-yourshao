"""
GAN models for font generation.
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, conditional=False, num_classes=26):
        """
        Generator network that produces 28×28 letter images.
        
        Args:
            z_dim: Dimension of latent vector z
            conditional: If True, condition on letter class
            num_classes: Number of letter classes (26)
        """
        super().__init__()
        self.z_dim = z_dim
        self.conditional = conditional
        
        # Calculate input dimension
        input_dim = z_dim + (num_classes if conditional else 0)
        
        # Architecture proven to work well for this task:
        # Project and reshape: z → 7×7×128
        self.project = nn.Sequential(
            nn.Linear(input_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True)
        )
        
        # Upsample: 7×7×128 → 14×14×64 → 28×28×1
        self.main = nn.Sequential(
            # TODO: Implement upsampling layers
            # Use ConvTranspose2d with appropriate padding/stride
            # Include BatchNorm2d and ReLU (except final layer)
            # Final layer should use Tanh activation
        )
    
    def forward(self, z, class_label=None):
        """
        Generate images from latent code.
        
        Args:
            z: Latent vectors [batch_size, z_dim]
            class_label: One-hot encoded class labels [batch_size, num_classes]
        
        Returns:
            Generated images [batch_size, 1, 28, 28] in range [-1, 1]
        """
        # TODO: Implement forward pass
        # If conditional, concatenate z and class_label
        # Project to spatial dimensions
        # Apply upsampling network
        pass

class Discriminator(nn.Module):
    def __init__(self, conditional=False, num_classes=26):
        """
        Discriminator network that classifies 28×28 images as real/fake.
        """
        super().__init__()
        self.conditional = conditional
        
        # Proven architecture for 28×28 images:
        self.features = nn.Sequential(
            # TODO: Implement convolutional layers
            # 28×28×1 → 14×14×64 → 7×7×128 → 3×3×256
            # Use Conv2d with appropriate stride
            # LeakyReLU(0.2) and Dropout2d(0.25)
        )
        
        # Calculate feature dimension after convolutions
        feature_dim = 256 * 3 * 3  # Adjust based on your architecture
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + (num_classes if conditional else 0), 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, class_label=None):
        """
        Classify images as real (1) or fake (0).
        
        Returns:
            Probability of being real [batch_size, 1]
        """
        # TODO: Extract features, flatten, concatenate class if conditional
        pass