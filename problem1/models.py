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
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 7->14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),   # 14->28
            nn.Tanh(),
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
        if self.conditional:
            if class_label is None:
                raise ValueError("conditional=True but class_label is None")
            z = torch.cat([z, class_label], dim=1)  # [B, z_dim + num_classes]

        x = self.project(z)                 # [B, 128*7*7]
        x = x.view(-1, 128, 7, 7)           # [B, 128, 7, 7]
        img = self.main(x)                  # [B, 1, 28, 28], range [0,1]
        return img



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

            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),   # 28->14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), # 14->7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0, bias=False), # 7->3
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

        )
        
        # Calculate feature dimension after convolutions
        feature_dim = 256 * 3 * 3  # Adjust based on your architecture
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + (num_classes if conditional else 0), 1),
            # nn.Sigmoid()
        )
    
    def forward(self, img, class_label=None):
        """
        Classify images as real (1) or fake (0).
        
        Returns:
            Probability of being real [batch_size, 1]
        """
        # TODO: Extract features, flatten, concatenate class if conditional
        img = img * 2 - 1

        x = self.features(img)
        x = torch.flatten(x, 1)

        if self.conditional:
            if class_label is None:
                raise ValueError("conditional=True but class_label is None")
            x = torch.cat([x, class_label], dim=1)

        out = self.classifier(x)           # [B, 1], probability of "real"
        return out