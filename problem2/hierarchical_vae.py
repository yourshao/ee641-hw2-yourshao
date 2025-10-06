"""
Hierarchical VAE for drum pattern generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalDrumVAE(nn.Module):
    def __init__(self, z_high_dim=4, z_low_dim=12):
        """
        Two-level VAE for drum patterns.
        
        The architecture uses a hierarchy of latent variables where z_high
        encodes style/genre information and z_low encodes pattern variations.
        
        Args:
            z_high_dim: Dimension of high-level latent (style)
            z_low_dim: Dimension of low-level latent (variation)
        """
        super().__init__()
        self.z_high_dim = z_high_dim
        self.z_low_dim = z_low_dim
        
        self.encoder_low = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=3, padding=1),  # [16, 9] → [16, 32]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # → [8, 64]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # → [4, 128]
            nn.ReLU(),
            nn.Flatten()  # → [512]
        )
        
        # Low-level latent parameters
        self.fc_mu_low = nn.Linear(512, z_low_dim)
        self.fc_logvar_low = nn.Linear(512, z_low_dim)
        
        # Encoder from z_low to z_high
        self.encoder_high = nn.Sequential(
            nn.Linear(z_low_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # High-level latent parameters
        self.fc_mu_high = nn.Linear(32, z_high_dim)
        self.fc_logvar_high = nn.Linear(32, z_high_dim)
        
        # Decoder: z_high → z_low → pattern
        # TODO: Implement decoder architecture
        # Mirror the encoder structure
        # Use transposed convolutions for upsampling

        self.dec_fc = nn.Linear(z_high_dim + z_low_dim, 128 * 4)  # [B, 512]等价
        self.dec_deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # len 4->8
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # len 8->16
            nn.ReLU(),
            nn.Conv1d(32, 9, kernel_size=3, padding=1)  # 输出 logits，通道=乐器数
        )


    def encode_hierarchy(self, x):
        """
        Encode pattern to both latent levels.
        
        Args:
            x: Drum patterns [batch_size, 16, 9]
            
        Returns:
            mu_low, logvar_low: Parameters for q(z_low|x)
            mu_high, logvar_high: Parameters for q(z_high|z_low)
        """
        # Reshape for Conv1d: [batch, 16, 9] → [batch, 9, 16]
        x = x.transpose(1, 2).float()
        
        # TODO: Encode to z_low parameters
        # TODO: Sample z_low using reparameterization
        # TODO: Encode z_low to z_high parameters
        h = self.encoder_low(x)        # [B, 512]

        mu_low = self.fc_mu_low(h)
        logvar_low = self.fc_logvar_low(h)
        z_low = self.reparameterize(mu_low, logvar_low)

        h_high = self.encoder_high(z_low)         # [B, 32]
        mu_high = self.fc_mu_high(h_high)
        logvar_high = self.fc_logvar_high(h_high)
        z_high = self.reparameterize(mu_high, logvar_high)

        return mu_low, logvar_low, z_low, mu_high, logvar_high, z_high


    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling.
        
        TODO: Implement
        z = mu + eps * std where eps ~ N(0,1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode_hierarchy(self, z_high, z_low=None, temperature=1.0):
        """
        Decode from latent variables to pattern.
        
        Args:
            z_high: High-level latent code
            z_low: Low-level latent code (if None, sample from prior)
            temperature: Temperature for binary output (lower = sharper)
            
        Returns:
            pattern_logits: Logits for binary pattern [batch, 16, 9]
        """
        # TODO: If z_low is None, sample from conditional prior p(z_low|z_high)
        # TODO: Decode z_high and z_low to pattern logits
        # TODO: Apply temperature scaling before sigmoid
        if z_low is None:
            z_low = torch.randn(z_high.size(0), self.z_low_dim, device=z_high.device)

        z = torch.cat([z_high, z_low], dim=-1)  # [B, z_high+z_low]
        d = self.dec_fc(z)                       # [B, 512]
        d = d.view(d.size(0), 128, 4)            # [B, 128, 4]
        logits = self.dec_deconv(d)              # [B, 9, 16]
        logits = logits.transpose(1, 2)          # -> [B, 16, 9]

        if temperature is not None and temperature > 0:
            logits = logits / float(temperature)
        return logits
    
    def forward(self, x, beta=1.0):
        """
        Full forward pass with loss computation.
        
        Args:
            x: Input patterns [batch_size, 16, 9]
            beta: KL weight for beta-VAE (use < 1 to prevent collapse)
            
        Returns:
            recon: Reconstructed patterns
            mu_low, logvar_low, mu_high, logvar_high: Latent parameters
        """
        # TODO: Encode, decode, compute losses
        mu_low, logvar_low, z_low, mu_high, logvar_high, z_high = self.encode_hierarchy(x)
        recon_logits = self.decode_hierarchy(z_high, z_low=z_low, temperature=1.0)
        return recon_logits, mu_low, logvar_low, mu_high, logvar_high