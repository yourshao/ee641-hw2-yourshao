"""
Dataset loader for font generation task.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np

class FontDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Initialize the font dataset.
        
        Args:
            data_dir: Path to font dataset directory
            split: 'train' or 'val'
        """
        self.data_dir = data_dir
        self.split = split
        
        # TODO: Load metadata from fonts_metadata.json
        # Expected structure:
        # {
        #   "train": [{"path": "A/font1_A.png", "letter": "A", "font": 1}, ...],
        #   "val": [...]
        # }
        metadata_path = os.path.join(data_dir, 'fonts_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.samples = metadata[split]
        self.letter_to_id = {chr(65+i): i for i in range(26)}  # A=0, B=1, ..., Z=25
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Returns:
            image: Tensor of shape [1, 28, 28]
            letter_id: Integer 0-25 representing A-Z
        """
        sample = self.samples[idx]
        
        # TODO: Load and process image
        # 1. Load image from sample['path']
        # 2. Convert to grayscale if needed
        # 3. Resize to 28x28 if needed
        # 4. Normalize to [0, 1]
        # 5. Convert to tensor
        
        image_path = os.path.join(self.data_dir, sample['path'])
        image = Image.open(image_path).convert('L')  # Ensure grayscale
        image = image.resize((28, 28), Image.LANCZOS)
        
        # Convert to numpy and normalize
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor and add channel dimension
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        # Get letter ID
        letter_id = self.letter_to_id[sample['letter']]
        
        return image_tensor, letter_id