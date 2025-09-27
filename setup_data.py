#!/usr/bin/env python3
"""
Setup script to generate datasets for EE641 HW2.
This script generates synthetic font and drum pattern datasets.

Usage:
    python setup_data.py --seed 641
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import warnings
warnings.filterwarnings('ignore')

def generate_font_dataset(output_dir, seed=641):
    """
    Generate synthetic font dataset with letters A-Z in multiple styles.
    
    Creates 28x28 grayscale images of letters in different synthetic fonts.
    """
    np.random.seed(seed)
    
    print("Generating font dataset...")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'fonts', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'fonts', 'val'), exist_ok=True)
    
    # Font styles (simulated with different transformations)
    font_styles = [
        'regular', 'bold', 'italic', 'narrow', 'wide',
        'serif', 'sans', 'mono', 'script', 'display'
    ]
    
    # Alphabet
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    metadata = {
        'letters': list(letters),
        'font_styles': font_styles,
        'image_size': 28,
        'train_samples': [],
        'val_samples': []
    }
    
    # Generate images
    for split in ['train', 'val']:
        n_samples_per_letter = 200 if split == 'train' else 60
        
        for letter_idx, letter in enumerate(letters):
            for sample_idx in range(n_samples_per_letter):
                # Random font style
                font_style = np.random.choice(font_styles)
                
                # Create 28x28 image
                img = Image.new('L', (28, 28), color=255)  # White background
                draw = ImageDraw.Draw(img)
                
                # Simulate different font styles with variations
                size_variation = np.random.uniform(0.8, 1.2)
                x_offset = np.random.uniform(-2, 2)
                y_offset = np.random.uniform(-2, 2)
                
                # Base size and position
                font_size = int(20 * size_variation)
                x_pos = 14 + x_offset
                y_pos = 14 + y_offset
                
                # Simulate font weight and style
                thickness = 1 if 'regular' in font_style else 2 if 'bold' in font_style else 1
                
                # Simple letter rendering (using PIL's basic font)
                # In practice, you'd use actual font files
                try:
                    # Use default font and simulate styles
                    draw.text((x_pos - font_size//2, y_pos - font_size//2), 
                             letter, fill=0)  # Black text
                    
                    # Add style-specific transformations
                    if 'italic' in font_style:
                        # Simple shear transformation
                        img = img.transform((28, 28), Image.AFFINE, 
                                          (1, 0.2, 0, 0, 1, 0))
                    
                    if 'narrow' in font_style:
                        # Horizontal squeeze
                        img = img.resize((20, 28))
                        new_img = Image.new('L', (28, 28), color=255)
                        new_img.paste(img, (4, 0))
                        img = new_img
                    
                    if 'wide' in font_style:
                        # Horizontal stretch
                        center_crop = img.crop((4, 0, 24, 28))
                        img = center_crop.resize((28, 28))
                    
                except:
                    # Fallback: create simple block letter
                    draw.rectangle([x_pos-5, y_pos-5, x_pos+5, y_pos+5], fill=0)
                    draw.text((x_pos-3, y_pos-3), letter, fill=255)
                
                # Add slight noise
                img_array = np.array(img)
                noise = np.random.normal(0, 5, img_array.shape)
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                
                # Save image
                filename = f'{letter}_{font_style}_{sample_idx:04d}.png'
                filepath = os.path.join(output_dir, 'fonts', split, filename)
                img.save(filepath)
                
                # Record metadata
                sample_info = {
                    'filename': filename,
                    'letter': letter,
                    'letter_idx': letter_idx,
                    'font_style': font_style,
                }
                
                if split == 'train':
                    metadata['train_samples'].append(sample_info)
                else:
                    metadata['val_samples'].append(sample_info)
    
    # Save metadata
    with open(os.path.join(output_dir, 'fonts', 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated {len(metadata['train_samples'])} training and "
          f"{len(metadata['val_samples'])} validation font images")

def generate_drum_dataset(output_dir, seed=641):
    """
    Generate synthetic drum pattern dataset.
    
    Creates 16x9 binary matrices representing drum patterns.
    16 timesteps, 9 instruments.
    """
    np.random.seed(seed)
    
    print("Generating drum pattern dataset...")
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, 'drums'), exist_ok=True)
    
    # Drum instruments
    instruments = [
        'kick', 'snare', 'hihat_closed', 'hihat_open',
        'tom_low', 'tom_high', 'crash', 'ride', 'clap'
    ]
    
    # Music styles
    styles = ['rock', 'jazz', 'hiphop', 'electronic', 'latin']
    
    # Generate patterns for each style
    patterns = []
    
    for style_idx, style in enumerate(styles):
        n_patterns = 200
        
        for pattern_idx in range(n_patterns):
            # Create 16x9 binary pattern
            pattern = np.zeros((16, 9), dtype=np.uint8)
            
            # Style-specific pattern generation
            if style == 'rock':
                # Basic rock beat
                pattern[::4, 0] = 1  # Kick on 1 and 3
                pattern[2::4, 0] = np.random.choice([0, 1], size=len(pattern[2::4, 0]), p=[0.3, 0.7])
                pattern[4::4, 1] = 1  # Snare on 2 and 4
                pattern[::2, 2] = 1  # Hi-hat on eighth notes
                pattern[15, 6] = np.random.choice([0, 1], p=[0.7, 0.3])  # Occasional crash
                
            elif style == 'jazz':
                # Swing pattern
                pattern[0, 0] = 1  # Kick on 1
                pattern[10, 0] = np.random.choice([0, 1], p=[0.4, 0.6])
                pattern[4::8, 1] = 1  # Snare
                # Swing hi-hat pattern
                for i in range(0, 16, 3):
                    pattern[i, 7] = 1  # Ride cymbal
                
            elif style == 'hiphop':
                # Hip-hop beat
                pattern[0, 0] = 1  # Kick
                pattern[10, 0] = 1
                pattern[4, 1] = 1  # Snare
                pattern[12, 1] = 1
                pattern[::2, 2] = np.random.choice([0, 1], size=8, p=[0.2, 0.8])  # Hi-hat
                pattern[7, 8] = np.random.choice([0, 1], p=[0.6, 0.4])  # Clap
                
            elif style == 'electronic':
                # Four-on-the-floor
                pattern[::4, 0] = 1  # Kick on every beat
                pattern[4::8, 1] = 1  # Snare
                pattern[2::4, 2] = 1  # Off-beat hi-hat
                pattern[::8, 8] = np.random.choice([0, 1], size=2, p=[0.3, 0.7])  # Clap
                
            elif style == 'latin':
                # Clave-inspired pattern
                pattern[[0, 3, 6, 10, 12], 0] = 1  # Clave-like kick
                pattern[[4, 12], 1] = 1  # Snare
                pattern[::2, 2] = 1  # Constant hi-hat
                pattern[[2, 5, 8, 11, 14], 4] = np.random.choice([0, 1], size=5, p=[0.4, 0.6])  # Toms
            
            # Add random variations
            noise_mask = np.random.random((16, 9)) < 0.05  # 5% random flips
            pattern = np.logical_xor(pattern, noise_mask).astype(np.uint8)
            
            # Ensure pattern isn't empty
            if pattern.sum() == 0:
                pattern[0, 0] = 1  # At least one kick
            
            patterns.append({
                'pattern': pattern.tolist(),
                'style': style,
                'style_idx': style_idx,
                'density': float(pattern.sum()) / (16 * 9),
                'pattern_idx': len(patterns)
            })
    
    # Split into train/val
    np.random.shuffle(patterns)
    n_train = int(0.8 * len(patterns))
    
    dataset = {
        'instruments': instruments,
        'styles': styles,
        'timesteps': 16,
        'train_patterns': patterns[:n_train],
        'val_patterns': patterns[n_train:]
    }
    
    # Save as JSON for easy loading
    with open(os.path.join(output_dir, 'drums', 'patterns.json'), 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Also save as numpy arrays for faster loading
    train_array = np.array([p['pattern'] for p in dataset['train_patterns']])
    val_array = np.array([p['pattern'] for p in dataset['val_patterns']])
    
    np.savez(os.path.join(output_dir, 'drums', 'patterns.npz'),
             train_patterns=train_array,
             val_patterns=val_array,
             train_styles=[p['style_idx'] for p in dataset['train_patterns']],
             val_styles=[p['style_idx'] for p in dataset['val_patterns']])
    
    print(f"Generated {len(dataset['train_patterns'])} training and "
          f"{len(dataset['val_patterns'])} validation drum patterns")

def main():
    parser = argparse.ArgumentParser(description='Generate datasets for EE641 HW2')
    parser.add_argument('--seed', type=int, default=641,
                        help='Random seed for reproducibility (default: 641)')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for datasets (default: data)')
    
    args = parser.parse_args()
    
    print(f"Generating datasets with seed {args.seed}...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate both datasets
    generate_font_dataset(args.output_dir, args.seed)
    generate_drum_dataset(args.output_dir, args.seed)
    
    print("\nDataset generation complete.")
    print(f"Output location: {os.path.abspath(args.output_dir)}/")
    print("  - fonts/: Letter images for GAN problem")
    print("  - drums/: Drum patterns for VAE problem")

if __name__ == '__main__':
    main()