"""
Provided evaluation metrics for HW2.
These functions are provided to ensure consistent evaluation across submissions.
"""

import torch
import numpy as np
from collections import Counter

def mode_coverage_score(generated_samples, classifier_fn=None):
    """
    Measure how many different letter modes the GAN generates.
    
    Args:
        generated_samples: Tensor of generated images [N, 1, 28, 28]
        classifier_fn: Function to classify letters (if None, use simple heuristic)
    
    Returns:
        Dict with:
            - coverage_score: Number of unique letters / 26
            - letter_counts: Dict of letter -> count
            - missing_letters: List of letters not generated
    """
    if classifier_fn is None:
        # Simple heuristic classifier based on image statistics
        # In practice, students would use a pre-trained classifier
        classifier_fn = _simple_letter_classifier
    
    # Classify all samples
    predictions = []
    for i in range(generated_samples.shape[0]):
        letter = classifier_fn(generated_samples[i])
        predictions.append(letter)
    
    # Count unique letters
    letter_counts = Counter(predictions)
    unique_letters = set(letter_counts.keys())
    all_letters = set(range(26))  # 0-25 for A-Z
    missing_letters = sorted(all_letters - unique_letters)
    
    coverage_score = len(unique_letters) / 26.0
    
    return {
        'coverage_score': coverage_score,
        'letter_counts': dict(letter_counts),
        'missing_letters': missing_letters,
        'n_unique': len(unique_letters)
    }

def _simple_letter_classifier(image):
    """
    Simple heuristic classifier for letters.
    This is a placeholder - in practice use a trained model.
    """
    # Extract simple features
    img = image.squeeze().cpu().numpy()
    
    # Use image statistics as features
    mean_val = img.mean()
    std_val = img.std()
    center_mass = img[10:18, 10:18].sum()
    
    # Simple hash to letter (this is just for demonstration)
    hash_val = int(mean_val * 100 + std_val * 10 + center_mass) % 26
    
    return hash_val

def font_consistency_score(generated_samples, n_samples=10):
    """
    Measure if generated letters maintain consistent style.
    
    Args:
        generated_samples: Dict of letter -> List of generated images
        n_samples: Number of samples per letter to compare
    
    Returns:
        consistency_score: Float between 0 and 1
    """
    if len(generated_samples) < 2:
        return 0.0
    
    consistency_scores = []
    
    for letter, images in generated_samples.items():
        if len(images) < 2:
            continue
        
        # Compare pairs of images
        for i in range(min(n_samples, len(images) - 1)):
            for j in range(i + 1, min(n_samples, len(images))):
                # Compute similarity (simplified - use MSE)
                diff = (images[i] - images[j]).pow(2).mean()
                similarity = torch.exp(-diff)  # Convert to similarity
                consistency_scores.append(similarity.item())
    
    if not consistency_scores:
        return 0.0
    
    return np.mean(consistency_scores)

def drum_pattern_validity(pattern):
    """
    Check if a drum pattern is musically valid.
    
    Args:
        pattern: Binary tensor [16, 9] or [batch, 16, 9]
    
    Returns:
        validity_score: Float between 0 and 1
    """
    if pattern.dim() == 3:
        # Batch mode
        scores = []
        for i in range(pattern.shape[0]):
            scores.append(drum_pattern_validity(pattern[i]))
        return np.mean(scores)
    
    pattern = pattern.cpu().numpy()
    
    # Check basic musical constraints
    score = 1.0
    
    # 1. Pattern should not be empty
    if pattern.sum() == 0:
        return 0.0
    
    # 2. Pattern should not be too dense (> 50% filled)
    density = pattern.sum() / (16 * 9)
    if density > 0.5:
        score *= 0.8
    
    # 3. Should have some kick drum (instrument 0)
    if pattern[:, 0].sum() == 0:
        score *= 0.7
    
    # 4. Should have some rhythmic structure (not random)
    # Check for repeating patterns
    first_half = pattern[:8]
    second_half = pattern[8:]
    similarity = np.sum(first_half == second_half) / (8 * 9)
    
    if similarity < 0.3:  # Too random
        score *= 0.8
    
    return score

def sequence_diversity(patterns):
    """
    Measure diversity of generated drum patterns.
    
    Args:
        patterns: Tensor of patterns [N, 16, 9]
    
    Returns:
        diversity_score: Float between 0 and 1
    """
    n = patterns.shape[0]
    if n < 2:
        return 0.0
    
    patterns_flat = patterns.view(n, -1)
    
    # Compute pairwise distances
    distances = []
    for i in range(min(100, n)):  # Sample for efficiency
        for j in range(i + 1, min(100, n)):
            dist = (patterns_flat[i] != patterns_flat[j]).float().mean()
            distances.append(dist.item())
    
    # Average distance is our diversity metric
    return np.mean(distances) if distances else 0.0