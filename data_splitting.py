"""
Data Splitting Module for Image Recognition

This module handles splitting the dataset into train, validation, and test sets
with stratification for reproducibility.
"""

from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
import numpy as np


class DataSplitter:
    """
    Handles splitting dataset into train, validation, and test sets with stratification.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the data splitter.
        
        Args:
            random_seed: Random seed for reproducibility (default: 42)
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def split_dataset(self, 
                     dataset: Any,
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1) -> Dict[str, Dict[str, List]]:
        """
        Split dataset into train, validation, and test sets with stratification.
        
        Args:
            dataset: Dataset object with 'train' split
            train_ratio: Ratio of data for training (default: 0.8)
            val_ratio: Ratio of data for validation (default: 0.1)
            test_ratio: Ratio of data for testing (default: 0.1)
            
        Returns:
            Dictionary containing 'train', 'val', and 'test' splits with indices and labels
        """
        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0")
        
        # Get the training data (we'll split it into train/val/test)
        full_dataset = dataset['train']
        labels = full_dataset['label']
        indices = list(range(len(full_dataset)))
        
        # First split: separate test set
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            stratify=labels,
            random_state=self.random_seed
        )
        
        # Get labels for train+val subset
        train_val_labels = [labels[i] for i in train_val_idx]
        
        # Second split: separate train and validation
        val_size = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            stratify=train_val_labels,
            random_state=self.random_seed
        )
        
        # Create split dictionary
        splits = {
            'train': {
                'indices': train_idx,
                'labels': [labels[i] for i in train_idx]
            },
            'val': {
                'indices': val_idx,
                'labels': [labels[i] for i in val_idx]
            },
            'test': {
                'indices': test_idx,
                'labels': [labels[i] for i in test_idx]
            }
        }
        
        return splits
    
    def display_split_info(self, splits: Dict[str, Dict[str, List]], 
                          label_names: List[str] = None) -> None:
        """
        Display information about the dataset splits.
        
        Args:
            splits: Dictionary containing split information
            label_names: Optional list of label names
        """
        print("\n" + "="*60)
        print("DATASET SPLIT INFORMATION")
        print("="*60)
        
        for split_name in ['train', 'val', 'test']:
            split_data = splits[split_name]
            labels = split_data['labels']
            
            print(f"\n{split_name.upper()} Split:")
            print(f"  Total samples: {len(labels)}")
            
            # Count samples per class
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"  Number of classes: {len(unique_labels)}")
            print(f"  Samples per class range: {counts.min()} - {counts.max()}")
            
            # Show distribution of top classes
            if label_names and len(label_names) > 0:
                top_classes = np.argsort(counts)[-5:][::-1]
                print(f"  Top 5 classes by count:")
                for idx in top_classes:
                    class_idx = unique_labels[idx]
                    class_name = label_names[class_idx] if class_idx < len(label_names) else f"Class {class_idx}"
                    print(f"    {class_name}: {counts[idx]} samples")
        
        print("\n" + "="*60)
    
    def verify_stratification(self, splits: Dict[str, Dict[str, List]], 
                             max_diff_threshold: float = 0.05) -> bool:
        """
        Verify that class distribution is similar across splits.
        
        Args:
            splits: Dictionary containing split information
            max_diff_threshold: Maximum allowed distribution difference (default: 0.05 = 5%)
            
        Returns:
            True if stratification is good, False otherwise
        """
        print("\nVerifying stratification...")
        
        # Get class distributions for each split
        distributions = {}
        for split_name in ['train', 'val', 'test']:
            labels = splits[split_name]['labels']
            unique, counts = np.unique(labels, return_counts=True)
            total = len(labels)
            distributions[split_name] = {label: count/total for label, count in zip(unique, counts)}
        
        # Compare distributions
        all_labels = set()
        for dist in distributions.values():
            all_labels.update(dist.keys())
        
        max_diff = 0
        for label in all_labels:
            train_ratio = distributions['train'].get(label, 0)
            val_ratio = distributions['val'].get(label, 0)
            test_ratio = distributions['test'].get(label, 0)
            
            diff = max(abs(train_ratio - val_ratio), 
                      abs(train_ratio - test_ratio),
                      abs(val_ratio - test_ratio))
            max_diff = max(max_diff, diff)
        
        print(f"Maximum distribution difference: {max_diff:.4f}")
        
        # Check against threshold
        is_good = max_diff < max_diff_threshold
        if is_good:
            print("✓ Stratification is good!")
        else:
            print(f"⚠ Stratification may need adjustment (threshold: {max_diff_threshold})")
        
        return is_good


def compute_class_weights(labels: List[int]) -> Dict[int, float]:
    """
    Compute class weights for handling class imbalance.
    
    Args:
        labels: List of labels
        
    Returns:
        Dictionary mapping class indices to weights
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    n_classes = len(unique_labels)
    
    # Compute weights inversely proportional to class frequency
    weights = {}
    for label, count in zip(unique_labels, counts):
        weights[int(label)] = total_samples / (n_classes * count)
    
    return weights


if __name__ == "__main__":
    # Example usage
    splitter = DataSplitter(random_seed=42)
    print("DataSplitter initialized successfully!")
    print(f"Random seed: {splitter.random_seed}")
