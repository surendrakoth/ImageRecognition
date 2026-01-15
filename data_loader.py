"""
Data Loader Module for Image Recognition

This module handles loading the Food-101 dataset from Hugging Face
and provides utilities for accessing and displaying dataset information.
"""

from datasets import load_dataset
from typing import Dict, Any
import numpy as np


class DataLoader:
    """
    Handles loading and basic information display for the Food-101 dataset.
    """
    
    def __init__(self, dataset_name: str = "food101"):
        """
        Initialize the data loader.
        
        Args:
            dataset_name: Name of the dataset to load (default: food101)
        """
        self.dataset_name = dataset_name
        self.dataset = None
        
    def load_data(self) -> Dict[str, Any]:
        """
        Load the dataset from Hugging Face.
        
        Returns:
            Dictionary containing train and validation splits
        """
        print(f"Loading {self.dataset_name} dataset...")
        self.dataset = load_dataset(self.dataset_name)
        print(f"Dataset loaded successfully!")
        return self.dataset
    
    def display_info(self) -> None:
        """
        Display basic information about the loaded dataset.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        print("\n" + "="*60)
        print("DATASET INFORMATION")
        print("="*60)
        
        for split_name in self.dataset.keys():
            split = self.dataset[split_name]
            print(f"\n{split_name.upper()} Split:")
            print(f"  Number of samples: {len(split)}")
            print(f"  Features: {list(split.features.keys())}")
            
            # Display label information
            if 'label' in split.features:
                label_names = split.features['label'].names
                print(f"  Number of classes: {len(label_names)}")
                
                # Count samples per class
                labels = split['label']
                unique_labels, counts = np.unique(labels, return_counts=True)
                print(f"  Samples per class range: {counts.min()} - {counts.max()}")
        
        print("\n" + "="*60)
    
    def show_examples(self, split: str = "train", num_examples: int = 3) -> None:
        """
        Display example entries from the dataset.
        
        Args:
            split: Dataset split to show examples from
            num_examples: Number of examples to display
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        print(f"\n{num_examples} Example(s) from {split} split:")
        print("-" * 60)
        
        for i in range(min(num_examples, len(self.dataset[split]))):
            example = self.dataset[split][i]
            print(f"\nExample {i+1}:")
            print(f"  Label: {example['label']}")
            
            if 'image' in example:
                img = example['image']
                print(f"  Image size: {img.size}")
                print(f"  Image mode: {img.mode}")
        
        print("-" * 60)
    
    def get_dataset(self) -> Dict[str, Any]:
        """
        Get the loaded dataset.
        
        Returns:
            The loaded dataset dictionary
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        return self.dataset


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    dataset = loader.load_data()
    loader.display_info()
    loader.show_examples()
