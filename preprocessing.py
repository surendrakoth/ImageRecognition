"""
Data Preprocessing Module for Image Recognition

This module handles cleaning, normalization, and transformation of images
for the image recognition pipeline.
"""

from PIL import Image
import numpy as np
from typing import Tuple, Optional, List
import torch
from torchvision import transforms


class ImagePreprocessor:
    """
    Handles image preprocessing including cleaning, normalization, and augmentation.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target size for resizing images (height, width)
        """
        self.target_size = target_size
        
        # Define normalization parameters (ImageNet stats)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Define basic preprocessing transform
        self.basic_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Define augmentation transform for training
        self.augmentation_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.target_size[0]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def convert_to_rgb(self, image: Image.Image) -> Image.Image:
        """
        Ensure image is in RGB format.
        
        Args:
            image: PIL Image
            
        Returns:
            RGB PIL Image
        """
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image
    
    def is_corrupted(self, image: Image.Image) -> bool:
        """
        Check if an image is corrupted or invalid.
        
        Args:
            image: PIL Image to check
            
        Returns:
            True if image is corrupted, False otherwise
        """
        try:
            # Try to load the image data
            image.load()
            # Check if image has valid size
            if image.size[0] == 0 or image.size[1] == 0:
                return True
            return False
        except Exception:
            return True
    
    def clean_image(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Clean and validate an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Cleaned RGB image or None if corrupted
        """
        # Check if corrupted
        if self.is_corrupted(image):
            return None
        
        # Convert to RGB
        image = self.convert_to_rgb(image)
        
        return image
    
    def preprocess_image(self, image: Image.Image, augment: bool = False) -> torch.Tensor:
        """
        Preprocess a single image.
        
        Args:
            image: PIL Image
            augment: Whether to apply data augmentation
            
        Returns:
            Preprocessed image tensor
        """
        # Clean the image first
        image = self.clean_image(image)
        if image is None:
            raise ValueError("Image is corrupted")
        
        # Apply appropriate transform
        if augment:
            return self.augmentation_transform(image)
        else:
            return self.basic_transform(image)
    
    def preprocess_batch(self, images: List[Image.Image], 
                        augment: bool = False) -> Tuple[torch.Tensor, List[int]]:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of PIL Images
            augment: Whether to apply data augmentation
            
        Returns:
            Tuple of (stacked tensor of valid images, list of valid indices)
        """
        valid_tensors = []
        valid_indices = []
        
        for idx, image in enumerate(images):
            try:
                tensor = self.preprocess_image(image, augment=augment)
                valid_tensors.append(tensor)
                valid_indices.append(idx)
            except ValueError:
                # Skip corrupted images
                continue
        
        if not valid_tensors:
            raise ValueError("No valid images in batch")
        
        return torch.stack(valid_tensors), valid_indices
    
    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize a tensor for visualization.
        
        Args:
            tensor: Normalized image tensor
            
        Returns:
            Denormalized numpy array (0-255 range)
        """
        # Reverse normalization
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy and scale to 0-255
        array = tensor.numpy().transpose(1, 2, 0)
        array = (array * 255).astype(np.uint8)
        
        return array


def compute_dataset_statistics(dataset, num_samples: int = 1000) -> Tuple[List[float], List[float]]:
    """
    Compute mean and std statistics for a dataset.
    
    Args:
        dataset: Dataset to compute statistics for
        num_samples: Number of samples to use for computation
        
    Returns:
        Tuple of (mean, std) for each channel
    """
    print(f"Computing dataset statistics using {num_samples} samples...")
    
    preprocessor = ImagePreprocessor()
    pixels = []
    
    sample_size = min(num_samples, len(dataset))
    
    for i in range(sample_size):
        image = dataset[i]['image']
        image = preprocessor.clean_image(image)
        if image is None:
            continue
        
        # Resize and convert to array
        image = image.resize((224, 224))
        array = np.array(image).astype(np.float32) / 255.0
        pixels.append(array)
    
    pixels = np.stack(pixels)
    
    # Compute mean and std per channel
    mean = pixels.mean(axis=(0, 1, 2))
    std = pixels.std(axis=(0, 1, 2))
    
    print(f"Computed mean: {mean}")
    print(f"Computed std: {std}")
    
    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor()
    print("ImagePreprocessor initialized successfully!")
    print(f"Target size: {preprocessor.target_size}")
    print(f"Normalization mean: {preprocessor.mean}")
    print(f"Normalization std: {preprocessor.std}")
