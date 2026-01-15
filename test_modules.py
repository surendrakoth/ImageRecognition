"""
Test script to verify all modules work correctly without downloading the full dataset.
This script tests the individual components to ensure the implementation is correct.
"""

import numpy as np
from PIL import Image
import torch

print("="*80)
print("TESTING DATA PREPARATION MODULES")
print("="*80)

# Test 1: Import all modules
print("\n[TEST 1] Importing modules...")
try:
    from data_loader import DataLoader
    from preprocessing import ImagePreprocessor
    from data_splitting import DataSplitter, compute_class_weights
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: ImagePreprocessor initialization
print("\n[TEST 2] Testing ImagePreprocessor...")
try:
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    assert preprocessor.target_size == (224, 224)
    assert preprocessor.mean == [0.485, 0.456, 0.406]
    assert preprocessor.std == [0.229, 0.224, 0.225]
    print("✓ ImagePreprocessor initialized correctly")
    print(f"  - Target size: {preprocessor.target_size}")
    print(f"  - Normalization mean: {preprocessor.mean}")
    print(f"  - Normalization std: {preprocessor.std}")
except Exception as e:
    print(f"✗ ImagePreprocessor test failed: {e}")
    exit(1)

# Test 3: Image preprocessing with synthetic image
print("\n[TEST 3] Testing image preprocessing...")
try:
    # Create a synthetic RGB image
    test_image = Image.new('RGB', (300, 300), color=(128, 128, 128))
    
    # Test RGB conversion
    rgb_image = preprocessor.convert_to_rgb(test_image)
    assert rgb_image.mode == 'RGB'
    print("✓ RGB conversion works")
    
    # Test corruption detection
    is_corrupt = preprocessor.is_corrupted(test_image)
    assert not is_corrupt
    print("✓ Corruption detection works")
    
    # Test cleaning
    cleaned = preprocessor.clean_image(test_image)
    assert cleaned is not None
    assert cleaned.mode == 'RGB'
    print("✓ Image cleaning works")
    
    # Test preprocessing without augmentation
    preprocessed = preprocessor.preprocess_image(test_image, augment=False)
    assert isinstance(preprocessed, torch.Tensor)
    assert preprocessed.shape == (3, 224, 224)
    print(f"✓ Preprocessing without augmentation works (shape: {preprocessed.shape})")
    
    # Test preprocessing with augmentation
    augmented = preprocessor.preprocess_image(test_image, augment=True)
    assert isinstance(augmented, torch.Tensor)
    assert augmented.shape == (3, 224, 224)
    print(f"✓ Preprocessing with augmentation works (shape: {augmented.shape})")
    
    # Test denormalization
    denormalized = preprocessor.denormalize(preprocessed)
    assert isinstance(denormalized, np.ndarray)
    assert denormalized.shape == (224, 224, 3)
    assert denormalized.dtype == np.uint8
    print(f"✓ Denormalization works (shape: {denormalized.shape})")
    
except Exception as e:
    print(f"✗ Image preprocessing test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Test DataSplitter with synthetic data
print("\n[TEST 4] Testing DataSplitter...")
try:
    # Create synthetic dataset structure
    class SyntheticDataset:
        def __init__(self, size=10000, num_classes=100):
            # Ensure balanced classes with enough samples for stratification
            samples_per_class = size // num_classes
            labels = []
            for i in range(num_classes):
                labels.extend([i] * samples_per_class)
            
            self.data = {
                'train': {
                    'label': labels
                }
            }
        
        def __getitem__(self, key):
            return self.data[key]
        
        def keys(self):
            return self.data.keys()
    
    class SyntheticSplit:
        def __init__(self, labels):
            self.labels = labels
        
        def __getitem__(self, key):
            if key == 'label':
                return self.labels
            raise KeyError(key)
        
        def __len__(self):
            return len(self.labels)
    
    # Create dataset with proper structure
    labels = []
    for i in range(100):
        labels.extend([i] * 100)
    
    class SyntheticDatasetFixed:
        def __init__(self, labels):
            self.split_data = SyntheticSplit(labels)
        
        def __getitem__(self, key):
            if key == 'train':
                return self.split_data
            raise KeyError(key)
        
        def keys(self):
            return ['train']
    
    synthetic_dataset = SyntheticDatasetFixed(labels)
    
    splitter = DataSplitter(random_seed=42)
    assert splitter.random_seed == 42
    print("✓ DataSplitter initialized with random_seed=42")
    
    # Test splitting
    splits = splitter.split_dataset(
        synthetic_dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    assert 'train' in splits
    assert 'val' in splits
    assert 'test' in splits
    print("✓ Dataset split into train/val/test")
    
    train_size = len(splits['train']['indices'])
    val_size = len(splits['val']['indices'])
    test_size = len(splits['test']['indices'])
    total_size = train_size + val_size + test_size
    
    assert total_size == 10000
    assert abs(train_size - 8000) < 100  # Allow small variance due to stratification
    assert abs(val_size - 1000) < 100
    assert abs(test_size - 1000) < 100
    print(f"✓ Split sizes correct: train={train_size}, val={val_size}, test={test_size}")
    
    # Test stratification verification
    is_stratified = splitter.verify_stratification(splits)
    print(f"✓ Stratification verification works (result: {is_stratified})")
    
except Exception as e:
    print(f"✗ DataSplitter test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Test class weights computation
print("\n[TEST 5] Testing class weights computation...")
try:
    # Create imbalanced labels
    labels = [0]*100 + [1]*50 + [2]*25
    weights = compute_class_weights(labels)
    
    assert len(weights) == 3
    assert weights[0] < weights[1] < weights[2]  # More rare classes get higher weights
    print("✓ Class weights computed correctly")
    print(f"  - Class 0 (100 samples): weight={weights[0]:.3f}")
    print(f"  - Class 1 (50 samples): weight={weights[1]:.3f}")
    print(f"  - Class 2 (25 samples): weight={weights[2]:.3f}")
    
except Exception as e:
    print(f"✗ Class weights test failed: {e}")
    exit(1)

# Test 6: Test batch preprocessing
print("\n[TEST 6] Testing batch preprocessing...")
try:
    images = [
        Image.new('RGB', (300, 300), color=(255, 0, 0)),
        Image.new('RGB', (400, 200), color=(0, 255, 0)),
        Image.new('RGB', (200, 400), color=(0, 0, 255))
    ]
    
    batch_tensor, valid_indices = preprocessor.preprocess_batch(images, augment=False)
    
    assert isinstance(batch_tensor, torch.Tensor)
    assert batch_tensor.shape[0] == 3  # 3 images
    assert batch_tensor.shape[1:] == (3, 224, 224)  # CHW format
    assert valid_indices == [0, 1, 2]
    print(f"✓ Batch preprocessing works (shape: {batch_tensor.shape})")
    
except Exception as e:
    print(f"✗ Batch preprocessing test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 7: Test RGBA to RGB conversion
print("\n[TEST 7] Testing RGBA to RGB conversion...")
try:
    rgba_image = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
    rgb_image = preprocessor.convert_to_rgb(rgba_image)
    
    assert rgb_image.mode == 'RGB'
    assert rgb_image.size == (100, 100)
    print("✓ RGBA to RGB conversion works")
    
except Exception as e:
    print(f"✗ RGBA conversion test failed: {e}")
    exit(1)

# Final summary
print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\nSummary:")
print("  ✓ Module imports successful")
print("  ✓ ImagePreprocessor initialization correct")
print("  ✓ Image preprocessing pipeline functional")
print("  ✓ DataSplitter creates proper splits")
print("  ✓ Class weights computation accurate")
print("  ✓ Batch preprocessing works")
print("  ✓ Image format conversions functional")
print("\nThe implementation is ready for use with real datasets!")
print("="*80)
