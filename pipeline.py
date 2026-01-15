"""
Complete Data Preparation Pipeline for Image Recognition

This script demonstrates the complete data preparation and preprocessing pipeline:
1. Load the Food-101 dataset
2. Display basic information
3. Clean and preprocess images
4. Split data into train/val/test sets
5. Save preprocessed examples

Run this script to prepare your data for modeling.
"""

import os
from data_loader import DataLoader
from preprocessing import ImagePreprocessor
from data_splitting import DataSplitter, compute_class_weights
import numpy as np
from PIL import Image
import json


def main():
    """
    Main pipeline execution.
    """
    print("="*80)
    print("IMAGE RECOGNITION - DATA PREPARATION PIPELINE")
    print("="*80)
    
    # Step 1: Load the dataset
    print("\n[STEP 1] Loading Food-101 Dataset")
    print("-" * 80)
    loader = DataLoader(dataset_name="food101")
    dataset = loader.load_data()
    
    # Display dataset information
    loader.display_info()
    loader.show_examples(split="train", num_examples=3)
    
    # Step 2: Initialize preprocessor
    print("\n[STEP 2] Initializing Image Preprocessor")
    print("-" * 80)
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    print(f"✓ Target image size: {preprocessor.target_size}")
    print(f"✓ Normalization mean: {preprocessor.mean}")
    print(f"✓ Normalization std: {preprocessor.std}")
    
    # Step 3: Validate and clean images
    print("\n[STEP 3] Validating Images")
    print("-" * 80)
    train_data = dataset['train']
    
    print("Checking first 100 images for corruption...")
    corrupted_count = 0
    rgb_conversion_count = 0
    
    for i in range(min(100, len(train_data))):
        image = train_data[i]['image']
        
        # Check if needs RGB conversion
        if image.mode != 'RGB':
            rgb_conversion_count += 1
        
        # Check if corrupted
        if preprocessor.is_corrupted(image):
            corrupted_count += 1
    
    print(f"✓ Images checked: 100")
    print(f"✓ Images needing RGB conversion: {rgb_conversion_count}")
    print(f"✓ Corrupted images: {corrupted_count}")
    
    # Step 4: Split dataset
    print("\n[STEP 4] Creating Train/Val/Test Splits")
    print("-" * 80)
    splitter = DataSplitter(random_seed=42)
    splits = splitter.split_dataset(
        dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # Get label names
    label_names = train_data.features['label'].names
    
    # Display split information
    splitter.display_split_info(splits, label_names=label_names)
    
    # Verify stratification
    splitter.verify_stratification(splits)
    
    # Step 5: Compute class weights for handling imbalance
    print("\n[STEP 5] Computing Class Weights")
    print("-" * 80)
    train_labels = splits['train']['labels']
    class_weights = compute_class_weights(train_labels)
    
    print(f"✓ Computed weights for {len(class_weights)} classes")
    print(f"✓ Weight range: {min(class_weights.values()):.3f} - {max(class_weights.values()):.3f}")
    
    # Show weights for a few classes
    print("\nSample class weights:")
    for i, (class_idx, weight) in enumerate(list(class_weights.items())[:5]):
        print(f"  {label_names[class_idx]}: {weight:.3f}")
    
    # Step 6: Preprocess and save example images
    print("\n[STEP 6] Preprocessing Sample Images")
    print("-" * 80)
    
    # Create output directory
    output_dir = "preprocessed_examples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process a few examples
    num_examples = 5
    train_indices = splits['train']['indices'][:num_examples]
    
    print(f"Processing {num_examples} examples...")
    
    for idx, dataset_idx in enumerate(train_indices):
        example = train_data[dataset_idx]
        image = example['image']
        label = example['label']
        label_name = label_names[label]
        
        # Save original
        original_path = os.path.join(output_dir, f"example_{idx}_original.png")
        image.save(original_path)
        
        # Preprocess without augmentation
        preprocessed = preprocessor.preprocess_image(image, augment=False)
        
        # Denormalize for visualization
        denormalized = preprocessor.denormalize(preprocessed)
        denorm_image = Image.fromarray(denormalized)
        processed_path = os.path.join(output_dir, f"example_{idx}_preprocessed.png")
        denorm_image.save(processed_path)
        
        # Preprocess with augmentation
        augmented = preprocessor.preprocess_image(image, augment=True)
        denormalized_aug = preprocessor.denormalize(augmented)
        aug_image = Image.fromarray(denormalized_aug)
        aug_path = os.path.join(output_dir, f"example_{idx}_augmented.png")
        aug_image.save(aug_path)
        
        print(f"  ✓ Example {idx+1}: {label_name} (Label: {label})")
    
    print(f"\n✓ Saved {num_examples * 3} images to '{output_dir}/'")
    
    # Step 7: Save split information
    print("\n[STEP 7] Saving Split Information")
    print("-" * 80)
    
    split_info = {
        'random_seed': 42,
        'train_size': len(splits['train']['indices']),
        'val_size': len(splits['val']['indices']),
        'test_size': len(splits['test']['indices']),
        'num_classes': len(label_names),
        'class_names': label_names,
        'train_indices': splits['train']['indices'][:100],  # Save first 100 for demo
        'val_indices': splits['val']['indices'][:100],
        'test_indices': splits['test']['indices'][:100]
    }
    
    split_file = "split_info.json"
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"✓ Split information saved to '{split_file}'")
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nSummary:")
    print(f"  • Dataset: Food-101")
    print(f"  • Total samples: {len(train_data)}")
    print(f"  • Train samples: {len(splits['train']['indices'])}")
    print(f"  • Validation samples: {len(splits['val']['indices'])}")
    print(f"  • Test samples: {len(splits['test']['indices'])}")
    print(f"  • Number of classes: {len(label_names)}")
    print(f"  • Image size: {preprocessor.target_size}")
    print(f"  • Random seed: 42 (for reproducibility)")
    print("\nNext steps:")
    print("  1. Use the splits dictionary for training")
    print("  2. Apply preprocessor.preprocess_image() with augment=True for training")
    print("  3. Apply preprocessor.preprocess_image() with augment=False for validation/test")
    print("  4. Use class_weights for handling class imbalance during training")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
