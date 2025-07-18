import os
import shutil
import pandas as pd
import argparse
import glob
from sklearn.model_selection import train_test_split
import numpy as np

def organize_images_with_split(csv_path, source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Organize images into train/val/test folders based on their labels from a CSV file.
    Rows with empty labels will be skipped.
   
    Args:
        csv_path: Path to the CSV file containing ImageFilename and LabelTrue columns
        source_dir: Directory containing the source images
        output_dir: Base directory where train/val/test folders will be created
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        random_state: Random seed for reproducibility (default: 42)
    """
    # Validate split ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    # Create output directory structure
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
   
    # Read the CSV file
    df = pd.read_csv(csv_path)
   
    # Check if required columns exist
    if 'ImageFilename' not in df.columns or 'LabelTrue' not in df.columns:
        raise ValueError("CSV file must contain 'ImageFilename' and 'LabelTrue' columns")
   
    # Replace empty/NaN labels with a special value and filter out empty labels
    df['LabelTrue'] = df['LabelTrue'].fillna('').astype(str)
    df_filtered = df[df['LabelTrue'].str.strip() != ''].copy()
    
    if len(df_filtered) == 0:
        print("Warning: No valid labels found in the dataset")
        return
   
    # Get unique labels and create directories for each split
    unique_labels = df_filtered['LabelTrue'].unique()
    for split in splits:
        for label in unique_labels:
            label_dir = os.path.join(output_dir, split, str(label))
            os.makedirs(label_dir, exist_ok=True)
    
    # print(f"Found {len(unique_labels)} unique labels: {list(unique_labels)}")
   
    # Split data for each class to maintain class distribution
    split_stats = {'train': 0, 'val': 0, 'test': 0}
    skipped = 0
    
    for label in unique_labels:
        # Get all images for this class
        class_data = df_filtered[df_filtered['LabelTrue'] == label].copy()
        
        if len(class_data) == 0:
            continue
        
        # If there's only one sample, put it in train
        if len(class_data) == 1:
            train_data = class_data
            val_data = pd.DataFrame()
            test_data = pd.DataFrame()
        elif len(class_data) == 2:
            # If only 2 samples, put one in train and one in test
            train_data = class_data.iloc[:1]
            val_data = pd.DataFrame()
            test_data = class_data.iloc[1:]
        else:
            # First split: separate train from (val + test)
            train_data, temp_data = train_test_split(
                class_data, 
                test_size=(val_ratio + test_ratio), 
                random_state=random_state,
                stratify=None
            )
            
            # Second split: separate val from test
            if len(temp_data) == 1:
                val_data = temp_data
                test_data = pd.DataFrame()
            else:
                val_size = val_ratio / (val_ratio + test_ratio)
                val_data, test_data = train_test_split(
                    temp_data, 
                    test_size=(1 - val_size), 
                    random_state=random_state,
                    stratify=None
                )
        
        # Copy images to respective split folders
        splits_data = [
            ('train', train_data),
            ('val', val_data),
            ('test', test_data)
        ]
        
        for split_name, split_data in splits_data:
            if len(split_data) == 0:
                continue
                
            dest_base_dir = os.path.join(output_dir, split_name, str(label))
            
            for _, row in split_data.iterrows():
                image_filename = os.path.basename(str(row['ImageFilename']))
                source_path = os.path.join(source_dir, image_filename)
                dest_path = os.path.join(dest_base_dir, image_filename)
                
                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_path)
                    split_stats[split_name] += 1
                else:
                    print(f"Warning: Source image not found: {source_path}")
                    skipped += 1

def organize_images(csv_path, source_dir, output_dir):
    """
    Original function for backward compatibility - now calls the split version with default ratios
    """
    organize_images_with_split(csv_path, source_dir, output_dir)

if __name__ == "__main__":
    # Parse command line arguments for split ratios
    parser = argparse.ArgumentParser(description='Organize images into train/val/test splits')
    parser.add_argument('--input_folder', type=str, 
                        help='Input folder path to search for CSV files')
    parser.add_argument('--output_folder', type=str, 
                        help='Output folder path for organized dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio (default: 0.15)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Use command line arguments or defaults
    input_folder_path = args.input_folder
    output_folder_path = args.output_folder
    
    # Search for files matching the pattern
    csv_files = glob.glob(os.path.join(input_folder_path, '**/LabelChecker_*.csv'), recursive=True)
    
    # Print the list of matching files
    print(f"Input folder: {input_folder_path}")
    print(f"Output folder: {output_folder_path}")
    print(f"Found {len(csv_files)} LabelChecker files:")
    print(f"Split ratios - Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {args.test_ratio}")
    print(f"Random seed: {args.random_state}")
    print("="*60)
    
    i = 0
    for file_path in csv_files:
        i += 1
        print(f"\nProcessing {i}/{len(csv_files)}: {file_path}")
        csv_path = file_path
        input_folder_path = os.path.dirname(file_path)
        folder_name = os.path.basename(input_folder_path)
        source_path = os.path.join(input_folder_path, folder_name)
        
        organize_images_with_split(
            csv_path, 
            source_path, 
            output_folder_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_state
        )