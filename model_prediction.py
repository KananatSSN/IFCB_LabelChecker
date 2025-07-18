import os
import pandas as pd
import glob
import argparse
from ultralytics import YOLO #8.3.44
import numpy as np

def predict_LC(csv_path, model_path, threshold=0.5):
    """
    Use YOLO model to predict labels for images with empty LabelTrue in CSV file.
    Updates the CSV file with predictions and probability scores.
   
    Args:
        csv_path: Path to the CSV file containing ImageFilename and LabelTrue columns
        model_path: Path to the YOLO model file
        threshold: Minimum confidence threshold for predictions (default: 0.5)
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
   
    # This is to make pandas stop crying
    df['LabelTrue'] = df['LabelTrue'].fillna('')
    df['ProbabilityScore'] = df['ProbabilityScore'].astype(float)
    df['LabelPredicted'] = df['LabelPredicted'].astype(str)
    df['Preprocessing'] = df['Preprocessing'].astype(str)
   
    # Load YOLO model
    model = YOLO(model_path)
    
    # Determine image folder path
    folder_path = os.path.dirname(csv_path)
    folder_name = os.path.basename(folder_path)
    image_folder = os.path.join(folder_path, folder_name)
    
    # Count images to process
    images_to_process = len(df[(df['LabelTrue'] == '') & (df['Preprocessing'] != 'small')])
    processed_count = 0
    error_count = 0
    prediction_count = 0
    
    print(f"     → Image folder: {os.path.basename(image_folder)}")
    print(f"     → Images to process: {images_to_process}")
    print(f"     → Confidence threshold: {threshold}")
    
    # Process images
    for index, row in df.iterrows():
        if index % 500 == 0 and index > 0:
            print(f"     → Processed {index}/{len(df)} rows...")
       
        if row['LabelTrue'] == '' and row['Preprocessing'] != 'small':
            processed_count += 1
            try:
                image_filename = row['ImageFilename']
                image_path = os.path.join(image_folder, image_filename)
               
                results = model(image_path, verbose=False)
                probability_score = float(results[0].probs.top1conf.item())  # Get confidence
                label_predicted = str(results[0].names[results[0].probs.top1])  # Get class name
                
                if probability_score >= threshold:
                    df.loc[index, 'ProbabilityScore'] = round(probability_score, 2)
                    df.loc[index, 'LabelPredicted'] = label_predicted
                    prediction_count += 1
                    
            except Exception as e:
                print(f"     → ERROR: Failed to process {image_filename}")
                error_count += 1
        else:
            df.loc[index, 'ProbabilityScore'] = np.nan
            df.loc[index, 'LabelPredicted'] = ''
    
    # Save updated CSV
    df.to_csv(csv_path, index=False, na_rep='')
    
    print(f"     → COMPLETED: {prediction_count} predictions made")
    print(f"     → Errors: {error_count} images skipped")
    
    return processed_count, prediction_count, error_count

def process_csv_files(folder_path, model_path, threshold=0.5):
    """
    Process all LabelChecker_*.csv files in the given folder and subfolders.
    
    Args:
        folder_path: Root directory to search for CSV files
        model_path: Path to the YOLO model file
        threshold: Minimum confidence threshold for predictions
    """
    # Search for files matching the pattern
    csv_files = glob.glob(os.path.join(folder_path, '**/LabelChecker_*.csv'), recursive=True)
    
    if not csv_files:
        print(f"No LabelChecker_*.csv files found in {folder_path}")
        return
    
    print(f"Found {len(csv_files)} LabelChecker CSV files:")
    print("="*60)
    
    total_processed = 0
    total_predictions = 0
    total_errors = 0
    
    for i, file_path in enumerate(csv_files, 1):
        csv_path = file_path
        relative_path = os.path.relpath(csv_path, folder_path)
        
        print(f"{i:3d}. {relative_path}")
        
        try:
            processed, predictions, errors = predict_LC(csv_path, model_path, threshold)
            total_processed += processed
            total_predictions += predictions
            total_errors += errors
            
        except Exception as e:
            print(f"     → ERROR: Failed to process file - {str(e)}")
    
    # Print summary
    print("="*60)
    print(f"PREDICTION SUMMARY:")
    print(f"Total CSV files processed: {len(csv_files)}")
    print(f"Total images processed: {total_processed}")
    print(f"Total predictions made: {total_predictions}")
    if total_errors > 0:
        print(f"Total errors: {total_errors}")
    print(f"Model used: {os.path.basename(model_path)}")
    print(f"Confidence threshold: {threshold}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use YOLO model to predict labels for images in LabelChecker CSV files')
    parser.add_argument('--input_folder', type=str, 
                        help='Input folder path to search for CSV files')
    parser.add_argument('--model_path', type=str, 
                        help='Path to YOLO model file')
    parser.add_argument('--threshold', type=float, default=0.8, 
                        help='Confidence threshold for predictions (default: 0.8)')
    
    args = parser.parse_args()
    
    # Validate input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder does not exist: {args.input_folder}")
        exit(1)
        
    # Validate model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file does not exist: {args.model_path}")
        exit(1)
    
    print(f"Input folder: {args.input_folder}")
    print(f"Model path: {args.model_path}")
    print(f"Confidence threshold: {args.threshold}")
    print()
    
    process_csv_files(args.input_folder, args.model_path, args.threshold)