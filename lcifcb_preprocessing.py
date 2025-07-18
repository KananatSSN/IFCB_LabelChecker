"""
ROI File Processor CLI
Processes .roi and .adc files to extract images and generate CSV metadata.
"""

import argparse
import sys
from pathlib import Path
import os
import numpy as np
import cv2
import pandas as pd
import uuid


def process_files(input_folder_path, output_folder_path, inplace=False):
    """
    Process ROI and ADC files to extract images and generate CSV metadata.
    
    Args:
        input_folder_path (str): Path to input folder
        output_folder_path (str): Path to output folder
        inplace (bool): If True, output to input folder and remove original files
    """
    unique_filenames = list({file.stem for file in Path(input_folder_path).iterdir() if file.is_file()})
    
    for filename in unique_filenames:
        roi_path = os.path.join(input_folder_path, f"{filename}.roi")
        adc_path = os.path.join(input_folder_path, f"{filename}.adc")
        hdr_path = os.path.join(input_folder_path, f"{filename}.hdr")
        
        # Check if required files exist
        if not (Path(roi_path).exists() and Path(adc_path).exists()):
            print(f"Skipping {filename}: missing .roi or .adc file")
            continue
        
        # Create output directory structure
        output_image_folder_path = os.path.join(output_folder_path, filename, filename)
        os.makedirs(output_image_folder_path, exist_ok=True)
        
        # Set up DataFrame columns
        LC_col_str = "Name,Date,Time,CollageFile,ImageFilename,Id,GroupId,Uuid,SrcImage,SrcX,SrcY,ImageX,ImageY,ImageW,ImageH,Timestamp,ElapsedTime,CalConst,CalImage,AbdArea,AbdDiameter,AbdVolume,AspectRatio,AvgBlue,AvgGreen,AvgRed,BiovolumeCylinder,BiovolumePSpheroid,BiovolumeSphere,Ch1Area,Ch1Peak,Ch1Width,Ch2Area,Ch2Ch1Ratio,Ch2Peak,Ch2Width,Ch3Area,Ch3Peak,Ch3Width,CircleFit,Circularity,CircularityHu,Compactness,ConvexPerimeter,Convexity,EdgeGradient,Elongation,EsdDiameter,EsdVolume,FdDiameter,FeretMaxAngle,FeretMinAngle,FiberCurl,FiberStraightness,FilledArea,FilterScore,GeodesicAspectRatio,GeodesicLength,GeodesicThickness,Intensity,Length,Perimeter,Ppc,RatioBlueGreen,RatioRedBlue,RatioRedGreen,Roughness,ScatterArea,ScatterPeak,SigmaIntensity,SphereComplement,SphereCount,SphereUnknown,SphereVolume,SumIntensity,Symmetry,Transparency,Width,BiovolumeHSosik,SurfaceAreaHSosik,Preprocessing,PreprocessingTrue,LabelPredicted,ProbabilityScore,LabelTrue"
        LC_columns = LC_col_str.split(",")
        LC_df = pd.DataFrame(columns=LC_columns)
        
        print(f"Processing {filename}...")
        
        # Read ROI data
        roi_data = np.fromfile(roi_path, dtype="uint8")
        
        # Process ADC file
        with open(adc_path, 'r') as adc_data:
            for i, adc_line in enumerate(adc_data):
                adc_line = adc_line.split(",")
                output_image_name = f"{filename}_{i:05d}.png"
                output_image_path = os.path.join(output_folder_path, filename, filename, output_image_name)
                
                # Populate DataFrame
                LC_df.loc[i, 'Name'] = filename
                LC_df.loc[i, 'ImageFilename'] = output_image_name
                LC_df.loc[i, 'Id'] = i
                LC_df.loc[i, 'GroupId'] = i
                LC_df.loc[i, 'ImageW'] = adc_line[15]
                LC_df.loc[i, 'ImageH'] = adc_line[16]
                LC_df.loc[i, 'Uuid'] = uuid.uuid4().hex.upper()
                
                # Extract and save image
                roi_x = int(adc_line[15])  # ROI width
                roi_y = int(adc_line[16])  # ROI height
                image_size = roi_x * roi_y
                
                if image_size > 0:
                    roi_start_bit = int(adc_line[17])
                    roi_end_bit = roi_start_bit + image_size
                    image = roi_data[roi_start_bit:roi_end_bit].reshape((roi_y, roi_x))
                    cv2.imwrite(output_image_path, image)
        
        # Save CSV file
        LC_file_name = f"LabelChecker_{filename}.csv"
        LC_file_path = os.path.join(output_folder_path, filename, LC_file_name)
        LC_df.to_csv(LC_file_path, index=False)
        
        # Remove original files if inplace option is enabled
        if inplace:
            print(f"Removing original files for {filename}...")
            try:
                os.remove(roi_path)
                print(f"  Removed {roi_path}")
            except FileNotFoundError:
                pass
            
            try:
                os.remove(adc_path)
                print(f"  Removed {adc_path}")
            except FileNotFoundError:
                pass
            
            try:
                os.remove(hdr_path)
                print(f"  Removed {hdr_path}")
            except FileNotFoundError:
                pass
        
        print(f"Completed processing {filename}")


def main():
    parser = argparse.ArgumentParser(description='Process ROI and ADC files to extract images and generate CSV metadata.')
    
    parser.add_argument('input_folder', 
                       help='Path to the input folder containing .roi, .adc, and .hdr files')
    
    parser.add_argument('-o', '--output', 
                       default=None,
                       help='Path to the output folder (default: same as input folder)')
    
    parser.add_argument('--inplace', 
                       action='store_true',
                       help='Process files in place (output to input folder and remove original files)')
    
    parser.add_argument('-v', '--verbose', 
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input folder
    input_folder_path = Path(args.input_folder)
    if not input_folder_path.exists():
        print(f"Error: Input folder '{input_folder_path}' does not exist.")
        sys.exit(1)
    
    if not input_folder_path.is_dir():
        print(f"Error: '{input_folder_path}' is not a directory.")
        sys.exit(1)
    
    # Determine output folder
    if args.output:
        output_folder_path = Path(args.output)
    else:
        output_folder_path = input_folder_path
    
    # If inplace is specified, force output to be same as input
    if args.inplace:
        output_folder_path = input_folder_path
        if args.output and Path(args.output) != input_folder_path:
            print("Warning: --inplace specified, ignoring output folder argument")
    
    # Create output folder if it doesn't exist
    output_folder_path.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        print(f"Input folder: {input_folder_path}")
        print(f"Output folder: {output_folder_path}")
        print(f"In-place processing: {args.inplace}")
    
    # Process files
    try:
        process_files(str(input_folder_path), str(output_folder_path), args.inplace)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()