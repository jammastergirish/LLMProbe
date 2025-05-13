#!/usr/bin/env python
"""
Download script for the Azaria-Mitchell True-False dataset.
This script downloads the dataset from the original URL and extracts it to the
correct location with the appropriate file naming convention.
"""

import os
import sys
import zipfile
import requests
from io import BytesIO
import pandas as pd

# Constants
DATASET_URL = "https://www.azariaa.com/Content/Datasets/true-false-dataset.zip"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "azaria-mitchell")

def download_dataset():
    """Download and extract the Azaria-Mitchell dataset."""
    print(f"Creating output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Downloading dataset from: {DATASET_URL}")
    try:
        response = requests.get(DATASET_URL)
        response.raise_for_status()  # Raise exception for HTTP errors
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False
    
    print("Download complete. Extracting files...")
    try:
        # Extract zip file contents
        z = zipfile.ZipFile(BytesIO(response.content))
        z.extractall(OUTPUT_DIR)
        
        # List extracted files
        extracted_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]
        if not extracted_files:
            print("No CSV files found in the downloaded dataset.")
            return False
            
        print(f"Successfully extracted {len(extracted_files)} files:")
        for file in extracted_files:
            # Format filenames to match the expected pattern
            category = file.split('_')[0].lower()
            new_filename = f"{category}_true_false.csv"
            old_path = os.path.join(OUTPUT_DIR, file)
            new_path = os.path.join(OUTPUT_DIR, new_filename)
            
            # Only rename if needed
            if file != new_filename:
                os.rename(old_path, new_path)
                print(f"  - Renamed {file} to {new_filename}")
            else:
                print(f"  - {file}")
            
            # Verify file structure and content
            try:
                df = pd.read_csv(new_path)
                if 'statement' not in df.columns or 'label' not in df.columns:
                    print(f"    WARNING: File {new_filename} does not have the expected columns ('statement' and 'label').")
                else:
                    print(f"    OK: {len(df)} examples ({df['label'].sum()} true, {len(df) - df['label'].sum()} false)")
            except Exception as e:
                print(f"    ERROR: Could not read {new_filename}: {e}")
        
        return True
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False

if __name__ == "__main__":
    print("Azaria-Mitchell True-False Dataset Downloader")
    print("=" * 50)
    success = download_dataset()
    
    if success:
        print("\nDataset downloaded and extracted successfully!")
        print(f"Dataset location: {OUTPUT_DIR}")
        print("\nYou can now use the 'azaria-mitchell' dataset option in the LLM Probe.")
    else:
        print("\nFailed to download or extract the dataset.")
        print("Please manually download the dataset from:")
        print(DATASET_URL)
        print(f"And extract it to: {OUTPUT_DIR}")