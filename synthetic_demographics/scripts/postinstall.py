import zipfile
import os
from pathlib import Path

def main():
    zip_path = "synthetic_demographics/output_database.zip"
    zip_abs_path = os.path.abspath(zip_path)
    zip_dir = os.path.dirname(zip_abs_path)
    
    print(f"Looking for zip file at: {zip_abs_path}")
    
    if os.path.exists(zip_path):
        print(f"Found zip file: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(zip_dir)
            print(f"Database extracted successfully to: {zip_dir}")
    else:
        print(f"Error: Could not find {zip_path}")
        print(f"Current working directory: {os.getcwd()}")
        print("Files in current directory:")
        for file in Path('.').iterdir():
            print(f"  - {file}")