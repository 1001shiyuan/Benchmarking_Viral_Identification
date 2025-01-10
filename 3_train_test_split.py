import os
import random
import shutil
import glob
import gzip
from Bio import SeqIO
from math import floor

def split_data(input_folder, output_base_folder, train_ratio=0.9):
    """
    Split sequences from input folder into training and test sets.
    
    Args:
        input_folder (str): Folder containing .fna.gz files
        output_base_folder (str): Base folder for output
        train_ratio (float): Ratio for training set (default 0.9)
    """
    # Create output directories
    train_dir = os.path.join(output_base_folder, 'train')
    test_dir = os.path.join(output_base_folder, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get list of all genome files
    genome_files = glob.glob(os.path.join(input_folder, "*.fna.gz"))
    
    # Randomly shuffle the files
    random.shuffle(genome_files)
    
    # Calculate split point
    n_files = len(genome_files)
    n_train = floor(n_files * train_ratio)
    
    # Split files into train and test sets
    train_files = genome_files[:n_train]
    test_files = genome_files[n_train:]
    
    print(f"Processing {input_folder}")
    print(f"Total files: {n_files}")
    print(f"Training files: {len(train_files)} ({train_ratio*100}%)")
    print(f"Test files: {len(test_files)} ({(1-train_ratio)*100}%)")
    
    # Copy files to respective directories
    def copy_files(files, dest_dir):
        for f in files:
            dest_file = os.path.join(dest_dir, os.path.basename(f))
            shutil.copy2(f, dest_file)
            print(f"Copied {f} to {dest_file}")
    
    copy_files(train_files, train_dir)
    copy_files(test_files, test_dir)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Input directories (using filtered bacterial data)
    viral_folder = "./positive_samples/viral"
    bacterial_folder = "./negative_samples/bacterial_filtered"
    
    # Output base folders
    output_dir = "./data_split"
    viral_output = os.path.join(output_dir, "viral")
    bacterial_output = os.path.join(output_dir, "bacterial")
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process viral (positive) samples
    print("\nProcessing viral samples...")
    split_data(viral_folder, viral_output)
    
    # Process bacterial (negative) samples
    print("\nProcessing bacterial samples...")
    split_data(bacterial_folder, bacterial_output)