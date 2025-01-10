#!/bin/bash
#SBATCH --job-name=genome_download       # Job name
#SBATCH --output=download_data.out       # Output file
#SBATCH --error=download_data.err        # Error file

# Load 'parallel'
module load parallel

# Run the Python script
python 1_download_data.py
