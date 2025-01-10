#!/bin/bash

# Input folders
BACTERIAL_FOLDER="./negative_samples/bacterial"
VIRAL_FOLDER="./positive_samples/viral"
OUTPUT_FOLDER="./negative_samples/bacterial_filtered"

# Create a logs directory if it doesn't exist
mkdir -p logs

# Submit the job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=filter_viral
#SBATCH --output=logs/filter_viral_%j.out
#SBATCH --error=logs/filter_viral_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --partition=compute1

# Print job information
echo "Job started at: \$(date)"
echo "Running on node: \$(hostname)"

# Load necessary modules
module purge
module load anaconda3
module load ncbi/blast/2.11.0

# Run the Python script
python 2_filter_viral_elements.py \
    --bacterial_folder "${BACTERIAL_FOLDER}" \
    --viral_folder "${VIRAL_FOLDER}" \
    --output_folder "${OUTPUT_FOLDER}"

echo "Job finished at: \$(date)"
EOF

echo "Job submitted. Check logs directory for output files."