#!/bin/bash

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train_clf
#SBATCH --output=logs/train_clf_%j.out
#SBATCH --error=logs/train_clf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=70:00:00
#SBATCH --partition=compute1
#SBATCH --mem=0

# Create logs directory
mkdir -p logs

# Print job information
echo "Job started at: \$(date)"
echo "Running on node: \$(hostname)"

# Load necessary modules
module purge
module load anaconda3

# Create results directory
mkdir -p results

# Run the Python script
python 8_train_classifier.py

echo "Job finished at: \$(date)"
EOF

echo "Job submitted. Check logs directory for output files."