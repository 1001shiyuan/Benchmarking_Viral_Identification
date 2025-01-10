#!/bin/bash

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=balance_data
#SBATCH --output=logs/balance_data_%j.out
#SBATCH --error=logs/balance_data_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=compute1

# Create logs directory
mkdir -p logs

# Print job information
echo "Job started at: \$(date)"
echo "Running on node: \$(hostname)"

# Load necessary modules
module purge
module load anaconda3

# Run the Python script
python 6_balance_datasets.py

echo "Job finished at: \$(date)"
EOF

echo "Job submitted. Check logs directory for output files."