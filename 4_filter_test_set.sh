#!/bin/bash

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=filter_test
#SBATCH --output=logs/filter_test_%j.out
#SBATCH --error=logs/filter_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --partition=compute1

# Print job information
echo "Job started at: \$(date)"
echo "Running on node: \$(hostname)"

# Create logs directory
mkdir -p logs

# Load necessary modules
module purge
module load anaconda3
module load ncbi/blast/2.11.0

# Run the Python script
python 4_filter_test_set.py

echo "Job finished at: \$(date)"
EOF

echo "Job submitted. Check logs directory for output files."