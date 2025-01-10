#!/bin/bash

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gen_contigs
#SBATCH --output=logs/gen_contigs_%j.out
#SBATCH --error=logs/gen_contigs_%j.err
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
python 5_generate_contigs.py

echo "Job finished at: \$(date)"
EOF

echo "Job submitted. Check logs directory for output files."