#!/bin/bash

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gen_embed
#SBATCH --output=logs/gen_embed_%j.out
#SBATCH --error=logs/gen_embed_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --partition=compute1
#SBATCH --gres=gpu:1

# Create logs directory
mkdir -p logs

# Print job information
echo "Job started at: \$(date)"
echo "Running on node: \$(hostname)"

# Load necessary modules
module purge
module load anaconda3
module load cuda/11.8.0

# Run the Python script
python 7_generate_embeddings.py

echo "Job finished at: \$(date)"
EOF

echo "Job submitted. Check logs directory for output files."