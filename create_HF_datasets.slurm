#!/bin/bash
# create_HF_datasets.slurm
# SLURM script to generate COGITAO datasets

#SBATCH --job-name=COGITAO_hf_dataset_creation
#SBATCH -D ./
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/data/ktroyan/COGITAO

set -e
mkdir -p logs

# -----------------------
# Load environment
# -----------------------
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "No virtualenv found!"
    exit 1
fi

# HuggingFace token (only required if --upload is used)
if [ -f ~/.secrets/hf_token ]; then
    export HF_TOKEN=$(cat ~/.secrets/hf_token)
fi

# -----------------------
# Info
# -----------------------
echo "Running on node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"

# -----------------------
# Run dataset generation
# -----------------------

srun -u uv run generate_experiment_data_parallel.py \
    "$@"

echo "Finished at: $(date)"