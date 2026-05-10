#!/bin/bash
#SBATCH --account=def-aarppe        # Your PI's HPC allocation
#SBATCH --gpus-per-node=a100:1      # Request exactly 1 NVIDIA A100 GPU
#SBATCH --cpus-per-task=4           # Request 4 CPU cores for data loading
#SBATCH --mem=32G                   # Request 32 GB of RAM
#SBATCH --time=06:00:00             # Time limit (6 hours - adjust as needed)
#SBATCH --job-name=whisper_cree     # Name of the job
#SBATCH --output=%x-%j.out          # Saves terminal output to a log file (whisper_cree-JOBID.out)

# 1. Load the required modules on Narval
module purge
module load StdEnv/2023
module load python/3.10
module load arrow/14.0.1  # Highly recommended for the HuggingFace datasets library

# 2. Create a fast, temporary virtual environment on the compute node
# We use $SLURM_TMPDIR because it is located on the node's extremely fast local storage
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# 3. Install necessary Python packages
# --no-index tells it to pull from the Alliance's pre-compiled wheels rather than the internet, which is much faster and safer
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio 
pip install --no-index transformers datasets pandas evaluate accelerate

# 4. Run your python training script
python train_whisper.py