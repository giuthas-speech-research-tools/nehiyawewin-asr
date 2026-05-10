# Whisper ASR Fine-Tuning Setup

This repository contains scripts for fine-tuning Hugging Face's Whisper models
on local speech data (e.g., Cree). The pipeline is configured via YAML profiles
to transition between local desktop testing (CPU) and high-performance
computing (HPC) environments like Digital Alliance's Narval (A100 GPUs).

## 1. Getting Started: Clone and Setup Environment

First, clone the repository and set up your Python environment. You can use
standard `venv` or a faster manager like `uv`.

```bash
# Clone the repository
git clone https://github.com/giuthas-speech-research-tools/whisper-test
cd whisper-test
```

### Using pip:
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install required dependencies
pip install torch transformers datasets evaluate pandas pyyaml

```

### Using uv:
No need to do anything here. Just replace all `python` calls with `uv python` in the next steps.


## 2. Prepare Data and Offline Assets

Compute nodes on HPC environments (like Narval) are air-gapped and lack internet access. You must download the model weights and evaluation metrics to your local directory *before* submitting your Slurm jobs.

**A. Generate Metadata**
Map your `.wav` and `.sro` files into a unified dataset:

```bash
python wrap_sro_data.py

```

*This assumes your audio files are in `wav/` and transcripts in `txt/`. It outputs `metadata.csv`.*

**B. Download Hugging Face Assets (Internet Connection Required)**
Run these commands on your desktop or the HPC **login node**:

```bash
# Download the Whisper-tiny model locally
huggingface-cli download openai/whisper-tiny --local-dir ./local-whisper-tiny

# Download WER and CER evaluation scripts locally
mkdir -p metrics
python -c "import evaluate; evaluate.load('wer').save_local('./metrics/wer')"
python -c "import evaluate; evaluate.load('cer').save_local('./metrics/cer')"

```

## 3. Running on a Desktop (Local CPU)

The desktop environments are driven directly via the terminal using the provided YAML configuration files.

**Short Sanity Check (Validates data mapping and loss calculation):**

```bash
python train_whisper.py --config configs/desktop_test.yaml
python test_whisper.py --config configs/desktop_test.yaml

```

**Full Overnight Run:**

```bash
python train_whisper.py --config configs/desktop_full.yaml
python test_whisper.py --config configs/desktop_full.yaml

```

## 4. Running on Digital Alliance Narval (HPC GPU)

To run on Narval, you cannot execute Python directly on the head node. You must submit a Slurm batch script that sets strict offline environment variables, requests an A100 GPU, and points to the `narval_full.yaml` config.

Create a script named `submit_narval.sh` in the project root:

```bash
#!/bin/bash
#SBATCH --account=def-youraccount      # Replace with your actual allocation account
#SBATCH --time=12:00:00                
#SBATCH --cpus-per-task=12             # Max CPU cores for 1 GPU on Narval
#SBATCH --mem=48G                      
#SBATCH --gpus-per-node=a100:1         
#SBATCH --job-name=cree_whisper
#SBATCH --output=%x-%j.out

module load python/3.10
source .venv/bin/activate

# CRITICAL: Force Hugging Face into strictly offline mode
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

echo "Starting GPU Training..."
python train_whisper.py --config configs/narval_full.yaml

echo "Starting Evaluation..."
python test_whisper.py --config configs/narval_full.yaml

```

Submit your job to the cluster scheduler:

```bash
sbatch submit_narval.sh

```

You can monitor your job's progress using `sq` and read the output logs using `tail -f cree_whisper-<job_id>.out`.

```

```