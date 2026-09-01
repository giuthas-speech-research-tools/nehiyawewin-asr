# Getting Started

To use the nehiyawewin-asr scripts you will need a couple of basic command line
tools: `git` and `uv`. `Git` is used for version control and for cloning the
repository (that is where the code is kept) and `uv` is used for setting up the Python environment and running the scripts. 

If you would like to or need to modify the scripts, you should also get a IDE
such as VSCodium.

Git also has various GUI implementations, but those are operating system
dependent and so you will have to look them up and select the one you would
like yourself. Git can be used from just the commandline without problems
though.

Here are some potentially useful links to these tools:
- [Getting `git`](https://git-scm.com/install/)
- [Getting `uv`](https://docs.astral.sh/uv/getting-started/installation/)
- [Getting VSCodium](https://vscodium.com/)


## 1. Clone and Setup Environment

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

No need to do anything here. Just replace all `python` calls with `uv run python` in the next steps.


## Prepare Data and Offline Assets

**A. Generate Metadata**
Map your `.wav` and `.sro` files into a unified dataset:

```bash
python wrap_sro_data.py
```

*This assumes your audio files are in `wav/` and transcripts in `txt/`. It
outputs `metadata.csv`.*

**B. Download Hugging Face Assets (Internet Connection Required)**

Run this where ever you are going to keep the local untuned whisper models. In
the altlab-gpu setup this is `/data/plains-cree-asr/hf_cache/`.


### Using pip

```bash
# Download the Whisper-tiny model locally
hf download openai/whisper-tiny --local-dir ./local-whisper-tiny
```

### Using uv

```bash
# Download the Whisper-tiny model locally
uvx hf download openai/whisper-tiny --local-dir ./local-whisper-tiny
```

### In practice on altlab-gpu

This is how this was done on altlab-gpu. Replace `tiny base small medium large`
below with a list of the models you actually want. 

```bash
mkdir local_whisper_models
for model in tiny base small medium large; 
do hf download openai/whisper-$model --local-dir ./local_whisper_models/whisper-$model; 
done
```


### Get the metrics scripts

```bash
# Download WER and CER evaluation scripts locally
mkdir -p metrics
mkdir -p ./metrics/wer
mkdir -p ./metrics/cer
curl -L https://huggingface.co/spaces/evaluate-metric/wer/raw/main/wer.py -o ./metrics/wer/wer.py
curl -L https://huggingface.co/spaces/evaluate-metric/cer/raw/main/cer.py -o ./metrics/cer/cer.py
```

## Upload files to HPC

First clone this repository on HPC. For this you should most likely generate a
ssh key (in what ever is the latest, bestest format) on HPC and then upload it
on github. 

Compute nodes on HPC environments (like Narval) maybe air-gapped and therefore
lack internet access. Because specialized tools like huggingface-cli might not
be installed on your HPC cluster, you should upload the weights and what not
locally on the node.

Run these inside the repository on your own machine.

```bash
# The wrapper file for the training and testing data.
rsync -avzP ./metadata.csv your_username@narval.computecanada.ca:~/whisper-test/
# Replace [whisper_model_directory] with the name of the model you are going to train.
rsync -avzP ./[whisper_model_directory] your_username@narval.computecanada.ca:~/whisper-test/
# The data
rsync -avzP ./[data_dir] your_username@narval.computecanada.ca:~/whisper-test/
```

For example this is how the first dataset and model were uploaded (minus username, because that would be telling).
```bash
rsync -avzP ./metadata.csv your_username@narval.computecanada.ca:~/whisper-test/
rsync -avzP ./local-whisper-small your_username@narval.computecanada.ca:~/whisper-test/
rsync -avzP ./sand-psalm your_username@narval.computecanada.ca:~/whisper-test/
```

## Running the training

### Running on a Desktop (Local CPU)

The desktop environments are driven directly via the terminal using the
provided YAML configuration files.

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

### Running on altlab-gpu (HPC GPU slice)

#### Get the tools

Make sure that you have all you need installed on the slice. If things are as
originally setup in May 2026 there will be `uv`, `pip` and the rest. To install
packages you'll either need sudo access, or in the case of `uv` you can just do
a local install in your home directory (this is the default).

*This is where the list of the commands to install everything goes*

#### Get the data

*comment here what actually was done*

##### Easiest ways

This may require some juggling depending on what the ethics and data
sovereignty rules say. The easiest ways to get data uploaded is with `scp` or
`rsync` which means uploading the files from a commandline from outside of the
system, or by having them in a place where they can be accessed with `wget` or
`curl` from inside the altlab-gpu environment. 

##### Google drive

**Note** Needs to be tested as this is copied/adapted from Stack Overflow answer https://stackoverflow.com/users/3063243/phi

**Caveats**
- Only works on open access files. ("Anyone who has a link can View") If this
  is not acceptable then do not use this method.
- Cannot download more than 50 files into a single folder.
    - You can consider using tar/zip to make it a single file to work around this limitation.

**/Caveats**

To move data from google drive, do it with `gdown` from altlab-gpu.
Install it with the following command:

```
pip install gdown
```

After that, you can download any file from Google Drive by running one of these commands:

```
gdown https://drive.google.com/uc?id=<file_id>  # for files
gdown <file_id>                                 # alternative format
gdown --folder https://drive.google.com/drive/folders/<file_id>  # for folders
gdown --folder --id <file_id>                                   # this format works for folders too
```

Example: to download the readme file from this directory

```
gdown https://drive.google.com/uc?id=0B7EVK8r0v71pOXBhSUdJWU1MYUk
```

The file_id should look something like 0Bz8a_Dbh9QhbNU3SGlFaDg. You can find this ID by right-clicking on the file of interest, and selecting Get link. As of November 2021, this link will be of the form:

```
# Files
https://drive.google.com/file/d/<file_id>/view?usp=sharing
# Folders
https://drive.google.com/drive/folders/<file_id>
```


### Run the training

```bash

```

### D. Analyse the results

- Training time should be reported **and is currently not logged**

Get the training

## Appendices

### A. Running on Digital Alliance Narval (HPC GPU)

**Note** This was never used, and therefore is untested.

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
