# Running the training

## Running on a Desktop (Local CPU)

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

## Running on altlab-gpu (HPC GPU slice)

### Get the tools

Make sure that you have all you need installed on the slice. If things are as
originally setup in May 2026 there will be `uv`, `pip` and the rest. To install
packages you'll either need sudo access, or in the case of `uv` you can just do
a local install in your home directory (this is the default).

*This is where the list of the commands to install everything goes*

### Get the data

*comment here what actually was done*

#### Easiest ways

This may require some juggling depending on what the ethics and data
sovereignty rules say. The easiest ways to get data uploaded is with `scp` or
`rsync` which means uploading the files from a commandline from outside of the
system, or by having them in a place where they can be accessed with `wget` or
`curl` from inside the altlab-gpu environment. 

#### Google drive

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


## Run the training

```bash

```


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
