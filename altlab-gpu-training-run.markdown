
### How to Run Whisper Fine-Tuning on altlab-gpu

**1. Create a New Configuration File**
Always start by duplicating the working base configuration file so you don't overwrite it. Run this in your terminal (replace `my_run.yaml` with your desired name):

```bash
cp configs/altlab-gpu-tiny.yaml configs/my_run.yaml

```

**2. Update Your Configuration File**
Open your newly created `configs/my_run.yaml` file in a text editor (like `nano`) and update the following critical lines:

* **`base_model_id`**: Set this to the absolute path of the local model you want to use. It **must** be inside the large drive directory:
`base_model_id: "/mnt/hf_cache/local_whisper_models/whisper_small"` *(or whichever model folder you are using)*.
* **`output_dir`**: This **must** also point to the large drive so the VM doesn't crash:
`output_dir: "/mnt/hf_cache/my_custom_model_output"`
* **Memory Adjustments**: If you are using `whisper-small` or larger on the 24GB H100 slice, ensure `gradient_checkpointing: true` is set. If the script immediately crashes with a CUDA Out of Memory error, drop `train_batch_size` to `16` or `8` and increase `grad_accum_steps` proportionally.

**3. Start a Persistent Terminal Session**
To ensure the script doesn't die when you close your laptop or lose internet, start a virtual terminal session using `tmux`:

```bash
tmux new -s whisper_run

```

**4. Route System Storage to the Large Drive (CRITICAL)**
The VM's default root drive is extremely small (29GB). Inside your new `tmux` session, you **must** run these exact commands before starting the script to route all processing to the 246GB `/mnt` drive:

```bash
export TMPDIR="/mnt/tmp"
export HF_DATASETS_CACHE="/mnt/hf_cache"

```

**5. Start the Training**
Launch the script using the `uv` environment, pointing to the specific YAML file you created in Step 1:

```bash
uv run python train_whisper.py --config configs/my_run.yaml

```

**6. Detach and Log Out**
Once the script prints `Commencing CPU-based Seq2Seq Training...` (or you see the PyTorch progress bar moving), it is safe to leave.

1. Press and hold **`Ctrl`**, then press **`B`**.
2. Release both keys, then press **`D`**.
You will be detached from the session and can safely close your SSH connection.

**7. Check Progress Later**
To view the live training later, log back into the VM and reattach to your session:

```bash
tmux attach -t whisper_run

```

*(To detach again, just repeat `Ctrl+B`, then `D`)*.