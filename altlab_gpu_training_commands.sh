
git pull
tmux new -s whisper
export TMPDIR="/data/plains-cree-asr/tmp"
export HF_DATASETS_CACHE="/data/plains-cree-asr/hf_cache"
uv run python train_whisper.py --config configs/altlab-gpu-small.yaml 
