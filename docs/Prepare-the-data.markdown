# Prepare the Data

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


## Using pip

```bash
# Download the Whisper-tiny model locally
hf download openai/whisper-tiny --local-dir ./local-whisper-tiny
```

## Using uv

```bash
# Download the Whisper-tiny model locally
uvx hf download openai/whisper-tiny --local-dir ./local-whisper-tiny
```

## In practice on altlab-gpu

This is how this was done on altlab-gpu. Replace `tiny base small medium large`
below with a list of the models you actually want. 

```bash
mkdir local_whisper_models
for model in tiny base small medium large; 
do hf download openai/whisper-$model --local-dir ./local_whisper_models/whisper-$model; 
done
```
