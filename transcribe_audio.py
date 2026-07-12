import argparse
from pathlib import Path

import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def transcribe_audio(model_dir: str, audio_path: str) -> str:
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    print(f"Loading model from {model_dir}...")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir
    ).to(device)
    processor = WhisperProcessor.from_pretrained(model_dir)

    # EXACTLY mirror your training script overrides.
    # We must also clear model.config to fix the logits processor warning
    # that silently swallows valid token generation.
    model.config.suppress_tokens = None
    model.generation_config.suppress_tokens = None
    model.generation_config.language = None
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.pad_token_id = (
        processor.tokenizer.pad_token_id
    )

    print(f"Transcribing {audio_path}...")
    waveform, sampling_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sampling_rate != 16000:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sampling_rate,
            new_freq=16000
        )
        sampling_rate = 16000

    audio_array = waveform.squeeze().numpy()
    chunk_len = 16000 * 30
    results = []

    for start in range(0, len(audio_array), chunk_len):
        chunk = audio_array[start: start + chunk_len]

        processed = processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = processed.input_features.to(device)

        with torch.no_grad():
            prediction_ids = model.generate(
                input_features,
                max_length=225,
            )

        # Extract tokens for this chunk
        raw_tokens = prediction_ids[0].tolist()

        # Filter out special tokens (Whisper specials are always >= 50257)
        text_tokens = []
        for token in raw_tokens:
            if token < 50257:
                text_tokens.append(token)

        # Decode only the pure text content tokens
        clean_text = processor.tokenizer.decode(text_tokens)

        if clean_text.strip():
            results.append(clean_text.strip())

    return " ".join(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe audio using a local ASR model."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the saved model directory.",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to the audio file or directory to be transcribed.",
    )

    args = parser.parse_args()
    input_path = Path(args.audio_path)

    # Define the broader set of supported audio extensions
    supported_extensions = {".wav", ".m4a", ".mp3", ".flac", ".ogg"}

    if input_path.is_file():
        audio_files = [input_path]
    elif input_path.is_dir():
        audio_files = [
            f for f in input_path.rglob("*")
            if f.suffix.lower() in supported_extensions
        ]
    else:
        raise ValueError(f"Invalid path provided: {input_path}")

    for file_path in audio_files:
        text = transcribe_audio(
            model_dir=args.model_dir,
            audio_path=str(file_path),
        )

        print(f"\n--- Transcription for {file_path.name} ---")
        print(text)
