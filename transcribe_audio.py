import argparse
from pathlib import Path

import torch
import torchaudio
from transformers import pipeline


def transcribe_audio(model_dir: str, audio_path: str) -> str:
    """
    Transcribe an audio file using a local Hugging Face ASR model.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    print(f"Loading model from {model_dir}...")

    # Initialize the automatic-speech-recognition pipeline
    transcriber = pipeline(
        task="automatic-speech-recognition",
        model=model_dir,
        tokenizer=model_dir,
        feature_extractor=model_dir,
        device=device,
        dtype=compute_dtype,
        model_kwargs={"use_cache": True},  # Re-enables fast inference
        chunk_length_s=30,  # For transcribing files longer than 30 seconds
        batch_size=8,  # Speeds up long-file processing by batching chunks
    )

    transcriber.tokenizer.pad_token_id = transcriber.tokenizer.eos_token_id

    print(f"Transcribing {audio_path}...")

    # Load with torchaudio to bypass Hugging Face's file streaming bug
    waveform, sampling_rate = torchaudio.load(audio_path)

    # Convert to mono if the recording is stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Whisper requires 16kHz. If the pipeline fails to resample 44.1kHz
    # audio internally, the model receives noise and outputs nothing.
    if sampling_rate != 16000:
        print(f"Resampling from {sampling_rate} Hz to 16000 Hz.")
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sampling_rate, new_freq=16000
        )
        sampling_rate = 16000

    audio_array = waveform.squeeze().numpy()

    # Run the raw audio array through the model
    result = transcriber(
        inputs={"array": audio_array, "sampling_rate": sampling_rate},
        return_timestamps=False,  # Bypasses the broken chunk-stitching logic
        generate_kwargs={
            "language": None,
            "task": "transcribe",
            "forced_decoder_ids": None,
            "suppress_tokens": None,
            "max_length": 225,
        },
    )

    return result["text"]


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
