import argparse
import sys
from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def transcribe_file(
    file_path: Path,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: str
) -> str:
    """
    Loads an audio file, resamples it to 16kHz, and transcribes it.

    Args:
        file_path (Path): The path to the audio file to transcribe.
        processor (WhisperProcessor): The loaded Hugging Face processor.
        model (WhisperForConditionalGeneration): The trained Whisper model.
        device (str): The device to perform computation on ('cuda' or 'cpu').

    Returns:
        str: The generated transcription text.
    """
    # AudioDecoder uses FFmpeg under the hood to automatically resample to
    # 16kHz and convert arbitrary audio formats/channels to a mono stream.
    decoder = AudioDecoder(
        str(file_path),
        sample_rate=16000,
        num_channels=1
    )
    audio_samples = decoder.get_all_samples()
    audio_array = audio_samples.data.squeeze().numpy()

    inputs = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(inputs, max_length=225)

    prediction = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]

    return prediction


def main() -> None:
    """
    Parses command-line arguments, loads the Whisper model, and processes
    the provided audio file(s) for transcription. Transcriptions are either
    printed to stdout or saved to .sro files depending on the arguments.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio using a fine-tuned Whisper model."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a single audio file or a directory of audio files."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="hf_cache/whisper-medium-finetuned/final",
        help="Path to the trained model directory."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write transcriptions to .sro files in a new directory."
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Path '{input_path}' does not exist.")
        sys.exit(1)

    # 1. Gather audio files
    if input_path.is_file():
        audio_files = [input_path]
    else:
        # Ignore hidden files or directories
        audio_files = [
            p for p in input_path.iterdir()
            if p.is_file() and not p.name.startswith(".")
        ]

    if not audio_files:
        print(f"No files found in '{input_path}'.")
        sys.exit(1)

    # 2. Configure output directory if saving
    out_dir = None
    if args.save:
        out_dir_name = f"plains_cree_transcription_{input_path.name}"
        out_dir = input_path.parent / out_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Transcriptions will be saved to: {out_dir}")

    # 3. Load model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from '{args.model_dir}' on {device}...")

    try:
        processor = WhisperProcessor.from_pretrained(args.model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_dir
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model or processor: {e}")
        sys.exit(1)

    # 4. Transcribe
    print(f"Found {len(audio_files)} file(s). Starting transcription...")
    for file_path in tqdm(audio_files, desc="Transcribing"):
        try:
            transcription = transcribe_file(
                file_path, processor, model, device
            )

            if args.save:
                out_file = out_dir / f"{file_path.stem}.sro"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(transcription + "\n")
            else:
                print(f"\n--- {file_path.name} ---\n{transcription}\n")

        except Exception as e:
            print(f"Error processing '{file_path.name}': {e}")


if __name__ == "__main__":
    main()
