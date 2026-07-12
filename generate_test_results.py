import os
import csv
import torch
from tqdm import tqdm
from config import config
from datasets import load_dataset, Audio, Features, Value
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def main() -> None:
    # 1. Re-create the exact same dataset split used during training
    print("Loading test dataset splits...")
    features = Features({
        "audio": Audio(sampling_rate=config.sampling_rate),
        "sentence": Value("string")
    })
    
    dataset_full = load_dataset(
        path="csv",
        data_files=config.metadata_csv,
        split="train",
        features=features
    )
    
    dataset = dataset_full.train_test_split(
        test_size=config.test_size,
        seed=config.random_seed
    )
    test_dataset = dataset["test"]

    # 2. Load the trained model and processor
    model_path = config.final_model_dir
    print(f"Loading model weights from {model_path}...")
    
    device = "cuda" if torch.cuda.is_available() and not getattr(config, "use_cpu", False) else "cpu"
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()

    # 3. Run inference on the test split
    print("Running inference on the test set...")
    results = []
    
    for item in tqdm(test_dataset, desc="Transcribing"):
        audio = item["audio"]
        reference = item["sentence"]
        # Extract the original string path if available
        audio_path = audio.get("path", "unknown_path")

        # Process the raw audio array
        inputs = processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"], 
            return_tensors="pt"
        ).input_features.to(device)

        # Generate prediction
        with torch.no_grad():
            predicted_ids = model.generate(inputs, max_length=225)
        
        # Decode output
        prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        results.append([audio_path, reference, prediction])

    # 4. Save to TSV
    output_file = os.path.join(model_path, "test_results.tsv")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["audio_path", "reference_sro", "predicted_sro"])
        writer.writerows(results)

    print(f"Test results successfully exported to {output_file}")

if __name__ == "__main__":
    main()