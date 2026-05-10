import os

from datasets import load_dataset, Audio, Features, Value
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
)


def main():
    print("Hello from whisper-test!")

    data_dir = "cree-asr/hindi/hi"

    # Define schema
    cv_features = Features({
        "client_id": Value("string"),
        "path": Value("string"),
        "sentence_id": Value("string"),
        "sentence": Value("string"),
        "sentence_domain": Value("string"),
        "up_votes": Value("int64"),
        "down_votes": Value("int64"),
        "age": Value("string"),
        "gender": Value("string"),
        "accents": Value("string"),
        "variant": Value("string"),
        "locale": Value("string"),
        "segment": Value("string"),
    })

    # Load dataset
    common_voice = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(data_dir, "train.tsv"),
            "test": os.path.join(data_dir, "test.tsv")
        },
        delimiter="\t",
        features=cv_features
    )

    # Fix paths and create audio column
    def prepare_audio_columns(example):
        full_path = os.path.join(data_dir, "clips", example["path"])
        example["path"] = full_path
        example["audio"] = full_path
        return example

    common_voice = common_voice.map(prepare_audio_columns)
    common_voice = common_voice.cast_column(
        "audio", Audio(sampling_rate=16000)
    )

    # ==========================================
    # SANITY CHECK
    # ==========================================
    print("\n--- Sanity Check: Path & Audio ---")
    sample = common_voice["train"][0]
    print(f"File Path: {sample['path']}")
    print(f"Audio Array Shape: {sample['audio']['array'].shape}")
    print(f"Sampling Rate: {sample['audio']['sampling_rate']}")
    print(f"Transcript: {sample['sentence']}")
    print("----------------------------------\n")

    # ==========================================
    # FEATURE EXTRACTION & TOKENIZATION
    # ==========================================
    print("Loading Processor, Feature Extractor, and Tokenizer...")

    # Use 'tiny' for local testing to save RAM and CPU cycles
    model_id = "openai/whisper-tiny"

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_id, language="Hindi", task="transcribe"
    )
    processor = WhisperProcessor.from_pretrained(
        model_id, language="Hindi", task="transcribe"
    )

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    print("Applying feature extraction and tokenization...")
    # Using num_proc=4 to utilize a few of your 16 CPU threads
    common_voice = common_voice.map(
        prepare_dataset,
        remove_columns=common_voice.column_names["train"],
        num_proc=4
    )

    print("\n--- Final Dataset Ready for Training ---")
    print(common_voice)

    # ==========================================
    # 3. TRAINING SETUP (1-Hour CPU PoC Mode)
    # ==========================================
    from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
    import torch
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    import evaluate

    print("\nSetting up Data Collator and Metrics...")

    # 3a. Data Collator (Pads audio and text to the same length in batches)
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [
                {"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt")

            label_features = [{"input_ids": feature["labels"]}
                              for feature in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt")

            # Replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100)

            # If bos token is appended in previous tokenization step, cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 3b. Load Metric (Word Error Rate)
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # 3c. Load the Model
    print("Loading Model Weights...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # The new, silent way to configure Whisper generation
    model.generation_config.language = "hindi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    # 3d. Training Arguments (PoC CPU Limits)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-tiny-hindi-finetuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=25,       # <--- SCALED DOWN: ~10% of new max_steps
        max_steps=250,         # <--- REDUCED: Will take ~40 mins at 9.54s/it
        eval_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=125,        # <--- REDUCED: Save halfway and at the end
        eval_steps=125,        # <--- REDUCED: Evaluate halfway and at the end
        logging_steps=25,      # <--- REDUCED: See logs more frequently
        report_to=["tensorboard"],
        use_cpu=True,
    )

    # --- CRITICAL PoC ADDITION ---
    # CPU text generation is extremely slow. We slice the test set down to
    # 50 samples so the two evaluation loops don't push the run over 1 hour.
    poc_test_dataset = common_voice["test"].select(
        range(min(50, len(common_voice["test"]))))

    # 3e. Initialize Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=poc_test_dataset,  # <--- Swapped to the truncated subset
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # ==========================================
    # 4. EXECUTE TRAINING
    # ==========================================
    print("\nStarting 1-Hour CPU PoC Training!")
    trainer.train()

    print("\nTraining Complete! Saving final PoC model...")
    trainer.save_model("./whisper-tiny-hindi-finetuned/final")
    processor.save_pretrained("./whisper-tiny-hindi-finetuned/final")


if __name__ == "__main__":
    main()
