"""
Training module for fine-tuning Whisper on Cree ASR data.

This script processes local dataset metadata, extracts audio
features, and runs a CPU-based Seq2Seq training loop.

Notes
-----
Cree is not a natively supported language in default Whisper,
so the language forcing configuration is disabled.
"""

import torch
import evaluate
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset, Audio, DatasetDict
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.trainer_utils import EvalPrediction

# Import our custom configuration
from config import config


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that pads audio inputs and text labels.

    Ensures that dynamically sized batches have matching
    sequence lengths by padding and appropriately masking
    loss calculations.

    Parameters
    ----------
    processor : Any
        The WhisperProcessor used for feature extraction and tokenization.

    Examples
    --------
    >>> collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    >>> batch = collator(features=[{"input_features": ...}])
    """
    processor: Any

    def __call__(
        self,
        features: list[dict[str, list[int] | torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """
        Pad inputs and labels to the maximum length in the batch.

        Parameters
        ----------
        features : list[dict[str, list[int] | torch.Tensor]]
            A list of dictionary examples containing features and labels.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing padded input features and labels.
        """
        # Isolate input features and pad them
        input_features: list[dict[str, Any]] = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]
        batch: dict[str, torch.Tensor] = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # Isolate labels and pad them
        label_features: list[dict[str, Any]] = [
            {"input_ids": feature["labels"]}
            for feature in features
        ]
        labels_batch: dict[str, torch.Tensor] = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        # Replace padding with -100 to correctly ignore in the loss function
        labels: torch.Tensor = labels_batch["input_ids"].masked_fill(
            mask=labels_batch.attention_mask.ne(1),
            value=-100
        )

        # Cut the beginning of sequence (BOS) token if previously appended
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main() -> None:
    """
    Execute the core data preparation and training loop.

    Loads the metadata CSV mapped by the wrap script, creates
    a deterministic split, applies feature extraction, and
    initiates the training process.
    """
    print("Loading datasets and setting up splits...")

    # Load dataset strictly using keyword arguments
    dataset_full = load_dataset(
        path="csv",
        data_files=config.metadata_csv,
        split="train"
    )

    # Cast the audio column to parse the WAV files appropriately
    dataset_full = dataset_full.cast_column(
        column="audio",
        feature=Audio(sampling_rate=config.sampling_rate)
    )

    # Split data to guarantee distinct train/test distributions
    dataset: DatasetDict = dataset_full.train_test_split(
        test_size=config.test_size,
        seed=config.random_seed
    )

    print("Initializing feature extractor, tokenizer, and processor...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path=config.base_model_id
    )

    # Removed language="Hindi" as Cree does not map natively.
    # Just specifying task="transcribe"
    tokenizer = WhisperTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.base_model_id,
        task="transcribe"
    )
    processor = WhisperProcessor.from_pretrained(
        pretrained_model_name_or_path=config.base_model_id,
        task="transcribe"
    )

    def prepare_dataset(batch: dict[str, Any]) -> dict[str, Any]:
        """
        Extract features from audio and tokenize transcriptions.

        Parameters
        ----------
        batch : dict[str, Any]
            A single batched item from the huggingface dataset.

        Returns
        -------
        dict[str, Any]
            The batch appended with 'input_features' and 'labels'.
        """
        audio = batch["audio"]

        # Extract mel spectrogram features
        extracted = feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        )
        batch["input_features"] = extracted.input_features[0]

        # Tokenize the SRO text transcript
        tokenized = tokenizer(text=batch["sentence"])
        batch["labels"] = tokenized.input_ids

        return batch

    print(f"Applying mapping via {config.num_processors} CPU threads...")
    dataset = dataset.map(
        function=prepare_dataset,
        remove_columns=dataset.column_names["train"],
        num_processors=config.num_processors
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load(path="wer")

    def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
        """
        Calculate the Word Error Rate during evaluation.

        Parameters
        ----------
        prediction : EvalPrediction
            Predictions and corresponding reference labels.

        Returns
        -------
        dict[str, float]
            A dictionary mapped with calculated WER percentage.
        """
        prediction_ids: torch.Tensor = prediction.predictions
        label_ids: torch.Tensor = prediction.label_ids

        # Replace -100 with the pad_token_id for correct decoding
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        prediction_str: list[str] = tokenizer.batch_decode(
            sequences=prediction_ids,
            skip_special_tokens=True
        )
        label_str: list[str] = tokenizer.batch_decode(
            sequences=label_ids,
            skip_special_tokens=True
        )

        wer_score: float = 100 * wer_metric.compute(
            predictions=prediction_str,
            references=label_str
        )
        return {"wer": wer_score}

    print("Loading Model Weights...")
    model = WhisperForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=config.base_model_id
    )

    # Disable language forcing for Cree specifically
    model.generation_config.language = None
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    print("Assembling Training Arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.train_batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        eval_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        report_to=["tensorboard"],
        use_cpu=config.use_cpu,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("Commencing CPU-based Seq2Seq Training...")
    trainer.train()

    print("Training Completed. Saving Model Weights and Configs...")
    trainer.save_model(output_dir=config.output_dir)
    processor.save_pretrained(save_directory=config.output_dir)


if __name__ == "__main__":
    main()
