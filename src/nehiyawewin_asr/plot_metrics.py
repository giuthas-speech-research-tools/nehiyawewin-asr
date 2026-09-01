import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_trainer_state(json_path: Path, output_path: Path) -> None:
    """Generate training curves from a Hugging Face trainer_state.json file."""
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    df = pd.DataFrame(data["log_history"])

    # Training and Eval metrics are logged on different steps.
    # We separate them to avoid broken lines in the graph.
    train_df = df.dropna(subset=['loss'])
    eval_df = df.dropna(subset=['eval_loss'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Loss curves
    ax1.plot(train_df['step'], train_df['loss'],
             label='Train Loss', color='blue')
    if not eval_df.empty:
        ax1.plot(eval_df['step'], eval_df['eval_loss'],
                 label='Eval Loss', color='orange')

    ax1.set_title('Loss Development')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Error metrics (WER & CER)
    if 'eval_wer' in eval_df.columns and 'eval_cer' in eval_df.columns:
        ax2.plot(eval_df['step'], eval_df['eval_wer'],
                 label='WER', color='red', marker='o')
        ax2.plot(eval_df['step'], eval_df['eval_cer'],
                 label='CER', color='green', marker='s')

        ax2.set_title('Error Rates')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Percentage (%)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
    else:
        # Gracefully handle older checkpoints that might not have WER/CER yet
        ax2.set_title('Error Rates (Not Yet Logged)')
        ax2.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    # Crucial: Close the figure to free memory and prevent plot overlapping in the loop
    plt.close(fig)
    print(f"Graph saved successfully as '{output_path.name}'")


def process_directories(base_dir: str) -> None:
    """Iterate through subdirectories and generate plots for trainer_state.json files."""
    base_path = Path(base_dir)

    if not base_path.is_dir():
        print(f"Error: '{base_dir}' is not a valid directory.")
        sys.exit(1)

    found_any = False

    # Iterate through all immediate subdirectories
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            json_path = subdir / "trainer_state.json"

            if json_path.exists():
                found_any = True
                print(f"Processing {subdir.name}...")

                # Save the PNG in the base directory using the subdirectory's name
                output_name = f"{subdir.name}_training_report.png"
                output_path = base_path / output_name

                plot_trainer_state(json_path, output_path)

    if not found_any:
        print(
            f"No 'trainer_state.json' files found in any subdirectories of '{base_dir}'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_metrics.py <path_to_parent_directory>")
        sys.exit(1)

    process_directories(sys.argv[1])
