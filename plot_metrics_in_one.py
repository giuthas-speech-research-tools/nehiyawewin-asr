import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# Okabe-Ito colorblind-friendly palette
CB_PALETTE = [
    '#0072B2', '#E69F00', '#009E73', '#D55E00',
    '#56B4E9', '#CC79A7', '#F0E442', '#000000'
]


def plot_trainer_state(
    json_path: Path,
    ax1: plt.Axes,
    ax2: plt.Axes,
    ax3: plt.Axes,
    run_name: str,
    model_color: str,
    fallback_time: str = None,
) -> str:
    """Generate training curves from a Hugging Face trainer_state.json file.

    Returns:
        str: The extracted training time as a formatted string.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    df = pd.DataFrame(data["log_history"])

    # Extract training time from train_results.json if it exists
    train_results_path = json_path.parent / "train_results.json"
    time_str = "Unknown"
    if train_results_path.exists():
        with open(train_results_path, "r", encoding="utf-8") as f:
            res = json.load(f)
            if "train_runtime" in res:
                # Convert seconds to minutes for easier reading
                time_str = f"{res['train_runtime'] / 60:.1f}m"

    if time_str == "Unknown" and fallback_time:
        time_str = fallback_time

    # Extract the best model checkpoint path assigned by Early Stopping
    best_ckpt = data.get("best_model_checkpoint", "Unknown")

    # Training and Eval metrics are logged on different steps.
    # We separate them to avoid broken lines in the graph.
    train_df = df.dropna(subset=['loss'])
    eval_df = df.dropna(subset=['eval_loss'])

    # Pre-calculate best step and its corresponding metrics for legends/markers
    best_step = None
    best_wer = None
    best_cer = None
    if best_ckpt != "Unknown":
        try:
            best_step = int(Path(best_ckpt).name.split('-')[-1])
        except (ValueError, IndexError):
            pass

    if best_step is not None and not eval_df.empty:
        eval_row = eval_df[eval_df['step'] == best_step]
        if not eval_row.empty:
            if 'eval_wer' in eval_row.columns:
                best_wer = eval_row['eval_wer'].values[0]
            if 'eval_cer' in eval_row.columns:
                best_cer = eval_row['eval_cer'].values[0]

    # Set the model name as the title of its specific loss subplot
    ax1.set_title("Loss development: " + run_name)

    # Plot 1: Loss curves (Legends kept clean, without model name)
    ax1.plot(
        train_df['step'],
        train_df['loss'],
        label='Train Loss',
        color=model_color,
        linestyle='-'
    )

    if not eval_df.empty:
        ax1.plot(
            eval_df['step'],
            eval_df['eval_loss'],
            label='Eval Loss',
            color=model_color,
            linestyle='--'
        )

    # Plot 2: Error metrics (WER & CER - Legends include model name)
    if 'eval_wer' in eval_df.columns:
        wer_label = f'{run_name},'
        if best_wer is not None:
            wer_label += f' best model WER: {best_wer:.2f}'

        ax2.plot(
            eval_df['step'],
            eval_df['eval_wer'],
            label=wer_label,
            color=model_color,
            linestyle='-'
        )
        # Mark only the best checkpoint with a hollow black circle
        if best_step is not None and best_wer is not None:
            ax2.plot(
                best_step, best_wer,
                marker='o', markeredgecolor='black', markerfacecolor='none',
                linestyle='None'
            )

    if 'eval_cer' in eval_df.columns:
        cer_label = f'{run_name},'
        if best_cer is not None:
            cer_label += f' best model CER: {best_cer:.2f}'

        ax3.plot(
            eval_df['step'],
            eval_df['eval_cer'],
            label=cer_label,
            color=model_color,
            linestyle='-'
        )
        # Mark only the best checkpoint with a hollow black circle
        if best_step is not None and best_cer is not None:
            ax3.plot(
                best_step, best_cer,
                marker='o', markeredgecolor='black', markerfacecolor='none',
                linestyle='None'
            )

    # Annotate the specific data point matching the best model checkpoint
    if best_step is not None:
        eval_row = eval_df[eval_df['step'] == best_step]
        if not eval_row.empty:
            best_eval_loss = eval_row['eval_loss'].values[0]
            # Keep arrow annotation for Loss only
            ax1.annotate(
                "Best",
                xy=(best_step, best_eval_loss),
                xytext=(30, 30),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=model_color),
                color=model_color,
                fontsize=8
            )

    return time_str


def process_directories(base_dir: str) -> None:
    """Iterate through subdirectories and generate plots."""
    base_path = Path(base_dir)

    if not base_path.is_dir():
        print(f"Error: '{base_dir}' is not a valid directory.")
        sys.exit(1)

    order_file = base_path / "model_order.txt"
    if not order_file.exists():
        print(f"Error: 'model_order.txt' not found in {base_dir}.")
        sys.exit(1)

    # Read the directory names in the order provided by the text file and get
    # the fallback estimates for training time while you are at it.
    dir_order = []
    with open(order_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                dir_name = parts[0]
                fallback = parts[1] if len(parts) > 1 else None
                dir_order.append((dir_name, fallback))

    valid_dirs = []
    # Identify which directories exist and contain valid state files
    for subdir_name, fallback_time in dir_order:
        subdir = base_path / subdir_name
        if subdir.is_dir():
            json_path = subdir / "trainer_state.json"
            if json_path.exists():
                valid_dirs.append((subdir, fallback_time))
            else:
                print(
                    f"Warning: No 'trainer_state.json' found in {subdir.name}."
                )
        else:
            print(
                f"Warning: Directory '{subdir_name}' listed in order file "
                f"does not exist."
            )

    n_models = len(valid_dirs)
    if n_models == 0:
        print(
            f"No 'trainer_state.json' files found in the specified "
            f"directories of '{base_dir}'."
        )
        return

    # Calculate grid dimensions: 2 columns for loss, total 4 columns width
    n_loss_rows = (n_models + 1) // 2

    # Initialize a single figure to overlay all training runs
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2 * n_loss_rows, 4)

    # Dynamically generate loss subplots split into 2 columns
    loss_axes = []
    for i in range(n_models):
        r = i // 2
        c = i % 2
        if i == 0:
            ax = fig.add_subplot(gs[r * 2:(r + 1) * 2, c])
        else:
            # Share axes with the first plot to enforce identical limits
            ax = fig.add_subplot(
                gs[r * 2:(r + 1) * 2, c],
                sharex=loss_axes[0],
                sharey=loss_axes[0]
            )
        loss_axes.append(ax)

    # WER and CER subplots take up the right half of the space (columns 2 & 3)
    ax2 = fig.add_subplot(gs[:n_loss_rows, 2:])
    ax3 = fig.add_subplot(gs[n_loss_rows:, 2:])

    # Store info for the custom legend card
    model_info = []

    # Iterate through the valid subdirectories according to the provided order
    for i, (subdir, fallback_time) in enumerate(valid_dirs):
        print(f"Processing {subdir.name}...")
        json_path = subdir / "trainer_state.json"
        model_color = CB_PALETTE[i % len(CB_PALETTE)]

        t_str = plot_trainer_state(
            json_path, loss_axes[i], ax2, ax3, subdir.name, model_color,
            fallback_time
        )
        model_info.append((subdir.name, t_str, model_color))

    # Configure the shared axes once all data has been plotted
    for i, ax in enumerate(loss_axes):
        # Only set x-label on the bottom-most subplots to reduce clutter
        if i // 2 == n_loss_rows - 1:
            ax.set_xlabel('Steps')

        ax.set_ylabel('Loss')
        ax.legend(
            # fontsize='small'
        )
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim((-.25, 3.25))

    ax2.set_title('WER Development')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Percentage (%)')

    # Primary WER legend
    leg_wer = ax2.legend(
        # fontsize='small',
        loc='upper center'
    )
    ax2.add_artist(leg_wer)  # Add back so the second legend doesn't wipe it

    # Custom "Legend-like Card" for Model Training Times
    time_handles = [
        Line2D([0], [0], color=c, lw=2, label=f"{name}: {t_str}")
        for name, t_str, c in model_info
    ]
    ax2.legend(
        handles=time_handles,
        # fontsize='small',
        title='Training times',
        loc='upper right'
    )
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax3.set_title('CER Development')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Percentage (%)')
    ax3.legend(
        # fontsize='small'
    )
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_ylim((5, 45))

    plt.tight_layout()
    output_path = base_path / "combined_training_report.png"
    plt.savefig(output_path, dpi=300)

    # Crucial: Close the figure to free memory and prevent plot overlapping
    plt.close(fig)
    print(f"Graph saved successfully as '{output_path.name}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_metrics.py <path_to_parent_directory>")
        sys.exit(1)

    process_directories(sys.argv[1])
