import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import click
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Okabe-Ito colorblind-friendly palette
CB_PALETTE = [
    '#0072B2', '#E69F00', '#009E73', '#D55E00',
    '#56B4E9', '#CC79A7', '#F0E442', '#000000'
]


@dataclass
class ModelMetrics:
    """
    Holds parsed training metrics and metadata for a single model.

    Attributes
    ----------
    name : str
        The name of the model run or subdirectory.
    color : str
        The hex color code assigned for plotting.
    time_str : str
        The formatted string representing the training duration.
    train_df : pd.DataFrame
        DataFrame containing the training loss data.
    eval_df : pd.DataFrame
        DataFrame containing the evaluation metrics data.
    best_step : int, optional
        The step number corresponding to the best evaluation checkpoint.
    best_wer : float, optional
        The best Word Error Rate recorded at the best checkpoint.
    best_cer : float, optional
        The best Character Error Rate recorded at the best checkpoint.
    """
    name: str
    color: str
    time_str: str
    train_df: pd.DataFrame
    eval_df: pd.DataFrame
    best_step: Optional[int]
    best_wer: Optional[float]
    best_cer: Optional[float]


def extract_model_metrics(
    json_path: Path,
    run_name: str,
    color: str,
    fallback_time: Optional[str]
) -> ModelMetrics:
    """
    Parses JSON logs to extract training DataFrames and best metrics.

    Parameters
    ----------
    json_path : Path
        The file path to the trainer_state.json file.
    run_name : str
        The name of the training run, typically the directory name.
    color : str
        The color assigned to this model for visualizations.
    fallback_time : str, optional
        A fallback training time estimate to use if the JSON does not contain
        the 'train_runtime' key.

    Returns
    -------
    ModelMetrics
        A data class containing all extracted DataFrames and calculated metrics
        for the given model.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    df = pd.DataFrame(data.get("log_history", []))

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

    return ModelMetrics(
        name=run_name,
        color=color,
        time_str=time_str,
        train_df=train_df,
        eval_df=eval_df,
        best_step=best_step,
        best_wer=best_wer,
        best_cer=best_cer
    )


def generate_plots(models: List[ModelMetrics], output_path: Path) -> None:
    """
    Generates and saves the Matplotlib figure showing training curves.

    Parameters
    ----------
    models : List[ModelMetrics]
        A list of ModelMetrics objects containing the plotting data and
        metadata for all evaluated models.
    output_path : Path
        The destination file path where the generated figure will be saved.
    """
    n_models = len(models)

    # Calculate grid dimensions: 2 columns for loss, total 4 columns width
    # We add 2 instead of 1 to ensure space for the initial legend slot
    n_loss_rows = (n_models + 2) // 2

    # Initialize a single figure to overlay all training runs
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2 * n_loss_rows, 4)

    # First slot (top left) reserved strictly for the legend
    ax_legend = fig.add_subplot(gs[0:2, 0])
    ax_legend.axis('off')

    # Dynamically generate loss subplots split into 2 columns
    loss_axes = []
    for i in range(n_models):
        slot = i + 1  # Shift by 1 to leave the 0th slot for the legend
        r = slot // 2
        c = slot % 2

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

    # Plot data for each model
    for ax1, model in zip(loss_axes, models):
        # Set the model name as the title of its specific loss subplot
        ax1.set_title(model.name)

        # Plot 1: Loss curves (Legends kept clean, without model name)
        if not model.eval_df.empty:
            ax1.plot(
                model.eval_df['step'], model.eval_df['eval_loss'],
                label='Evaluation Loss', color=model.color, linestyle='-'
            )

        ax1.plot(
            model.train_df['step'], model.train_df['loss'],
            label='Training Loss', color=model.color, linestyle='--'
        )

        # Plot 2: Error metrics (WER & CER - Legends include model name)
        if 'eval_wer' in model.eval_df.columns:
            wer_label = f"{model.name},"
            if model.best_wer is not None:
                wer_label += f" best model WER: {model.best_wer:.2f}"

            ax2.plot(
                model.eval_df['step'], model.eval_df['eval_wer'],
                label=wer_label, color=model.color, linestyle='-'
            )

            # Mark only the best checkpoint with a hollow black circle
            if model.best_step is not None and model.best_wer is not None:
                ax2.plot(
                    model.best_step, model.best_wer,
                    marker='o', markeredgecolor='black',
                    markerfacecolor='none', linestyle='None'
                )

        if 'eval_cer' in model.eval_df.columns:
            cer_label = f"{model.name},"
            if model.best_cer is not None:
                cer_label += f" best model CER: {model.best_cer:.2f}"

            ax3.plot(
                model.eval_df['step'], model.eval_df['eval_cer'],
                label=cer_label, color=model.color, linestyle='-'
            )

            # Mark only the best checkpoint with a hollow black circle
            if model.best_step is not None and model.best_cer is not None:
                ax3.plot(
                    model.best_step, model.best_cer,
                    marker='o', markeredgecolor='black',
                    markerfacecolor='none', linestyle='None'
                )

        # Annotate the specific data point matching the best model checkpoint
        if model.best_step is not None:
            eval_row = model.eval_df[model.eval_df['step'] == model.best_step]
            if not eval_row.empty:
                best_eval_loss = eval_row['eval_loss'].values[0]
                # Keep arrow annotation for Loss only
                ax1.annotate(
                    "Best", xy=(model.best_step, best_eval_loss),
                    xytext=(30, 30), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color=model.color),
                    color=model.color, fontsize=8
                )

    # Configure the shared axes once all data has been plotted
    for i, ax in enumerate(loss_axes):
        # Only set x-label on the bottom-most subplots to reduce clutter If the
        # slot 2 spots directly below this one exceeds n_models, it's at the
        # bottom
        slot = i + 1
        if slot + 2 > n_models:
            ax.set_xlabel('Steps')

        ax.set_ylabel('Loss')
        ax.legend(
            # fontsize='small'
        )
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim((-.25, 3.25))

    ax2.set_title('Word Error Rate (WER)')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Percentage (%)')

    # Primary WER legend
    leg_wer = ax2.legend(
        # fontsize='small',
        loc='upper right'
    )
    ax2.add_artist(leg_wer)  # Add back so the second legend doesn't wipe it
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax3.set_title('Character Error Rate (CER)')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Percentage (%)')
    ax3.legend(
        # fontsize='small'
    )
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_ylim((5, 45))

    # Custom "Legend-like Card" for Model Training Times
    time_handles = [
        Line2D([0], [0], color=m.color, lw=2, label=f"{m.name}: {m.time_str}")
        for m in models
    ]

    # Place the legend in the reserved first subplot slot
    ax_legend.legend(
        handles=time_handles,
        fontsize='large',
        title='Training times',
        title_fontsize='x-large',
        loc='upper center'  # Center it beautifully in the first subplot
    )

    # rect=[left, bottom, right, top] leaves a 4% margin at the top for the
    # title
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Place text at roughly 25% of the figure width (center of the first two
    # columns)
    fig.text(
        0.25, 0.98, 'Loss development',
        ha='center', va='top', fontsize=16, fontweight='bold'
    )
    fig.text(
        0.77, 0.98, 'Error rate development',
        ha='center', va='top', fontsize=16, fontweight='bold'
    )

    plt.savefig(output_path, dpi=300)

    # Crucial: Close the figure to free memory and prevent plot overlapping
    plt.close(fig)
    print(f"Graph saved successfully as '{output_path.name}'")


def generate_latex_table(
    models: List[ModelMetrics],
    output_path: Path
) -> None:
    """
    Generates a LaTeX table summarizing model performance.

    Parameters
    ----------
    models : List[ModelMetrics]
        A list of ModelMetrics objects containing the performance metrics
        for all evaluated models.
    output_path : Path
        The destination file path where the generated .tex file will be saved.
    """
    valid_models = [
        (i, m.best_wer, m.best_cer)
        for i, m in enumerate(models)
        if m.best_wer is not None and m.best_cer is not None
    ]

    best_idx = -1
    if valid_models:
        best_idx = min(valid_models, key=lambda x: x[1] + x[2])[0]

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"\textbf{Model} & \textbf{Training Time} & "
        r"\textbf{Best WER (\%)} & \textbf{Best CER (\%)} \\",
        r"\hline"
    ]

    for i, model in enumerate(models):
        wer_str = f"{model.best_wer:.2f}" if model.best_wer else "N/A"
        cer_str = f"{model.best_cer:.2f}" if model.best_cer else "N/A"
        safe_name = model.name.replace("_", r"\_")

        if i == best_idx:
            row = (
                rf"\textbf{{{safe_name}}} & \textbf{{{model.time_str}}} & "
                rf"\textbf{{{wer_str}}} & \textbf{{{cer_str}}} \\"
            )
        else:
            row = rf"{safe_name} & {model.time_str} & {wer_str} & {cer_str} \\"

        latex_lines.append(row)

    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{Model Performance Comparison}",
        r"\label{tab:model_comparison}",
        r"\end{table}"
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines) + "\n")
    print(f"LaTeX table saved successfully as '{output_path.name}'")


def process_directories(
    base_dir: str,
    generate_plot: bool,
    generate_latex: bool,
    plot_filename: str,
    latex_filename: str
) -> None:
    """
    Iterate over subdirectories to generate plots, and/or create a LaTeX table.

    Parameters
    ----------
    base_dir : str
        The root directory containing the ordered model subdirectories.
    generate_plot : bool
        Flag indicating whether the matplotlib figure should be created.
    generate_latex : bool
        Flag indicating whether the LaTeX comparison table should be created.
    plot_filename : str
        The filename to use when saving the matplotlib figure.
    latex_filename : str
        The filename to use when saving the LaTeX table.
    """
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

    # Store info for the custom legend card and LaTeX table
    models: List[ModelMetrics] = []

    # Iterate through the valid subdirectories according to the provided order
    for i, (subdir, fallback_time) in enumerate(valid_dirs):
        print(f"Processing {subdir.name}...")
        json_path = subdir / "trainer_state.json"
        color = CB_PALETTE[i % len(CB_PALETTE)]

        metrics = extract_model_metrics(
            json_path, subdir.name, color, fallback_time
        )
        models.append(metrics)

    # Phase 1: Output Matplotlib Figure
    if generate_plot:
        plot_path = base_path / plot_filename
        generate_plots(models, plot_path)

    # Phase 2: Output LaTeX Table
    if generate_latex:
        tex_path = base_path / latex_filename
        generate_latex_table(models, tex_path)


@click.command()
@click.argument(
    "base_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--plot/--no-plot",
    default=False,
    help="Toggle generation of the matplotlib figure. Default is --no-plot."
)
@click.option(
    "--latex/--no-latex",
    default=True,
    help="Toggle generation of the LaTeX table. Default is --latex."
)
@click.option(
    "--plot-file",
    "plot_filename",
    default="combined_training_report.png",
    help="Filename for the output plot."
)
@click.option(
    "--latex-file",
    "latex_filename",
    default="model_comparison.tex",
    help="Filename for the output LaTeX table."
)
def main(
    base_dir: str,
    plot: bool,
    latex: bool,
    plot_filename: str,
    latex_filename: str
) -> None:
    """
    Parses Hugging Face training logs and outputs comparison artifacts.

    BASE_DIR is the root directory containing model subdirectories and a
    'model_order.txt' file.
    """
    process_directories(
        base_dir=base_dir,
        generate_plot=plot,
        generate_latex=latex,
        plot_filename=plot_filename,
        latex_filename=latex_filename
    )


if __name__ == "__main__":
    main()
