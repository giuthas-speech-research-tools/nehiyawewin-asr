"""
Whisper ASR Comprehensive Error Analysis Tool.

Features:
- Custom phonetic tokenization (handles â <-> aa / ah).
- Flags transcriptions with >10% Character Error Rate (CER).
- Exports detailed word and character error examples.
- Exports complete frequency tables to CSV and LaTeX.
- Console output filters out single-occurrence anomalies.
"""

import csv
import sys
from collections import Counter
from typing import Dict, List, Set, Tuple

import click
import jiwer


def load_tsv_data(file_path: str) -> List[Dict[str, str]]:
    """
    Load ASR test results from a TSV file.

    Parameters
    ----------
    file_path : str
        The path to the input TSV file.

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries representing the rows of the TSV file.

    Raises
    ------
    KeyError
        If the file is empty or lacks the required columns.
    """
    data = []
    with open(file_path, mode="r", encoding="utf-8") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        if not reader.fieldnames:
            raise KeyError("The TSV file is empty or lacks headers.")

        required = {"audio_path", "reference_sro", "predicted_sro"}
        if not required.issubset(set(reader.fieldnames)):
            raise KeyError(f"TSV must contain columns: {required}")

        for row in reader:
            data.append({
                "audio_path": row["audio_path"],
                "reference": row["reference_sro"],
                "hypothesis": row["predicted_sro"],
            })
    return data


def tokenize_phonetics(text: str) -> List[str]:
    """
    Tokenize string into characters, grouping Algonquian digraphs.

    Converts spaces to '<space>' to prevent JiWER from dropping them.
    Normalizes long vowel digraphs to their preferred circumflex forms.

    Parameters
    ----------
    text : str
        The input text to tokenize.

    Returns
    -------
    List[str]
        A list of phonetic tokens.
    """
    replacements = [
        # a variants
        ("aa", "â"), ("ah", "â"), ("aA", "â"), ("aH", "â"),
        ("Aa", "Â"), ("Ah", "Â"), ("AA", "Â"), ("AH", "Â"),
        # e variants
        ("ee", "ê"), ("eh", "ê"), ("eE", "ê"), ("eH", "ê"),
        ("Ee", "Ê"), ("Eh", "Ê"), ("EE", "Ê"), ("EH", "Ê"),
        # i variants
        ("ii", "î"), ("ih", "î"), ("iI", "î"), ("iH", "î"),
        ("Ii", "Î"), ("Ih", "Î"), ("II", "Î"), ("IH", "Î"),
        # o variants
        ("oo", "ô"), ("oh", "ô"), ("oO", "ô"), ("oH", "ô"),
        ("Oo", "Ô"), ("Oh", "Ô"), ("OO", "Ô"), ("OH", "Ô"),
        # u variants
        ("uu", "û"), ("uh", "û"), ("uU", "û"), ("uH", "û"),
        ("Uu", "Û"), ("Uh", "Û"), ("UU", "Û"), ("UH", "Û"),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    tokens = []
    i = 0
    n = len(text)
    lower_text = text.lower()

    # Vowel digraphs are now handled by normalizations above. We leave an empty
    # set in case you want to add consonant digraphs (e.g., "ch", "sh") later.
    digraphs = set()

    while i < n:
        if i < n - 1 and lower_text[i:i + 2] in digraphs:
            tokens.append(text[i:i + 2])
            i += 2
        else:
            char = text[i]
            if char == " ":
                tokens.append("<space>")
            else:
                tokens.append(char)
            i += 1
    return tokens


def tex_escape(text: str) -> str:
    """
    Escape basic characters for LaTeX rendering.

    Parameters
    ----------
    text : str
        The raw text string.

    Returns
    -------
    str
        The LaTeX-escaped string.
    """
    return (
        text.replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def extract_character_errors(
    ref: str, hyp: str, audio_path: str, high_error_flag: str
) -> Tuple[Dict[str, int], List[List[str]], Dict[str, Counter]]:
    """
    Analyze ASR errors at a granular token/character level.

    Parameters
    ----------
    ref : str
        The reference transcript.
    hyp : str
        The predicted transcript.
    audio_path : str
        The path to the audio file.
    high_error_flag : str
        "Yes" if CER > 10%, "No" otherwise.

    Returns
    -------
    Tuple[Dict[str, int], List[List[str]], Dict[str, Counter]]
        A tuple containing the error tally, example rows, and frequencies.
    """
    tally = {"substitute": 0, "insert": 0, "delete": 0}
    freqs = {"substitute": Counter(), "insert": Counter(), "delete": Counter()}
    examples = []

    ref_toks = tokenize_phonetics(ref)
    hyp_toks = tokenize_phonetics(hyp)

    ref_str = " ".join(ref_toks)
    hyp_str = " ".join(hyp_toks)

    result = jiwer.process_words(ref_str, hyp_str)

    for alignment in result.alignments[0]:
        if alignment.type == "equal":
            continue

        r_start, r_end = alignment.ref_start_idx, alignment.ref_end_idx
        h_start, h_end = alignment.hyp_start_idx, alignment.hyp_end_idx

        ref_chunk = ref_toks[r_start:r_end]
        hyp_chunk = hyp_toks[h_start:h_end]

        if alignment.type == "substitute":
            r_sub = "".join(ref_chunk)
            h_sub = "".join(hyp_chunk)
            tally["substitute"] += 1
            freqs["substitute"][(r_sub, h_sub)] += 1
            examples.append([
                audio_path, high_error_flag, "Substitution",
                r_sub, h_sub, "", "", ref, hyp
            ])

        elif alignment.type == "insert":
            for h_t in hyp_chunk:
                tally["insert"] += 1
                freqs["insert"][h_t] += 1
                examples.append([
                    audio_path, high_error_flag, "Insertion",
                    "", "", h_t, "", ref, hyp
                ])

        elif alignment.type == "delete":
            for r_t in ref_chunk:
                tally["delete"] += 1
                freqs["delete"][r_t] += 1
                examples.append([
                    audio_path, high_error_flag, "Deletion",
                    "", "", "", r_t, ref, hyp
                ])

    return tally, examples, freqs


def extract_word_errors(
    ref: str, hyp: str, audio_path: str, high_error_flag: str
) -> Tuple[Dict[str, int], List[List[str]]]:
    """
    Analyze ASR errors at the word level.

    Parameters
    ----------
    ref : str
        The reference transcript.
    hyp : str
        The predicted transcript.
    audio_path : str
        The path to the audio file.
    high_error_flag : str
        "Yes" if CER > 10%, "No" otherwise.

    Returns
    -------
    Tuple[Dict[str, int], List[List[str]]]
        A tuple containing the error tally and example rows.
    """
    tally = {"substitute": 0, "insert": 0, "delete": 0}
    examples = []

    result = jiwer.process_words(ref, hyp)
    ref_words = ref.split()
    hyp_words = hyp.split()

    for alignment in result.alignments[0]:
        if alignment.type == "equal":
            continue

        r_start, r_end = alignment.ref_start_idx, alignment.ref_end_idx
        h_start, h_end = alignment.hyp_start_idx, alignment.hyp_end_idx

        ref_err = " ".join(ref_words[r_start:r_end])
        hyp_err = " ".join(hyp_words[h_start:h_end])

        tally[alignment.type] += 1

        char_error_types: Set[str] = set()
        if alignment.type == "substitute":
            r_str = " ".join(tokenize_phonetics(ref_err))
            h_str = " ".join(tokenize_phonetics(hyp_err))
            char_res = jiwer.process_words(r_str, h_str)
            for c_align in char_res.alignments[0]:
                if c_align.type != "equal":
                    char_error_types.add(c_align.type)
        else:
            char_error_types.add(alignment.type)

        char_types_str = ", ".join(sorted(list(char_error_types)))

        if alignment.type == "substitute":
            examples.append([
                audio_path, high_error_flag, "Substitution",
                ref_err, hyp_err, "", "", char_types_str, ref, hyp
            ])
        elif alignment.type == "insert":
            examples.append([
                audio_path, high_error_flag, "Insertion",
                "", "", hyp_err, "", char_types_str, ref, hyp
            ])
        elif alignment.type == "delete":
            examples.append([
                audio_path, high_error_flag, "Deletion",
                "", "", "", ref_err, char_types_str, ref, hyp
            ])

    return tally, examples


def analyze_all_errors(
    data: List[Dict[str, str]]
) -> Tuple[Dict, List, Dict, List, Dict]:
    """
    Process all transcription records and aggregate errors.

    Parameters
    ----------
    data : List[Dict[str, str]]
        The list of dictionaries containing ASR transcriptions.

    Returns
    -------
    Tuple[Dict, List, Dict, List, Dict]
        Aggregated results: word tally, word examples, character tally,
        character examples, and character error frequencies.
    """
    w_tally = {"substitute": 0, "insert": 0, "delete": 0}
    c_tally = {"substitute": 0, "insert": 0, "delete": 0}
    c_freqs = {
        "substitute": Counter(),
        "insert": Counter(),
        "delete": Counter()
    }
    w_examples = []
    c_examples = []

    for item in data:
        ref = item["reference"]
        hyp = item["hypothesis"]
        audio_path = item["audio_path"]

        if not ref and not hyp:
            continue

        ref_toks = tokenize_phonetics(ref)
        hyp_toks = tokenize_phonetics(hyp)
        ref_str = " ".join(ref_toks)
        hyp_str = " ".join(hyp_toks)
        cer_result = jiwer.process_words(ref_str, hyp_str)

        cer = (
            cer_result.wer if len(ref_toks) > 0
            else (1.0 if len(hyp_toks) > 0 else 0.0)
        )
        high_error_flag = "Yes" if cer > 0.10 else "No"

        c_counts, c_rows, c_row_freqs = extract_character_errors(
            ref, hyp, audio_path, high_error_flag
        )
        for k, v in c_counts.items():
            c_tally[k] += v
        for k, counter in c_row_freqs.items():
            c_freqs[k].update(counter)
        c_examples.extend(c_rows)

        w_counts, w_rows = extract_word_errors(
            ref, hyp, audio_path, high_error_flag
        )
        for k, v in w_counts.items():
            w_tally[k] += v
        w_examples.extend(w_rows)

    return w_tally, w_examples, c_tally, c_examples, c_freqs


def export_examples_to_csv(
    examples: List[List[str]], output_path: str, headers: List[str]
) -> None:
    """
    Export error examples to a CSV file.

    Parameters
    ----------
    examples : List[List[str]]
        The list of rows to write.
    output_path : str
        The path for the output CSV file.
    headers : List[str]
        The column headers for the CSV file.
    """
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(examples)


def export_frequencies_to_csv(
    freqs: Dict[str, Counter], output_path: str
) -> None:
    """
    Export complete sorted frequencies of character errors to a CSV.

    Parameters
    ----------
    freqs : Dict[str, Counter]
        The dictionary of frequency counters.
    output_path : str
        The path for the output CSV file.
    """
    headers = [
        "Error Type", "Reference Sound(s)", "Predicted Sound(s)", "Count"
    ]
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for (r_char, h_char), count in freqs["substitute"].most_common():
            writer.writerow(["Substitution", r_char, h_char, count])

        for h_char, count in freqs["insert"].most_common():
            writer.writerow(["Insertion", "", h_char, count])

        for r_char, count in freqs["delete"].most_common():
            writer.writerow(["Deletion", r_char, "", count])


def export_frequencies_to_latex(
    freqs: Dict[str, Counter], output_path: str
) -> None:
    """
    Export complete error frequency tables as a LaTeX document.

    Parameters
    ----------
    freqs : Dict[str, Counter]
        The dictionary of frequency counters.
    output_path : str
        The path for the output LaTeX file.
    """
    with open(output_path, mode="w", encoding="utf-8") as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\begin{document}\n\n")

        # Substitutions
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{llc}\n")
        f.write("\\toprule\n")
        f.write("Reference & Predicted & Frequency \\\\\n")
        f.write("\\midrule\n")
        for (r, p), count in freqs["substitute"].most_common():
            f.write(f"{tex_escape(r)} & {tex_escape(p)} & {count} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Character Substitutions}\n")
        f.write("\\end{table}\n\n")

        # Insertions
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\toprule\n")
        f.write("Inserted Sound & Frequency \\\\\n")
        f.write("\\midrule\n")
        for p, count in freqs["insert"].most_common():
            f.write(f"{tex_escape(p)} & {count} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Character Insertions}\n")
        f.write("\\end{table}\n\n")

        # Deletions
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\toprule\n")
        f.write("Deleted Sound & Frequency \\\\\n")
        f.write("\\midrule\n")
        for r, count in freqs["delete"].most_common():
            f.write(f"{tex_escape(r)} & {count} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Character Deletions}\n")
        f.write("\\end{table}\n\n")

        f.write("\\end{document}\n")


@click.command()
@click.argument(
    "input_tsv", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option(
    "--word-output", default="word_error_analysis.csv", show_default=True
)
@click.option(
    "--char-output", default="character_error_analysis.csv", show_default=True
)
@click.option(
    "--freq-output",
    default="character_error_frequencies.csv", show_default=True
)
@click.option(
    "--latex-output",
    default="character_error_frequencies.tex", show_default=True
)
def main(
    input_tsv: str,
    word_output: str,
    char_output: str,
    freq_output: str,
    latex_output: str
) -> None:
    """Run the comprehensive error analysis script."""
    click.echo(f"Loading data from '{input_tsv}'...")
    try:
        data = load_tsv_data(input_tsv)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    click.echo(f"Processing {len(data)} records...")
    w_tally, w_ex, c_tally, c_ex, c_freqs = analyze_all_errors(data)

    click.secho(
        "\n=== Repeated errors (Frequency > 1) ===", fg="yellow", bold=True
    )
    click.echo("Substitutions:")
    for (r, p), count in c_freqs["substitute"].most_common():
        if count > 1:
            click.echo(f"  '{r}' substituted with '{p}' ({count} times)")

    click.echo("\nInsertions:")
    for p, count in c_freqs["insert"].most_common():
        if count > 1:
            click.echo(f"  '{p}' inserted ({count} times)")

    click.echo("\nDeletions:")
    for r, count in c_freqs["delete"].most_common():
        if count > 1:
            click.echo(f"  '{r}' deleted ({count} times)")
    click.echo()

    base_headers = [
        "Audio Path", "High Error (>10%)", "Error Type",
        "Substituted", "Substituted With", "Inserted", "Deleted"
    ]
    char_headers = base_headers + ["Full Reference SRO", "Full Predicted SRO"]
    word_headers = base_headers + [
        "Character Error Types", "Full Reference SRO", "Full Predicted SRO"
    ]

    export_examples_to_csv(c_ex, char_output, char_headers)
    export_examples_to_csv(w_ex, word_output, word_headers)
    export_frequencies_to_csv(c_freqs, freq_output)
    export_frequencies_to_latex(c_freqs, latex_output)

    click.secho("The following files were written:", fg="green")
    click.echo(f"  - {char_output}")
    click.echo(f"  - {word_output}")
    click.echo(f"  - {freq_output}")
    click.echo(f"  - {latex_output}")


if __name__ == "__main__":
    main()
