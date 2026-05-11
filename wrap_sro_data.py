import os
import pandas as pd

wav_dir = 'sand-psalm/wav'
txt_dir = 'sand-psalm/sro'
output_csv = 'metadata.csv'

dataset_records = []

for wav_filename in os.listdir(wav_dir):
    if wav_filename.endswith('.wav'):
        base_name = os.path.splitext(wav_filename)[0]
        wav_path = os.path.join(wav_dir, wav_filename)
        sro_path = os.path.join(txt_dir, f"{base_name}.sro")

        if os.path.exists(sro_path):
            with open(sro_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()

            dataset_records.append({
                "audio": wav_path,
                "sentence": transcript
            })

df = pd.DataFrame(dataset_records)
df.to_csv(output_csv, index=False)
print(f"Mapped {len(dataset_records)} files. Saved to {output_csv}")
