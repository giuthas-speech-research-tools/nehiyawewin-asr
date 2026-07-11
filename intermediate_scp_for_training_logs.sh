# These are used to get the logs from the gpu server to the jump server.
scp altlab-gpu:whisper-test/hf_cache/whisper-tiny-finetuned/checkpoint-350/trainer_state.json tiny/
scp altlab-gpu:whisper-test/hf_cache/whisper-base-finetuned/checkpoint-250/trainer_state.json base/
scp altlab-gpu:whisper-test/hf_cache/whisper-small-finetuned/checkpoint-125/trainer_state.json small/
scp altlab-gpu:whisper-test/hf_cache/whisper-medium-finetuned/checkpoint-125/trainer_state.json medium/
scp altlab-gpu:whisper-test/hf_cache/whisper-large-finetuned/checkpoint-250/trainer_state.json large/


scp /mnt/hf_cache/whisper-small-finetuned/training_results.tsv small/

scp /mnt/hf_cache/whisper-tiny-finetuned/test_results.tsv tiny/
scp /mnt/hf_cache/whisper-base-finetuned/test_results.tsv base/
scp /mnt/hf_cache/whisper-small-finetuned/test_results.tsv small/
scp /mnt/hf_cache/whisper-medium-finetuned/test_results.tsv medium/
scp /mnt/hf_cache/whisper-large-finetuned/test_results.tsv large/