# These are used to get the logs from the gpu server to the jump server.
scp altlab-gpu:whisper-test/hf_cache/whisper-tiny-finetuned/checkpoint-350/trainer_state.json tiny/
scp altlab-gpu:whisper-test/hf_cache/whisper-base-finetuned/checkpoint-250/trainer_state.json base/
scp altlab-gpu:whisper-test/hf_cache/whisper-small-finetuned/checkpoint-200/trainer_state.json small/
scp altlab-gpu:whisper-test/hf_cache/whisper-medium-finetuned/checkpoint-125/trainer_state.json medium/
scp altlab-gpu:whisper-test/hf_cache/whisper-large-finetuned/checkpoint-250/trainer_state.json large/

# Get the extra logs
scp whisper-test/hf_cache/whisper-small-finetuned/train_results.tsv small/

# Get the test results at the end of each training.
scp altlab-gpu:whisper-test/hf_cache/whisper-tiny-finetuned/test_results.tsv tiny/
scp altlab-gpu:whisper-test/hf_cache/whisper-base-finetuned/test_results.tsv base/
scp altlab-gpu:whisper-test/hf_cache/whisper-small-finetuned/test_results.tsv small/
scp altlab-gpu:whisper-test/hf_cache/whisper-medium-finetuned/test_results.tsv medium/
scp altlab-gpu:whisper-test/hf_cache/whisper-large-finetuned/test_results.tsv large/

# and to get these to the local system run
scp -r juhapert@134.87.11.120:test_results/* .
# or similar