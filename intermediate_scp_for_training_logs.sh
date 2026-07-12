# These are used to get the logs from the gpu server to the jump server.
scp altlab-gpu:whisper-test/hf_cache/whisper-tiny-finetuned/checkpoint-350/trainer_state.json tiny/
scp altlab-gpu:whisper-test/hf_cache/whisper-base-finetuned/checkpoint-250/trainer_state.json base/
scp altlab-gpu:whisper-test/hf_cache/whisper-small-finetuned/checkpoint-200/trainer_state.json small/
scp altlab-gpu:whisper-test/hf_cache/whisper-medium-finetuned/checkpoint-200/trainer_state.json medium/
scp altlab-gpu:whisper-test/hf_cache/whisper-large-finetuned/checkpoint-175/trainer_state.json large/

# Get the extra logs
scp altlab-gpu:whisper-test/hf_cache/whisper-large-finetuned/train_results.json large/
scp altlab-gpu:whisper-test/hf_cache/whisper-large-finetuned/all_results.json large/
scp altlab-gpu:whisper-test/hf_cache/whisper-small-finetuned/train_results.json small/

# Get the test results at the end of each training.
for size in tiny base small medium large; 
do scp altlab-gpu:whisper-test/hf_cache/whisper-$size-finetuned/test_results.tsv $size/; 
done

# and to get these to the local system run
scp -r juhapert@134.87.11.120:test_results/* .
# or similar