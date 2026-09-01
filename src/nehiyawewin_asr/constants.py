"""
Constants used across the metric extraction module.

This module isolates string literals, glob patterns, and
configuration details to maintain separation of responsibilities.
"""

# Expected filename for Hugging Face trainer state logs.
STATE_FILE_NAME: str = "trainer_state.json"

# Glob pattern for matching TensorBoard event files.
TFEVENTS_PATTERN: str = "runs/**/events.out.tfevents.*"

# Headers utilized for the output CSV file.
CSV_HEADERS: list[str] = [
    "epoch",
    "step",
    "loss",
    "estimated_elapsed_time_sec"
]

# JSON dictionary keys used within the trainer_state file.
KEY_LOG_HISTORY: str = "log_history"
KEY_STEPS_PER_SEC: str = "train_steps_per_second"
KEY_LOSS: str = "loss"
KEY_EPOCH: str = "epoch"
KEY_STEP: str = "step"

# Fallback string when elapsed time cannot be estimated.
UNKNOWN_TIME: str = "Unknown"
