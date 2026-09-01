# nêhiyawêwin / Plains Cree ASR

This repository contains scripts for fine-tuning Hugging Face's Whisper models
on local speech data that we have used in training nêhiyawêwin / Plains Cree
ASR models. It **does not** contain the trained models nor the training data. 

The pipeline is configured via YAML profiles to transition between
local desktop testing (CPU) and high-performance computing (HPC) environments
like Digital Alliance's Narval (A100 GPUs).

Documentation:
- [Getting started](docs/Getting_started.markdown)
- [How to run a training on the altlab-gpu server](docs/altlab-gpu-training-run.markdown)
- Future plans:
  - [Training with code switching data](docs/Code-switching-training.markdown) 
  - [Integrating a Finite State Transducer to Whisper](docs/FST-integration.markdown)

At time of writing (31st August 2026), all documentation is preliminary as it
has only had one set of human eys on it so far. This should change within a
couple of months.

## Package name

The package is properly called nêhiyawêwin-asr, but you should use the
ASCII-only (linguistically incorrect) spelling of `nehiyawewin-asr` in code and
on the commandline as computers are not always that good with matching
diacritics between systems and this can lead to broken functionality.
