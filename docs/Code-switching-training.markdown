Code-switching is one of the toughest challenges in speech recognition, but
Whisper's architecture is uniquely positioned to handle it if you structure
your next training run correctly.

The core hurdle you are going to face is **catastrophic forgetting**. Because
your base model was pre-trained heavily on English, fine-tuning it purely on
Plains Cree overwrites its internal English dictionary. When a speaker switches
to English mid-sentence, a model trained only on Cree will try to spell the
English words using Cree orthography (SRO).

To build a model that handles fluid code-switching, here is how you should
adjust your approach for the next iteration.

## 1. The Single-Token Strategy

Whisper processes audio in 30-second chunks and assigns a single language token
at the very beginning of that chunk. It cannot dynamically swap language tokens
mid-sentence.

Because you cannot tell the model "this chunk is Cree *and* English," you must
train it to expect both under a single umbrella token. Sticking with the
`<|en|>` token (like your current script accidentally did) is actually a highly
effective strategy for code-switching. By passing English, Cree, and mixed
audio all under the `<|en|>` token during training, you teach the model that
the "English" token space now includes Plains Cree.

## 2. Rebalance Your Training Data

To teach the model to switch languages without forgetting how to spell standard
English, you need to curate a blended dataset. Your next training run should
combine your new Cree data with existing high-quality datasets.

| Data Type | Suggested Mix | Purpose |
| --- | --- | --- |
| **Pure Plains Cree** | 60-70% | Core task learning and SRO formatting. |
| **Code-Switched** | 10-20% | Teaches the acoustic boundaries where languages swap. |
| **Pure English** | 10-20% | Acts as "regularization" to preserve English spelling. |

*Note: You can pull the pure English audio from open-source datasets like
LibriSpeech or Common Voice. You only need a fraction to keep the English
weights intact.*

## 3. Unify the Orthography

Whisper is a byte-level model, meaning it pays close attention to exact
character usage. If your Cree transcripts use macrons (ā, ē, ī, ō) but your
English transcripts use standard ASCII, you must ensure your text normalization
pipeline is perfectly consistent before the text hits the tokenizer.

If an English loanword is pronounced with a Cree accent, decide *now* whether
your transcribers will spell it using English rules or Cree phonetics, and
enforce that rule strictly across the whole corpus. Inconsistent ground-truth
labels will confuse the model during code-switching segments.

## 4. Freeze the Encoder

If your new dataset is still relatively small (under 100 hours), consider
freezing Whisper's encoder blocks during your next training run. Whisper
already knows what human speech sounds like; you are primarily teaching its
*decoder* how to map those sounds to a new mixed-language vocabulary. Freezing
the encoder prevents the model from overfitting to the specific background
noise or microphone quality of your new dataset.