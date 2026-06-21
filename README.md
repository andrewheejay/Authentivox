# Authentivox — Real-Time Voice Phishing Detection

Authentivox is a binary classifier that tells real human speech apart from
AI-generated (deepfake) voices. It extracts acoustic features from audio
frames and uses a Random Forest model to flag synthetic speech — light
enough to run without a GPU, with real-time call screening as the target use
case.

[Read the paper](docs/Authentivox-Paper.pdf)

---

## Why

The people most often targeted by voice phishing are the least equipped to
recognize a cloned voice mid-call. The goal here was a detector light enough
to run on a normal phone, in real time, while the call is still happening —
not a heavyweight model that needs a server.

## How it works

1. **Feature extraction (Librosa).** Each audio file is reduced to a set of
   acoustic features per frame: chromagram, RMS energy, spectral centroid,
   bandwidth, rolloff, zero-crossing rate, and the first 20 MFCCs. MFCCs in
   particular capture the spectral artifacts that voice-conversion models
   leave behind.
2. **Classification (Random Forest).** An ensemble of decision trees votes on
   each frame. Random Forest was chosen over XGBoost because it gives
   comparable accuracy at a fraction of the compute — the trade that matters
   for running on-device without a GPU.
3. **Validation.** 28 repeated random 70:30 train/test splits, matching the
   paper's reported approach (average/median/standard deviation across runs,
   rather than k-fold cross-validation). There are only 8 real files against
   56 converted ones, so each repeat undersamples the fake class down to the
   real class's size before splitting — otherwise the classifier sees far
   more fake frames than real ones and starts defaulting to "fake."

## Dataset

[Kaggle DEEP-VOICE: DeepFake Voice Recognition](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition)
— 64 samples (8 real, 56 converted) from eight public figures, with voices
converted into one another using Retrieval-based Voice Conversion to simulate
real-time manipulation. The dataset is **not** redistributed here; download it
from Kaggle into `./data` (expected layout: `data/AUDIO/REAL/*.wav`,
`data/AUDIO/FAKE/*.wav`).

## Results

Measured by actually running this notebook end-to-end (28 repeated 70:30
splits, each on a class-balanced sample, Random Forest, `n_estimators=50`):

| Metric    | Mean    | Std. Dev |
|-----------|---------|----------|
| Accuracy  | 91.01%  | 0.11     |
| Precision | 0.921   | 0.002    |
| Recall    | 0.898   | 0.002    |
| F1-Score  | 0.909   | 0.001    |
| MCC       | 0.82    | 0.002    |
| ROC AUC   | 0.91    | 0.001    |

The paper reports a higher 95.93% accuracy with 0.96 precision/F1 — that
result came from validating on individual real/fake voice pairs one at a
time, which is naturally balanced 1:1. The code that produced that exact run
is no longer recoverable; only its final summary numbers survived. An
earlier version of this notebook pooled the full dataset without balancing
classes first and got a higher 94.18% accuracy and 0.979 precision, but
recall was only 0.547 — with 8 real files against 56 converted ones, the
unbalanced classifier defaulted toward predicting "fake." Balancing each
split (above) trades a few points of accuracy and precision for recall
nearly doubling, which is the more honest number to report for a detector
meant to actually catch real voices, not just avoid false alarms.

## Recognition

- Published in the *International Journal of STEAM*
- Silver Medal, 2024 Korean Science and Engineering Fair (KSEF)
- KUSEF Finalist · CJSJ Semi-Finalist

## Running it

```bash
git clone https://github.com/andrewheejay/Authentivox.git
cd Authentivox
pip install -r requirements.txt
# download the DEEP-VOICE dataset from Kaggle into ./data
jupyter notebook voice_phishing_detection.ipynb
```

Heads up: the 28 repeated Random Forest fits take roughly an hour on a
modern laptop CPU — the original code never set `max_depth`, so trees grow
fully, and each repeat still trains on ~320K balanced frames.

## Stack

Python · Librosa · scikit-learn · pandas · NumPy · matplotlib

## License

MIT
