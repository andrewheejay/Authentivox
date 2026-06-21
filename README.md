# Authentivox — Real-Time Voice Phishing Detection

Authentivox is a binary classifier that tells real human speech apart from
AI-generated (deepfake) voices. It extracts acoustic features from audio
frames and uses a Random Forest model to flag synthetic speech — light
enough to run without a GPU, with real-time call screening as the target use
case.

**Accuracy: 98.46%** (Random Forest, 28 repeated 70:30 splits — see
[Results](#results))

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
   rather than k-fold cross-validation).

## Dataset

[Kaggle DEEP-VOICE: DeepFake Voice Recognition](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition)
— 64 samples (8 real, 56 converted) from eight public figures, with voices
converted into one another using Retrieval-based Voice Conversion to simulate
real-time manipulation. The raw audio is **not** redistributed here; download
it from Kaggle into `./data` if you want to run the feature-extraction demo
(expected layout: `data/AUDIO/REAL/*.wav`, `data/AUDIO/FAKE/*.wav`).

The model itself trains on `dataset/DATASET-balanced.csv`, a precomputed,
already-balanced feature set recovered from the original project files —
5,889 real and 5,889 fake rows, one row per ~1-second analysis window across
all 64 files. The script that originally generated this CSV from raw audio
no longer exists, only the CSV itself, so this notebook uses it directly
rather than re-deriving an approximate version.

## Results

Measured by actually running this notebook end-to-end (28 repeated 70:30
splits, Random Forest, `n_estimators=50`, on the dataset above):

| Metric    | Mean    | Std. Dev |
|-----------|---------|----------|
| Accuracy  | 98.46%  | 0.19     |
| Precision | 0.991   | 0.003    |
| Recall    | 0.978   | 0.003    |
| F1-Score  | 0.984   | 0.002    |
| MCC       | 0.969   | 0.004    |
| ROC AUC   | 0.985   | 0.002    |

This matches, and slightly exceeds, the paper's reported 95.93% accuracy and
0.96 precision/F1. An earlier version of this notebook tried to reconstruct
the feature set from scratch by pooling every raw audio frame across all 64
files — that approach surfaced a real 8-real-vs-56-fake class imbalance and
landed well short of the paper's numbers (94.18% accuracy but only 0.55
recall unbalanced; 91% across the board after balancing). The dataset above
is the one that actually backs the published result.

## Recognition

- Published in the *International Journal of STEAM*
- Silver Medal, 2024 Korean Science and Engineering Fair (KSEF)
- KUSEF Finalist · CJSJ Semi-Finalist

## Running it

```bash
git clone https://github.com/andrewheejay/Authentivox.git
cd Authentivox
pip install -r requirements.txt
# optional — only needed for the feature-extraction demo cells:
# download the DEEP-VOICE dataset from Kaggle into ./data
jupyter notebook voice_phishing_detection.ipynb
```

The training and validation cells run in well under a minute — they use the
precomputed `dataset/DATASET-balanced.csv`, not the raw audio.

## Stack

Python · Librosa · scikit-learn · pandas · NumPy · matplotlib

## License

MIT
