# Chehre — Multimodal Preprocessing Pipeline

Preprocessing pipelines for the **Chehre** dataset, a multimodal corpus pairing
**facial-reaction video** with **emoji-based affect labels**. This repository turns
raw recordings and raw survey annotations into clean, model-ready inputs:

1. **Face extraction (computer vision)** — detect and crop faces from raw video into
   fixed-size frames suitable for downstream facial-expression models.
2. **Label analysis (NLP)** — quantify how human annotators describe each emoji by
   embedding their text labels with BERT and measuring pairwise semantic
   similarity, surfacing redundant or near-synonymous labels before they reach a model.


> "Chehre is a dataset of participant reaction videos to emoji stimuli, collected
> for emotion-recognition research at the Rosie Lab in Simon Fraser University. 

---

## Why this exists

Raw multimodal data is rarely usable as-is. Reaction videos contain background,
movement, and inconsistent framing. The emotion text labels collected from surveys
are noisy and full of near-duplicates ("happy", "joyful", "cheerful"). This preprocessing pipeline
standardizes both modalities so the downstream model sees consistent, de-duplicated,
well-cropped inputs.

---

## Repository structure

```
chehre-preprocessing/
├── face_crop.py               # Pipeline 1: detect + crop faces from video → frames/video
├── embeddings.py              # BERT word embeddings + cosine-similarity utilities
├── emoji_label_similarity.py  # Pipeline 2: emoji label JSON → similarity matrix + heatmap
├── emojiJSON.json             # Example emoji survey-label data
├── examples/
│   └── word_similarity_heatmap.png   # Sample output (see below)
├── requirements.txt
├── .gitignore
└── LICENSE
```

> **Note:** `emoji_label_similarity.py` is the renamed, refactored version of the old
> `dict.py`. `face_crop.py` replaces the old `detect_crop.py` / `test.py` pair (see
> *Migration notes* at the bottom).

---

## Installation

```bash
git clone https://github.com/audrey-sk/Chehre-preprocessing.git
cd Chehre-preprocessing
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

The Chehre pipeline uses OpenCV's bundled Haar cascade by default (no extra download).
To use the higher-accuracy RetinaFace detector, also install the optional dependency:

```bash
pip install retina-face        # pulls in TensorFlow; only needed for --detector retinaface
```

---

## Pipeline 1 — Face extraction & cropping

Detects the primary face in each video frame and crops a fixed-size square centered on
it, then re-assembles the crops into a video.

```bash
# Crop a video to 512×512 face-centered frames and write a new video
python face_crop.py --input video.mp4 --output cropped_video.mp4 --size 512

# Also dump every cropped frame into ./frames/
python face_crop.py --input video.mp4 --output cropped_video.mp4 --save-frames frames/

# Use RetinaFace instead of the default Haar cascade
python face_crop.py --input video.mp4 --output cropped_video.mp4 --detector retinaface
```

- **Default detector is Haar cascade** for zero-setup, CPU-only operation; **RetinaFace**
  is available as an opt-in for better recall on occluded faces.
- **Headless-safe:** runs on a server with no display. Frame preview is opt-in
  (`--show`) rather than always-on, so it won't crash in a non-GUI environment.
- **Graceful fallback:** if no face is detected in a frame, the crop falls back to the
  frame center instead of failing, so the output stays continuous.

---

## Pipeline 2 — Emoji label semantic similarity

Reads the emoji survey JSON, extracts the leading descriptor word from each annotation,
embeds every unique word **once** with BERT, and computes the full pairwise
cosine-similarity matrix in a single vectorized operation.

```bash
python emoji_label_similarity.py \
  --input emojiJSON.json \
  --out-csv word_similarity_matrix.csv \
  --out-heatmap examples/word_similarity_heatmap.png \
  --pooling mean
```

**Outputs**
- `word_similarity_matrix.csv` — an matrix of cosine similarities between label words.
- a heatmap visualizing the matrix:

![Word similarity heatmap](examples/word_similarity_heatmap.png)

**Features**
- **Embed-once, compare-all:** each unique word is embedded a single time, then all
  pairwise similarities come from one normalized matrix multiply. This replaces the
  earlier approach that re-ran BERT for every word *pair* — reducing the number of model
  forward passes from roughly *N²* to *N* (a large speedup as the vocabulary grows).
- **Configurable subword pooling:** BERT splits rare words into subword tokens
  (`aardvark → a ##ard ##var ##k`). `--pooling mean` averages the subword embeddings; `--pooling first` keeps the original
  first-token behavior. Both are exposed so the choice is explicit and reproducible.

---

## Reproducibility

- All randomness-free; outputs are deterministic given the same inputs and model.
- Model name (`bert-base-cased`), pooling strategy, and all I/O paths are CLI flags —
  no hard-coded paths buried in the source.
- `requirements.txt` pins the dependency set; run `pip freeze > requirements.lock.txt`
  to capture exact versions for a fully frozen environment.


---

## Migration notes

The original repository contained `detect_crop.py` (Haar-cascade version) and `test.py`
(a RetinaFace driver that imported `extractframes`, `crop`, and `facedetect`). Both have been
consolidated into a single, self-contained, runnable `face_crop.py` that supports either
detector via a flag.

---

## License

Released under the [MIT License](LICENSE).
