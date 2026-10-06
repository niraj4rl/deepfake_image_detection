# ForensiScan

ForensiScan is a CPU-only classical Digital Image Processing pipeline for detecting face-swap images. The default pipeline uses real FaceForensics++ c23 data only. It never generates or consumes the synthetic development dataset unless the explicitly named `ml.data.make_dummy_data` module is run manually.

## Real data setup

FaceForensics++ requires access approval from its official source. After downloading the c23 videos, place them at:

```text
data/raw/original_sequences/youtube/c23/videos/*.mp4
data/raw/manipulated_sequences/Deepfakes/c23/videos/*.mp4
data/raw/manipulated_sequences/FaceSwap/c23/videos/*.mp4
data/raw/manipulated_sequences/Face2Face/c23/videos/*.mp4
```

Copy the official `train.json`, `val.json`, and `test.json` files into `ml/data/splits/`. Dataset files are ignored by git and must not be committed.

The pipeline uses whichever configured manipulation folders are available, but at least the real folder and one configured manipulation folder are needed for a useful classifier. Missing data is reported as an error or warning; it is never replaced with fake data.

## Setup and real-data pipeline

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
python run_all.py
```

The pipeline samples frames, detects and crops faces, writes `data/manifest.csv`, builds the cached feature matrix at `data/features/features.npz`, and trains `models/best_model.joblib`. `run_all.py` stops before writing anything when the real dataset is missing.

Individual stages are also available:

```powershell
python -m ml.data.extract_frames
python -m ml.data.build_manifest
python -m ml.features.extractor
python -m ml.train
```

## Limitations and ethics

The model is trained on FaceForensics++ face-swap data and will not generalize reliably to other generators or heavily compressed images. Results are probabilistic indicators, not proof, and should not be used for legal, journalistic, or personal accusations. FaceForensics++ must be used according to its research and education terms.

The synthetic helper exists only for isolated plumbing smoke tests and must never be used for report metrics. Real uploaded and dataset images remain local.