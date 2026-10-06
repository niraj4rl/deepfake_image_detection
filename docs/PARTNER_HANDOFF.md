# ForensiScan Partner Task

## Objective

Complete and validate ForensiScan, a CPU-only face-swap deepfake detector based on classical Digital Image Processing features and classical machine learning.

The final project must use real FaceForensics++ data for every reported training and evaluation result. Synthetic data must never be used for model training, metrics, screenshots, or the project report.

## Current Repository State

Already implemented:

- Real-data FF++ frame sampling in `ml/data/extract_frames.py`.
- Face detection and cropping in `ml/data/face_crop.py`.
- Official-split-aware manifest generation in `ml/data/build_manifest.py`.
- Classical feature extraction in `ml/features/`:
  - FFT and DCT
  - ELA at JPEG qualities 90 and 75
  - Noise residuals and fixed high-pass filters
  - LBP, GLCM, and Gabor texture features
  - YCrCb/HSV color statistics and edge features
- The combined extractor produces 317 finite features per 256x256 face crop.
- Parallel feature caching in `ml/features/extractor.py`.
- Initial CPU SVM training in `ml/train.py`.
- FastAPI service in `backend/app/main.py` with:
  - health and model-info endpoints
  - image validation and local inference
  - SQLite history
  - thumbnails
  - CSV export
  - history retrieval and deletion
- React/Vite frontend in `frontend/` with Analyze, Batch, History, and About views.
- Frontend production build currently passes.
- Synthetic helper `ml/data/make_dummy_data.py` exists only for isolated plumbing tests. It must not be called by the real pipeline.

## Required Dataset

The dataset is not currently present in the repository. Obtain FaceForensics++ through its official access process. Do not download it from an unofficial source and do not commit the dataset to git.

Place the c23 videos in this layout:

```text
data/raw/original_sequences/youtube/c23/videos/*.mp4
data/raw/manipulated_sequences/Deepfakes/c23/videos/*.mp4
data/raw/manipulated_sequences/FaceSwap/c23/videos/*.mp4
data/raw/manipulated_sequences/Face2Face/c23/videos/*.mp4
```

Place the official split files here:

```text
ml/data/splits/train.json
ml/data/splits/val.json
ml/data/splits/test.json
```

The raw dataset and generated files must remain git-ignored.

## Immediate Run Sequence

From the repository root, after the dataset is available:

```powershell
python -m ml.data.extract_frames
python -m ml.data.build_manifest
python -m ml.features.extractor
python -m ml.train
```

Or run the guarded combined command:

```powershell
python run_all.py
```

The command must stop with a clear error if real data or split files are missing. It must never create a synthetic manifest automatically.

## Remaining Engineering Work

### 1. Validate and improve data splitting

- Confirm that the official split JSON format is handled correctly.
- Add a test proving no `group_id` appears in more than one split.
- Confirm all frames from one video remain in one split.
- Confirm fake videos inherit the split of their target real video ID.
- Verify real/fake balance per split and document the final counts.
- Add a concise log for skipped videos, unreadable frames, and frames without detected faces.

### 2. Complete training and evaluation

Extend `ml/train.py` to satisfy the specification:

- Use `VarianceThreshold` and `StandardScaler` in a scikit-learn pipeline.
- Compare Logistic Regression, RBF SVM, Random Forest, and XGBoost CPU `hist`.
- Use `GroupKFold` on the training split only.
- Select the final model using validation data.
- Evaluate the test split exactly once after model selection.
- Save:
  - `models/best_model.joblib`
  - `models/metadata.json`
  - `reports/tables/model_comparison.csv`
  - `reports/tables/model_comparison.md`
- Record accuracy, precision, recall, F1, ROC-AUC, average precision, confusion matrix, and per-manipulation recall.
- Compute and store the validation Youden-J threshold.
- Never tune against the test set.

Add these missing scripts:

```text
ml/evaluate.py
ml/ablation.py
ml/robustness.py
ml/make_figures.py
```

### 3. Finish backend behavior

- Add a dedicated `predictor.py` service and typed schemas if helpful.
- Ensure inference uses exactly the same face crop and feature extractor as training.
- Add batch endpoint `POST /api/analyze/batch` with a maximum of 20 files.
- Enforce configured upload size and MIME/magic-byte validation.
- Add backend tests for health, invalid uploads, oversized uploads, no-face images, batch failures, history, CSV export, and inference parity.
- Verify the backend returns `503` when no real trained model exists.

### 4. Finish frontend behavior

- Add real batch progress and per-file failure states.
- Add history filtering, pagination, sorting, deletion confirmation, and clear-all confirmation.
- Add model metrics from `/api/model/info` to the About page.
- Add face bounding-box overlay and evidence panel when backend evidence is enabled.
- Add frontend tests for verdict, confidence gauge, upload validation, and error states.
- Verify at 375px, 768px, and 1280px widths.

### 5. Documentation and report outputs

- Update `README.md` with final setup, dataset access instructions, commands, screenshots, metrics, and limitations.
- Add generated report tables and figures under `reports/`.
- Document all design decisions in `docs/DECISIONS.md`.
- State clearly that results are probabilistic indicators, not proof.
- State that FaceForensics++ usage must follow its research/education terms.

## Acceptance Criteria

The task is complete only when all of the following are true:

- `python run_all.py` completes successfully on the supplied real FF++ data.
- No generated result contains dummy-data samples.
- The manifest has the required columns and no split leakage.
- The feature cache contains finite values and records feature names, groups, splits, and image paths.
- A real-data model and metadata are saved under `models/`.
- Validation and held-out test metrics are saved under `reports/tables/`.
- Ablation, robustness, and required report figures exist under `reports/`.
- `pytest` passes for ML, backend, and frontend tests.
- `npm run build` passes with zero TypeScript errors.
- FastAPI starts with:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

- Vite starts with:

```powershell
cd frontend
npm install
npm run dev
```

- The frontend can analyze a valid face image using the trained real-data model.
- The UI and README include the model limitations and privacy disclaimer.

## Important Constraint

Do not solve missing-data errors by calling `ml.data.make_dummy_data`. If the FF++ dataset is unavailable, stop and report the missing path or file. The project must be honest about the absence of real training data.