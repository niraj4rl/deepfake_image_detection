# ForensiScan: Classical-DIP Deepfake Image Detector
### Build specification for an AI coding agent (Antigravity)

> **How to use this document:** Place it in the repo root as `SPEC.md`. Build the project phase by phase (Section 15). After each phase, run the verification commands and check the acceptance criteria before moving on. Do not skip ahead. If something here is ambiguous, choose the simplest option that satisfies the stated goal and note the decision in `docs/DECISIONS.md`.

---

## 1. Project Summary

**Goal:** Build a full-stack web application that decides whether an uploaded face image is a **real** photo or a **face-swap deepfake**, using **classical Digital Image Processing (DIP) feature extraction + classical machine learning** (no deep learning).

**Why this design:** This is a college Digital Image Processing (DIP) project. The grading value lies in demonstrating DIP techniques (frequency-domain analysis, error level analysis, noise residuals, texture analysis, color and edge analysis) and showing that they measurably separate real from fake faces. A modern React frontend wraps the model so it is demonstrable and usable.

**Primary deliverables**
1. Reproducible Python ML pipeline (dataset prep, feature extraction, training, evaluation, ablations, robustness tests).
2. FastAPI backend serving the trained model.
3. Polished React (Vite) frontend: single-image analysis, batch analysis, and history.
4. Generated figures and metrics tables suitable for pasting into a project report.

**Non-goals**
- No CNN/ViT/deep learning models.
- No video deepfake detection (images/frames only).
- No GAN-only or diffusion-image detection (scope is face-swap deepfakes from FaceForensics++).
- No user accounts or authentication.

---

## 2. Hard Constraints

| Constraint | Detail |
|---|---|
| Compute | **CPU-only laptop.** No GPU code paths. Training must finish in under ~30 minutes. Inference under ~2 seconds per image. |
| Models | scikit-learn and XGBoost (CPU `hist` tree method) only. |
| Dataset size | Keep to a few thousand images total (see Section 5). |
| Reproducibility | Fixed random seed (`42`) everywhere. All paths configurable via `config.yaml`. |
| OS | Must run on Windows, macOS and Linux. Use `pathlib`, no shell-specific scripts as the only entry point (provide Python entry points, optionally Makefile too). |
| Privacy | Uploaded images stay local; nothing is sent to third-party services. |

---

## 3. Tech Stack

**Backend / ML (Python 3.10+)**
- `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic` v2
- `sqlmodel` (or SQLAlchemy 2.x) + SQLite for history
- `opencv-python`, `scikit-image`, `numpy`, `scipy`, `Pillow`
- `scikit-learn`, `xgboost`, `joblib`
- `pandas`, `matplotlib`, `seaborn` (reports and figures)
- `pytest`, `httpx` (tests), `pyyaml`, `tqdm`
- Face detection: OpenCV DNN **res10 SSD** (Caffe) as primary; Haar cascade (`haarcascade_frontalface_default.xml`) as fallback.

**Frontend**
- React 18 + **Vite** + **TypeScript**
- Tailwind CSS (with CSS variables for theme tokens)
- React Router v6
- TanStack Query (data fetching and cache)
- `react-dropzone` (uploads), `framer-motion` (subtle animations), `lucide-react` (icons), `recharts` (small charts)
- Vitest + React Testing Library (a few component tests)

---

## 4. Repository Structure

```
forensiscan/
├── SPEC.md
├── README.md
├── config.yaml
├── Makefile                      # optional convenience targets
├── docs/
│   └── DECISIONS.md
├── data/                         # git-ignored except .gitkeep
│   ├── raw/                      # FF++ videos (user-provided)
│   ├── frames/                   # extracted full frames
│   ├── faces/                    # cropped 256x256 faces
│   ├── manifest.csv              # one row per face image
│   └── features/                 # cached feature matrices (.npz)
├── ml/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── extract_frames.py
│   │   ├── face_crop.py
│   │   ├── build_manifest.py
│   │   └── splits.py
│   ├── features/
│   │   ├── frequency.py          # FFT + DCT
│   │   ├── ela_noise.py          # ELA + noise residuals
│   │   ├── texture.py            # LBP, GLCM, Gabor
│   │   ├── color_edge.py         # color stats + edges
│   │   └── extractor.py          # combines all groups -> vector + names
│   ├── train.py
│   ├── evaluate.py
│   ├── ablation.py
│   ├── robustness.py
│   └── make_figures.py
├── models/                       # saved artifacts (joblib, metrics.json)
├── reports/
│   ├── figures/                  # PNGs for the report
│   └── tables/                   # CSVs and markdown tables
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes.py
│   │   ├── schemas.py
│   │   ├── db.py
│   │   ├── services/predictor.py
│   │   └── storage.py
│   ├── storage/                  # uploads + thumbnails (git-ignored)
│   ├── requirements.txt
│   └── tests/
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.ts
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── api/client.ts
        ├── components/
        ├── pages/
        ├── hooks/
        ├── types/
        └── styles/
```

---

## 5. Dataset: FaceForensics++ (c23 subset)

### 5.1 Source and access
- FaceForensics++ requires filling the **official access form** (from the FaceForensics GitHub repo). The user receives a download script by email. **The agent cannot download the data**, so build the pipeline to expect videos under `data/raw/` and clearly document the expected layout in the README.
- Compression level: **c23** (light compression; the most commonly used).
- Needed folders:
  ```
  data/raw/original_sequences/youtube/c23/videos/*.mp4
  data/raw/manipulated_sequences/Deepfakes/c23/videos/*.mp4
  data/raw/manipulated_sequences/FaceSwap/c23/videos/*.mp4
  data/raw/manipulated_sequences/Face2Face/c23/videos/*.mp4
  ```
- Include a short `README` section: if only some folders are available, the pipeline must still run on whatever manipulation types exist.
- **Fallback for development before the real data arrives:** provide `ml/data/make_dummy_data.py` that generates a tiny synthetic dataset (e.g. random-noise faces with simple artifacts) *only* so the whole pipeline and the UI can be tested end to end. The report must never use dummy-data results; print a loud warning when dummy data is in use.

### 5.2 Which manipulations
Use **Deepfakes, FaceSwap, Face2Face** (three face-manipulation methods). Skip NeuralTextures and FaceShifter to keep the subset small.

### 5.3 Subset definition (to keep CPU training light)
- Use the official FF++ split JSONs (`train.json`, `val.json`, `test.json`; lists of video-ID pairs). Store a copy in `ml/data/splits/`.
- Take a subset of **real video IDs**: 100 train, 20 val, 20 test (configurable).
- Real class: videos from `original_sequences` with those IDs.
- Fake class: for each manipulation method, the fake videos whose **target ID** (the first number in `XXX_YYY.mp4`) is in the chosen set.
- From each video extract **10 frames, evenly spaced** across its duration.
- Face-crop every frame (Section 5.4).
- **Balance:** target a 1:1 real:fake ratio per split by undersampling fake images, stratified evenly across the three manipulation methods. Expected total roughly 2,500–3,000 face images.
- **No leakage:** split by **video ID group**, never by frame. All frames from one video, and fakes derived from the same target ID, stay in the same split. Add a unit test asserting that no group appears in more than one split.

### 5.4 Face detection and cropping
1. Detect with OpenCV DNN res10 SSD (download model files in a helper script; cache in `models/face_detector/`). Fallback to Haar if the DNN files are missing.
2. Pick the **largest** face. If none is found, skip the frame and log it.
3. Expand the box by a **1.3x margin**, make it square, clamp to image bounds.
4. Resize to **256×256** (Lanczos) and save as **PNG (lossless)** so that cropping does not add extra JPEG artifacts.
5. Write `data/manifest.csv` with columns:
   `image_path, label (0=real,1=fake), manipulation (real|Deepfakes|FaceSwap|Face2Face), video_id, group_id, frame_idx, split`

---

## 6. Feature Engineering (the core DIP content)

Every extractor takes a **256×256 BGR uint8 face crop** and returns a 1-D `float32` vector **plus a list of feature names**. `extractor.py` concatenates all groups and exposes `FEATURE_NAMES`. Each function must be deterministic, vectorized where possible, and unit-tested for output length and absence of NaN/Inf (replace with 0 and log if they occur).

Define a shared **face-region masks** helper:
- `interior_mask`: central ellipse covering the inner ~70% of the crop.
- `ring_mask`: annulus between ~70% and ~95% of the ellipse radius. This is where face-swap blending boundaries tend to live, so several features are computed on the ring vs. the interior.

### 6.1 Frequency domain (FFT + DCT), `frequency.py`
Face-swap pipelines resample and blend, leaving unusual high-frequency energy.
- Convert to grayscale float, apply a **Hann window** to reduce spectral leakage.
- 2D FFT → `fftshift` → log magnitude spectrum `log(1+|F|)`.
- **Azimuthally averaged power spectrum (radial profile):** average the power over rings of equal radius → resample to **64 bins**, normalize by total energy. (64 features.)
- Scalar spectral features: high-frequency energy ratio (radius > 50% of max), mid/low ratios, spectral centroid, spectral flatness, **slope of the log-log radial profile** (linear fit), peak-to-mean ratio of the high band. (~8 features.)
- **DCT:** global `cv2.dct` on the grayscale image; energy ratios in low / mid / high zig-zag bands, plus mean and std of coefficient magnitudes per band. (~9 features.)

### 6.2 Error Level Analysis + noise residuals, `ela_noise.py`
**ELA**
- Re-save the crop as JPEG in memory at **quality 90** (and a second pass at **75**), reload, compute `abs(original − recompressed)`, scale for visibility.
- Per quality level: mean, std, max, 95th percentile, entropy of the ELA gray image; per-channel means; ring-vs-interior mean ratio; **4×4 grid block means (16 values)**. (~30 features per quality; configurable.)

**Noise residuals**
- Median residual: `img_gray − cv2.medianBlur(img_gray, 3)`.
- **SRM-style high-pass filters:** apply at least 3 fixed kernels (first-order, second-order and the 3rd-order "SQUARE 5×5" kernel from the Spatial Rich Model), truncate responses to ±2 (T=2).
- Per residual: mean, variance, skewness, kurtosis, and variance computed on a 2×2 grid; ring/interior variance ratio. Also Laplacian variance (blur/sharpness proxy). (~30 features.)

### 6.3 Texture, `texture.py`
- **LBP** (`skimage.feature.local_binary_pattern`, method `uniform`): (P=8, R=1) → 10-bin histogram and (P=16, R=2) → 18-bin histogram, computed on the interior and on the ring separately, L1-normalized. (~56 features.)
- **GLCM** (`skimage.feature.graycomatrix`): quantize to 32 gray levels, distances `[1, 2]`, angles `[0, 45, 90, 135]°`; properties contrast, dissimilarity, homogeneity, energy, correlation, ASM; average over angles → 12 values per property set. (~12 features.)
- **Gabor bank:** 4 orientations × 3 spatial frequencies (`cv2.getGaborKernel`); mean and std of magnitude response for each → 24 features.

### 6.4 Color and edge, `color_edge.py`
- Convert to **YCbCr** and **HSV**. Per channel: mean, std, skewness. (18 features.)
- Chrominance (Cb, Cr) 16-bin histograms (L1-normalized). (32 features.)
- **Canny** edges (thresholds from Otsu-based median rule): overall edge density, ring density, interior density, ring/interior ratio.
- **Sobel** gradient magnitude: mean, std, 95th percentile; 8-bin **edge-orientation histogram**. (~16 features.)

### 6.5 Feature vector
- Total expected: **~250–300 features**. Print the exact count and the group-wise breakdown after extraction.
- Extraction for the full dataset must be **parallelized** with `joblib.Parallel(n_jobs=-1)` and **cached** to `data/features/features.npz` (X, y, groups, split, feature_names, image_paths). Re-run only when the config or code hash changes.

---

## 7. Training, Evaluation and Analysis

### 7.1 Preprocessing and models
Pipeline (scikit-learn `Pipeline`):
1. `VarianceThreshold` (drop constant features)
2. `StandardScaler` (or `RobustScaler`; make it a hyperparameter)
3. Optional `SelectKBest(f_classif or mutual_info_classif)` (k tuned)
4. Classifier

Models to train and compare:
- Logistic Regression (baseline)
- **SVM (RBF)**, with `probability=True`
- **Random Forest**
- **XGBoost** (`tree_method="hist"`, `n_jobs=-1`)
- **Soft-voting ensemble** of the best versions of SVM + RF + XGBoost

### 7.2 Hyperparameter search
- `GridSearchCV` or `RandomizedSearchCV` with **`GroupKFold(n_splits=5)`** on the **train split only**, using `group_id`. Scoring: ROC-AUC.
- Keep grids small enough for CPU (log the total fit count).
- Evaluate on the **validation split** to choose the final model; report on the **test split exactly once**.

### 7.3 Metrics and outputs (`evaluate.py`)
For each model on validation and test: accuracy, precision, recall, F1, ROC-AUC, average precision, confusion matrix.
Additionally: **per-manipulation recall** (Deepfakes / FaceSwap / Face2Face), to show which fakes are hardest.
Save:
- `models/best_model.joblib` (the full pipeline, including scaler and selector)
- `models/metadata.json`: model name, hyperparameters, feature count, feature names, training date, dataset counts, test metrics, decision threshold, library versions
- `reports/tables/model_comparison.csv` and `.md`

**Decision threshold:** default 0.5, but also compute the threshold that maximizes Youden's J on validation. Store in metadata and let the backend use it.

### 7.4 Ablation study (`ablation.py`)
Retrain the best classifier configuration with:
- Each feature group **alone** (frequency, ELA/noise, texture, color/edge)
- **Leave-one-group-out**
- All groups

Output a table plus a bar chart (accuracy and AUC). This is the key DIP-relevant result, because it shows which classical techniques actually contribute.

### 7.5 Robustness tests (`robustness.py`)
Apply to the **test images** (not the train set) and re-evaluate the saved model:
- JPEG recompression at quality 95, 75, 50
- Gaussian blur (σ = 1, 2)
- Additive Gaussian noise (σ = 5, 10)
- Resize down to 50% and back up

Output a table and line chart of accuracy/AUC per perturbation. Expect degradation, especially for JPEG-based features; report this honestly.

### 7.6 Figures for the report (`make_figures.py`)
Generate PNGs into `reports/figures/` (dpi 200, consistent style):
1. Sample grid: real vs each manipulation type
2. For one real and one fake: original, FFT log-spectrum, radial profile overlay, ELA map, noise residual, LBP map, Canny edges (side by side). Include a pipeline diagram if easy.
3. Mean radial power spectra: real vs fake (with std band)
4. ROC curves for all models
5. Confusion matrix of the best model
6. Feature-group ablation chart
7. Top-20 feature importances (RF/XGBoost) with readable feature names
8. Robustness chart
9. t-SNE or PCA 2-D scatter of the standardized features, colored by label

---

## 8. Backend API (FastAPI)

### 8.1 General
- Base path `/api`. JSON responses. CORS allow `http://localhost:5173` (configurable).
- Load the model **once at startup** (lifespan handler). If the model file is missing, `/api/health` reports `model_loaded: false` and analyze endpoints return **503** with a helpful message.
- Accept `image/jpeg`, `image/png`, `image/webp`; max **10 MB** per file; validate magic bytes (not just the extension) by decoding with Pillow, and reject corrupt files with 422.
- Handle EXIF orientation (apply `ImageOps.exif_transpose`) before processing.
- Run the CPU-heavy work in a threadpool (`run_in_threadpool`) so the event loop is not blocked.

### 8.2 Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `{status, model_loaded, model_name, version}` |
| GET | `/api/model/info` | Contents of `metadata.json` (metrics, feature count, dataset counts) |
| POST | `/api/analyze` | Single image upload (`multipart/form-data`, field `file`) → result |
| POST | `/api/analyze/batch` | Up to **20** files (field `files`) → list of per-file results (individual failures must not fail the whole batch) |
| GET | `/api/history` | Query: `page`, `page_size`, `label` (real/fake), `q` (filename search), `sort` → paginated list |
| GET | `/api/history/{id}` | One result |
| DELETE | `/api/history/{id}` | Delete record + stored files |
| DELETE | `/api/history` | Clear all (frontend asks for confirmation) |
| GET | `/api/history/export.csv` | CSV export |
| GET | `/api/files/{id}/thumbnail` | Serve stored thumbnail |

### 8.3 Analyze result schema
```json
{
  "id": "uuid",
  "filename": "photo.jpg",
  "created_at": "2026-10-06T15:30:00Z",
  "face_detected": true,
  "face_box": {"x": 120, "y": 80, "w": 200, "h": 200},
  "verdict": "fake",
  "fake_probability": 0.87,
  "confidence": 0.87,
  "threshold": 0.5,
  "model_name": "VotingEnsemble",
  "processing_ms": 640,
  "warnings": ["Low resolution image; accuracy may be reduced"],
  "thumbnail_url": "/api/files/uuid/thumbnail"
}
```
Rules:
- `confidence` = probability of the *predicted* class.
- Verdict values: `real`, `fake`, or `inconclusive` (when no face is detected → `face_detected=false`, verdict `inconclusive`, probability `null`, with a clear message; do **not** run the classifier on the whole image).
- Add warnings for: small face (< 64 px), small image (< 200 px), a very high compression estimate if easily available.
- Optional stretch (flag in `config.yaml`: `include_evidence`): include base64 thumbnails of the **ELA map and FFT spectrum**, which the frontend can show in an expandable "Evidence" panel. This is valuable for a DIP demo but not required.

### 8.4 Persistence
- SQLite via SQLModel: table `analysis` (id, filename, created_at, verdict, fake_probability, confidence, face_detected, face_box JSON, processing_ms, warnings JSON, upload_path, thumbnail_path, model_name).
- Store uploads under `backend/storage/uploads/` with UUID names (never trust client filenames for paths) and **256px-max thumbnails** under `thumbnails/`.

### 8.5 Inference parity (critical)
`services/predictor.py` **must import and reuse** `ml/features/extractor.py` and the same face-crop code as training, so that train-time and serve-time preprocessing are identical. Add a test that runs a saved dataset image through both the offline path and the API path and asserts the probabilities match within 1e-6.

### 8.6 Backend tests (pytest)
- Health endpoint; analyze with a valid image, an invalid file, an oversized file, and an image with no face.
- Batch with a mix of valid and invalid files.
- History pagination, filtering, deletion, and CSV export.
- Inference parity test (above).

---

## 9. Frontend (React + Vite + TypeScript)

### 9.1 Design direction
A clean, modern, "forensic lab" feel; professional, not a template look.
- **Dark theme by default** with a light-theme toggle (store the choice in `localStorage`).
- Design tokens as CSS variables: background, surface, border, text, muted, accent (a cool cyan/teal), `--real` (green), `--fake` (red/orange), `--warn` (amber).
- Typography: Inter (UI) + JetBrains Mono (numbers and technical labels), loaded via a font CDN or `@fontsource`.
- Generous spacing, 12–16px radius cards, soft borders, subtle shadows, minimal motion (150–250 ms transitions, respect `prefers-reduced-motion`).
- Fully **responsive** (mobile, tablet and desktop) and **keyboard accessible** (focus rings, ARIA labels, semantic HTML, WCAG AA contrast).

### 9.2 Layout
- Top navigation bar: logo + name "ForensiScan", links **Analyze · Batch · History · About**, theme toggle, API status dot (green if `/api/health` says model loaded; red otherwise, with a tooltip).
- Content max-width ~1100px, centered. Footer with a short disclaimer.

### 9.3 Pages

**1. Analyze (`/`)**
- Large drag-and-drop zone (`react-dropzone`) with click-to-browse and paste-from-clipboard support. Accept jpg/png/webp, client-side size check (10 MB) with friendly errors.
- After selecting: image preview, filename and size, "Analyze" button (auto-analyze optional toggle).
- Loading state: animated scanning effect over the image + skeleton for the result card.
- **Result card:**
  - Big verdict badge: **REAL** (green) / **FAKE** (red) / **INCONCLUSIVE** (amber).
  - **Confidence gauge** (semicircle or horizontal bar) showing the fake probability with the threshold marker; numeric percentage in mono font.
  - Face bounding box drawn on the preview (SVG/canvas overlay) when a face was detected.
  - Processing time, model name, and warning chips.
  - Collapsible **"Evidence"** panel (only if the backend provides ELA/FFT images) with captions explaining what each image shows.
  - Buttons: "Analyze another", "View in history".
- Error states: network failure, 503 model not loaded (explain how to train), 422 invalid file; each with a clear message and a retry.

**2. Batch (`/batch`)**
- Multi-file drop zone (up to 20 files). Show a queue list with thumbnails, per-file status (queued / analyzing / done / failed) and a progress bar for the whole batch.
- Results table: thumbnail, filename, verdict badge, fake probability bar, processing time. Sortable by probability and verdict; summary header ("14 real · 5 fake · 1 inconclusive").
- Actions: **Export CSV**, "Clear", and per-row "view details" drawer.

**3. History (`/history`)**
- Paginated grid/table toggle of past analyses with thumbnails, filter chips (All / Real / Fake / Inconclusive), a filename search box (debounced), and sort control.
- Click an item → side drawer with the full details (same result card).
- Delete single item (with confirmation) and "Clear all" (confirmation dialog). Export CSV.
- Empty state with an illustration/message and a link to Analyze.

**4. About (`/about`)**
- Short, readable explanation of the method: the pipeline diagram (static SVG/image) and one short card per technique (FFT, ELA, noise residuals, LBP/GLCM/Gabor, color/edge) explaining what artifact it looks for.
- **Model card:** data from `/api/model/info` (dataset, test accuracy/AUC/F1, feature count, date).
- **Limitations and ethics disclaimer** (see Section 12), displayed prominently.

### 9.4 Frontend technical requirements
- Typed API client in `src/api/client.ts` using `fetch`; base URL from `VITE_API_URL` (default `http://localhost:8000`). Types mirror backend schemas in `src/types/`.
- TanStack Query for history and model info (cache and invalidation after upload/delete). Mutations for analyze.
- Reusable components: `Dropzone`, `VerdictBadge`, `ConfidenceGauge`, `ResultCard`, `FaceOverlay`, `ImageThumb`, `Drawer`, `ConfirmDialog`, `Toast`, `EmptyState`, `Skeleton`, `ThemeToggle`, `ApiStatus`.
- Global error boundary and toast notifications; no unhandled promise rejections.
- Release the memory of object URLs (`URL.revokeObjectURL`) after use.
- ESLint + Prettier configured; `npm run build` must pass with zero TypeScript errors.
- A few Vitest tests: `VerdictBadge`, `ConfidenceGauge`, and `Dropzone` file validation.

---

## 10. Configuration (`config.yaml`)
```yaml
seed: 42
paths:
  raw_data: data/raw
  faces: data/faces
  manifest: data/manifest.csv
  features: data/features/features.npz
  models: models
data:
  manipulations: [Deepfakes, FaceSwap, Face2Face]
  quality: c23
  subset: {train_ids: 100, val_ids: 20, test_ids: 20}
  frames_per_video: 10
  face_size: 256
  face_margin: 1.3
features:
  ela_qualities: [90, 75]
  fft_radial_bins: 64
  lbp: [[8, 1], [16, 2]]
training:
  cv_folds: 5
  scoring: roc_auc
backend:
  max_upload_mb: 10
  max_batch_files: 20
  cors_origins: ["http://localhost:5173"]
  include_evidence: false
```

---

## 11. Commands the Project Must Support
Document these in `README.md` and verify each works:

```bash
# one-time setup
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

# ML pipeline
python -m ml.data.extract_frames
python -m ml.data.build_manifest
python -m ml.features.extractor        # builds and caches features
python -m ml.train
python -m ml.evaluate
python -m ml.ablation
python -m ml.robustness
python -m ml.make_figures

# backend
uvicorn backend.app.main:app --reload --port 8000

# frontend
cd frontend && npm install && npm run dev      # http://localhost:5173
```
Also provide a single `python run_all.py` (or `make all`) that runs the ML pipeline in order and stops with a clear error if the raw data is missing.

---

## 12. Limitations and Ethics (must appear in the UI and README)
- The model is trained on FaceForensics++ face-swap data. It **will not generalize reliably** to new deepfake generators, diffusion-generated faces, or heavily compressed social-media images.
- Results are **probabilistic indicators, not proof**. Do not use for legal, journalistic or personal accusations.
- Classical features such as ELA are sensitive to JPEG history; re-saved real images can look suspicious and vice versa.
- Uploaded images are stored locally for the history feature; provide a way to delete them.
- Dataset use must follow the FaceForensics++ terms (research/education only). **Do not commit dataset files to the repo.**

---

## 13. Code Quality Rules
- Type hints and docstrings on all public Python functions; `ruff` (or flake8) + `black` formatting.
- No hard-coded absolute paths; no magic numbers outside `config.yaml` or named constants.
- Log with the `logging` module (no stray `print` except CLI summaries).
- Never swallow exceptions silently; return meaningful HTTP errors.
- Keep functions small and unit-testable; add tests as you go (target: feature extractors, splits leakage test, API endpoints, inference parity).
- Commit-sized, clearly named steps; keep `README.md` updated at the end of each phase.

---

## 14. Definition of Done (Acceptance Checklist)
- [ ] Pipeline runs end to end on real FF++ data and produces `models/best_model.joblib` and `metadata.json`.
- [ ] Split-leakage test passes (no group in two splits).
- [ ] Test-set metrics table (all models), ablation table/chart, robustness table/chart, and the 9 report figures exist in `reports/`.
- [ ] Best model test ROC-AUC is reported honestly (no tuning on the test set).
- [ ] All backend tests pass; inference parity test passes.
- [ ] `/api/analyze` responds in under ~2 s on CPU for a typical image.
- [ ] Frontend: Analyze, Batch, History and About pages work against the live backend; all empty/error/loading states implemented.
- [ ] `npm run build` has zero errors; layout verified at 375px, 768px and 1280px widths; dark and light themes both usable.
- [ ] README has setup, data-access instructions, commands, screenshots and limitations.

---

## 15. Build Phases (execute in order)

**Phase 0: Scaffold.** Create the repo structure, `config.yaml`, requirements, linting, `.gitignore` (data, models, storage, node_modules). Add `make_dummy_data.py` so every later phase can be smoke-tested.
*Verify:* imports succeed; `pytest` runs (even if few tests).

**Phase 1: Data pipeline.** Frame extraction, face detection/cropping, manifest, group-aware splits, leakage test.
*Verify:* manifest has the expected columns and balanced classes; spot-check 10 random crops visually; leakage test passes.

**Phase 2: Feature extractors.** Implement the four groups plus the combined extractor, with unit tests, caching and parallel execution.
*Verify:* consistent vector length, no NaN/Inf, feature count printed, one image takes well under 1 s.

**Phase 3: Training and evaluation.** Pipelines, GroupKFold search, model comparison, threshold selection, save artifacts.
*Verify:* metrics files written; test set touched only once; training under ~30 min on CPU.

**Phase 4: Analysis.** Ablation, robustness, and figure generation scripts.
*Verify:* all tables and the 9 figures exist and are legible.

**Phase 5: Backend.** FastAPI app, predictor service with inference parity, persistence, all endpoints, tests.
*Verify:* `pytest backend/tests` green; manual curl of `/api/analyze` returns the schema in 8.3.

**Phase 6: Frontend.** Design tokens and layout shell → Analyze page → Batch → History → About → polish (animations, a11y, responsive, error states) → tests.
*Verify:* build passes; walk through every page with the backend running; run the checklist in Section 14.

**Phase 7: Docs and wrap-up.** README with screenshots, `docs/DECISIONS.md`, final full run of `run_all.py`, then tick through Section 14.
