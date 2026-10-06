# Decisions

- FaceForensics++ files are never downloaded or committed. `ml.data.make_dummy_data` is retained only as an explicitly named plumbing smoke test and is never called by `run_all.py`.
- The default pipeline fails closed when real FF++ videos, official split JSONs, sampled frames, or a manifest are missing. This prevents synthetic results from being mistaken for report results.
- The current shared extractor remains deterministic and CPU-only; its cache records the real image paths, labels, groups, splits, feature names, and configuration hash.
