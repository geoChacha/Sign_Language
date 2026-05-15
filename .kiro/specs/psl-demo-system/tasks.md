# Implementation Plan: PSL Demo System

## Overview

Implement the PSL Demo System in Python by creating five new files (`config.py`, `augment.py`, `evaluate.py`, `demo.py`, `run_pipeline.py`), updating two existing files (`train.py`, `test.py`), and adding a property-based test suite (`tests/test_psl.py`). The implementation follows a strict dependency order: shared config first, augmentation second, training third, then evaluation/inference/demo in parallel, and the unified entry point last.

## Tasks

- [x] 1. Create `config.py` — shared constants and paths
  - Define all model hyperparameters: `MODEL_DIM=128`, `EMBED_DIM=128`, `NUM_HEADS=4`, `NUM_LAYERS=2`, `SEQ_LEN=100`, `INPUT_DIM=225`
  - Define inference constant: `CONFIDENCE_THRESHOLD=0.50`
  - Define path constants: `DATASET_DIRS`, `MODEL_PATH`, `AUGMENTED_DIR`, `EVAL_REPORT_PATH`, `TRAINING_LOG_PATH`
  - _Requirements: 1.1_

- [x] 2. Create `augment.py` — keypoint augmentation
  - [x] 2.1 Implement `augment_sequence(seq: np.ndarray) -> np.ndarray`
    - Apply each of the five transforms independently with probability 0.5: Gaussian noise (σ=0.01), temporal jitter (drop up to 10% of frames), time-scaling (factor ∈ [0.8, 1.2]), horizontal mirror (negate every 3rd column starting at 0), rotation ±15° in xy-plane
    - Re-pad or truncate result to exactly `(SEQ_LEN, INPUT_DIM)` after each transform
    - _Requirements: 2.2, 2.3_

  - [ ]* 2.2 Write property test for augmented sequence shape invariant
    - **Property 1: Augmented sequence shape invariant**
    - **Validates: Requirements 2.3**
    - Generate random `(100, 225)` float32 arrays with `hypothesis.extra.numpy.arrays`; apply each transform individually and via `augment_sequence`; assert output shape is `(100, 225)` in all cases
    - Tag: `# Feature: psl-demo-system, Property 1: Augmented sequence shape invariant`

  - [ ]* 2.3 Write property test for augmented sequence validity
    - **Property 2: Augmented sequence validity**
    - **Validates: Requirements 2.5**
    - Generate random sequences, apply `augment_sequence`, assert `np.sum(np.any(seq != 0, axis=1)) >= 10`
    - Tag: `# Feature: psl-demo-system, Property 2: Augmented sequence validity`

  - [x] 2.4 Implement `run_augmentation(source_dir, output_dir, augmentations_per_sample=8) -> None`
    - Iterate all JSON files in `source_dir/{class_name}/`; for each file call `augment_sequence` up to 3 times if validity check fails; write valid results as `{original_stem}_aug_{n}.json` in `output_dir/{class_name}/`
    - Skip and log a warning for any sequence that remains invalid after 3 attempts
    - _Requirements: 2.1, 2.4, 2.5_

  - [ ]* 2.5 Write property test for augmentation output count
    - **Property 3: Augmentation output count**
    - **Validates: Requirements 2.1, 2.4**
    - Use a temporary directory with synthetic JSON files; run `run_augmentation` with a fixed `augmentations_per_sample=K`; assert exactly `N × K` files are written
    - Tag: `# Feature: psl-demo-system, Property 3: Augmentation output count`

  - [ ]* 2.6 Write property test for augmentation naming convention
    - **Property 8: Augmentation naming convention**
    - **Validates: Requirements 2.4**
    - Assert every output filename matches the pattern `{original_stem}_aug_{n}.json` for n ∈ [1, augmentations_per_sample]
    - Tag: `# Feature: psl-demo-system, Property 8: Augmentation naming convention`

- [x] 3. Checkpoint — verify augmentation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Update `train.py` — fix config, load augmented data, fix dropout
  - [x] 4.1 Replace all local hyperparameter definitions with imports from `config.py`
    - Remove local `MODEL_DIM=64`, `EMBED_DIM`, `NUM_HEADS`, `NUM_LAYERS`, `SEQ_LEN`, `INPUT_DIM` constants
    - Add `from config import MODEL_DIM, EMBED_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN, INPUT_DIM, DATASET_DIRS, MODEL_PATH, TRAINING_LOG_PATH`
    - _Requirements: 1.2_

  - [x] 4.2 Update `TransformerEmbedding` and `PositionalEncoding` to use imported constants
    - Change `dropout=0.3` → `dropout=0.1` in `TransformerEncoderLayer`
    - Change `num_layers=2` to use `NUM_LAYERS` from config (already 2, but now sourced from config)
    - _Requirements: 6.2, 6.3_

  - [x] 4.3 Update `load_dataset()` to accept `dirs: list[str]` and load from all directories
    - Signature: `load_dataset(dirs: list[str] = DATASET_DIRS) -> dict[str, list[np.ndarray]]`
    - Iterate over each directory in `dirs`; merge samples per class across directories
    - _Requirements: 6.1_

  - [x] 4.4 Update `train()` to use the new `load_dataset` signature and save to `MODEL_PATH` from config
    - Pass `DATASET_DIRS` to `load_dataset`; save checkpoint to `MODEL_PATH`; log to `TRAINING_LOG_PATH`
    - Print final validation accuracy and saved model path on completion
    - _Requirements: 6.4, 6.5, 6.6_

  - [ ]* 2.7 Write property test for embedding L2-normalisation invariant
    - **Property 5: Embedding L2-normalisation invariant**
    - **Validates: Requirements 6.2**
    - Generate random `(1, 100, 225)` float32 tensors; pass through `TransformerEmbedding.forward()`; assert `|torch.norm(emb, p=2, dim=1) - 1.0| < 1e-5` for every output
    - Tag: `# Feature: psl-demo-system, Property 5: Embedding L2-normalisation invariant`

- [x] 5. Create `evaluate.py` — LOO-CV evaluation (replaces `emo.py`)
  - [x] 5.1 Implement `run_loo_cv(dirs, model_path) -> dict`
    - Load all `(seq, label)` pairs from `dirs` using the same JSON loading logic as `train.py`
    - For each index `i`: build a database from all other sequences, embed the held-out sequence, find nearest neighbour by cosine similarity, record `(true_label, predicted_label)`
    - Return `{accuracy, per_class_accuracy, confusion_matrix}`
    - Print warning if overall accuracy < 0.60 (Requirement 3.5)
    - _Requirements: 3.1, 3.2, 3.4, 3.5_

  - [ ]* 5.2 Write property test for LOO-CV held-out exclusion
    - **Property 4: LOO-CV held-out exclusion**
    - **Validates: Requirements 3.2, 3.4**
    - Generate a small synthetic dataset (e.g., 3 classes × 3 sequences); instrument `run_loo_cv` or test the database-building logic directly; assert that for each fold `i`, the embedding of sequence `i` is not present in the database used for that fold
    - Tag: `# Feature: psl-demo-system, Property 4: LOO-CV held-out exclusion`

  - [x] 5.3 Implement `save_report(results: dict, path: str) -> None`
    - Write `evaluation_report.txt` in the format specified in the design: overall accuracy, sequences evaluated, per-class accuracy table, confusion matrix, and optional warning line
    - _Requirements: 3.3_

- [x] 6. Fix `test.py` — three bug fixes and public interface
  - [x] 6.1 Fix Bug 1: move `import json` to the top of the file (remove it from inside `if __name__ == "__main__"`)
    - _Requirements: 4.1_

  - [x] 6.2 Fix Bug 2: derive class label from `sign_folder.name` in `build_database()`
    - Replace any `file.stem.split(...)` label extraction with `sign_folder.name`
    - Update `build_database` signature to `build_database(model: TransformerEmbedding) -> tuple[np.ndarray, list[str]]` loading from `DATASET_DIRS`
    - _Requirements: 4.2_

  - [x] 6.3 Fix Bug 3: extract true label from parent folder name in `run_test()`
    - Replace `video_file.stem.split("_")[0]` with folder-based or case-insensitive prefix-match logic
    - _Requirements: 4.3_

  - [x] 6.4 Implement `classify_sequence(seq, db_emb, db_labels, threshold) -> tuple[str, float]`
    - Return `("Unknown", confidence)` when max cosine similarity < `CONFIDENCE_THRESHOLD`; otherwise return `(predicted_label, confidence)`
    - _Requirements: 4.4_

  - [x] 6.5 Implement `run_test(test_dir: str) -> None`
    - Print filename, true label, predicted label, confidence, and correctness for each video
    - Handle empty `test_dir` (print message and exit 0); handle `None` return from `video_to_sequence` (skip with warning)
    - Replace all local hyperparameter definitions with imports from `config.py`
    - _Requirements: 4.5, 4.6, 1.2_

  - [ ]* 6.6 Write property test for confidence threshold — Unknown below threshold
    - **Property 6: Confidence threshold — Unknown below threshold**
    - **Validates: Requirements 4.4, 5.4**
    - Generate random unit-norm embedding pairs where cosine similarity is forced below `CONFIDENCE_THRESHOLD`; assert `classify_sequence` returns `"Unknown"`
    - Tag: `# Feature: psl-demo-system, Property 6: Confidence threshold — Unknown below threshold`

- [x] 7. Checkpoint — verify config, training, evaluation, and inference
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Create `demo.py` — real-time webcam demo
  - [x] 8.1 Implement rolling buffer and MediaPipe keypoint extraction loop
    - Open webcam with `cv2.VideoCapture(0)`; exit with code 1 if not opened
    - Accumulate `deque(maxlen=SEQ_LEN)` of `(INPUT_DIM,)` keypoint vectors per frame
    - _Requirements: 5.1, 5.2, 5.7_

  - [x] 8.2 Implement inference trigger and overlay rendering
    - When `len(buffer) == SEQ_LEN`: stack buffer into `(SEQ_LEN, INPUT_DIM)` array, run `classify_sequence`, display `"Sign: {label}"` and `"Conf: {score:.2f}"` with `cv2.putText`
    - Draw buffer progress bar at bottom of frame; draw FPS counter top-right
    - Display `"Unknown"` when confidence < `CONFIDENCE_THRESHOLD`
    - _Requirements: 5.3, 5.4, 5.5_

  - [x] 8.3 Implement `run_demo(model_path, camera_index) -> None` public interface
    - Load model from `model_path`; build embedding database; enter capture loop; release webcam and close windows on 'q' keypress
    - Import all hyperparameters from `config.py`
    - _Requirements: 5.6, 5.8, 1.2_

- [x] 9. Create `run_pipeline.py` — unified entry point
  - [x] 9.1 Implement argument parsing and mode dispatch
    - `argparse` with `--mode` accepting `augment`, `train`, `evaluate`, `test`, `demo`, `all`
    - Dispatch each mode to the corresponding module function
    - _Requirements: 7.1_

  - [x] 9.2 Implement `--mode all` sequential execution with timing
    - Execute: `augment.run_augmentation()` → `train.train()` → `evaluate.run_loo_cv()` → `demo.run_demo()`
    - Wrap each step with `time.time()` and print `"[step] completed in {elapsed:.1f}s"`
    - _Requirements: 7.2, 7.4_

  - [x] 9.3 Implement model existence check for `--mode demo`
    - Check `Path(MODEL_PATH).exists()`; if missing, print `"Model not found — run with --mode train first"` and `sys.exit(1)`
    - _Requirements: 7.3_

- [ ] 10. Create `tests/test_psl.py` — property-based test suite
  - [ ] 10.1 Set up Hypothesis test file with shared fixtures
    - Import `hypothesis`, `hypothesis.extra.numpy`, `pytest`; import `augment`, `evaluate`, `test` modules
    - Define reusable `@given` strategies for `(100, 225)` float32 arrays and unit-norm embedding vectors
    - _Requirements: 1.1, 2.3_

  - [ ]* 10.2 Implement Property 1 test — shape invariant
    - Already specified in task 2.2; consolidate into `tests/test_psl.py` if not already there
    - _Requirements: 2.3_

  - [ ]* 10.3 Implement Property 2 test — sequence validity
    - Already specified in task 2.3; consolidate into `tests/test_psl.py` if not already there
    - _Requirements: 2.5_

  - [ ]* 10.4 Implement Property 4 test — LOO-CV held-out exclusion
    - Already specified in task 5.2; consolidate into `tests/test_psl.py` if not already there
    - _Requirements: 3.2, 3.4_

  - [ ]* 10.5 Implement Property 5 test — L2-normalisation invariant
    - Already specified in task 2.7 (under train.py); consolidate into `tests/test_psl.py` if not already there
    - _Requirements: 6.2_

  - [ ]* 10.6 Implement Property 6 test — confidence threshold Unknown
    - Already specified in task 6.6; consolidate into `tests/test_psl.py` if not already there
    - _Requirements: 4.4, 5.4_

- [x] 11. Final checkpoint — full pipeline verification
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- `config.py` (Task 1) must be completed before any other task — everything imports from it
- `augment.py` (Task 2) must be completed before `train.py` (Task 4) — training loads augmented data
- `train.py` (Task 4) must be completed before `evaluate.py` (Task 5), `test.py` (Task 6), and `demo.py` (Task 8) — all depend on the saved model
- `run_pipeline.py` (Task 9) must be completed last — it orchestrates all other modules
- Property tests in `tests/test_psl.py` (Task 10) consolidate the inline property sub-tasks from Tasks 2–6; implement them once in the test file
- The existing `emo.py` is superseded by `evaluate.py` but should not be deleted until `evaluate.py` is verified
