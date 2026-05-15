# Requirements Document

## Introduction

This document defines requirements for a complete, working demo system for Pakistan Sign Language (PSL) recognition covering 5 signs: **condolences**, **Electric_Swtich**, **First_aid_box**, **Human_Rights**, and **Quraan**.

The system uses a MediaPipe keypoint extraction pipeline feeding a Transformer encoder trained with triplet loss. Given the extreme scarcity of PSL data (only 1 raw video per class), the demo must maximise recognition quality through aggressive data augmentation, a unified and consistent model configuration, a proper leave-one-out evaluation strategy, and a real-time webcam demo with honest confidence reporting.

The requirements address six areas:
1. Data augmentation to expand the 2-sample-per-class dataset
2. Model configuration consistency across all scripts
3. A correct, honest evaluation pipeline (no self-evaluation)
4. A fixed inference pipeline (`test.py`)
5. A real-time webcam demo
6. A unified entry-point script that orchestrates the full pipeline

---

## Glossary

- **System**: The PSL Demo System as a whole.
- **Pipeline**: The ordered sequence of processing steps from raw video to recognised sign.
- **Augmentor**: The component responsible for generating augmented keypoint sequences from existing JSON samples.
- **Trainer**: The component responsible for training the Transformer embedding model.
- **Evaluator**: The component responsible for computing honest held-out accuracy metrics.
- **Inferencer**: The component responsible for loading a trained model and classifying a single video or webcam stream.
- **Demo**: The real-time webcam component that displays live sign predictions to the user.
- **Database**: The in-memory set of labelled embeddings built from training samples and used for nearest-neighbour classification.
- **Embedding**: A 128-dimensional L2-normalised vector produced by the Transformer encoder for a keypoint sequence.
- **Confidence**: The cosine similarity score (range −1 to 1) between a query embedding and its nearest database embedding.
- **Confidence Threshold**: The minimum cosine similarity below which the Inferencer reports "Unknown" instead of a class label.
- **LOO-CV**: Leave-One-Out Cross-Validation — for each sample, train on all other samples and test on the held-out sample.
- **SEQ_LEN**: Fixed sequence length of 100 frames used throughout the Pipeline.
- **INPUT_DIM**: Fixed keypoint dimensionality of 225 (33 pose × 3 + 21 × 2 hands × 3).
- **MODEL_DIM**: Fixed internal Transformer dimension of 128.
- **EMBED_DIM**: Fixed output embedding dimension of 128.
- **NUM_HEADS**: Fixed number of attention heads of 4.
- **NUM_LAYERS**: Fixed number of Transformer encoder layers of 2.
- **SMOOTH_ALPHA**: Exponential smoothing factor of 0.2 applied to raw keypoint sequences.

---

## Requirements

### Requirement 1: Unified Model Configuration

**User Story:** As a developer, I want all scripts to share a single model configuration, so that loading a saved checkpoint never fails due to architecture mismatches.

#### Acceptance Criteria

1. THE System SHALL define MODEL_DIM=128, EMBED_DIM=128, NUM_HEADS=4, NUM_LAYERS=2, SEQ_LEN=100, and INPUT_DIM=225 in a single shared configuration module (`config.py`).
2. THE Trainer, Evaluator, Inferencer, and Demo SHALL each import model hyperparameters exclusively from `config.py` and SHALL NOT redefine them locally.
3. WHEN `transformer_embedding.pt` is loaded by any script, THE Inferencer SHALL successfully reconstruct the model architecture using the values from `config.py` without raising a shape mismatch error.

---

### Requirement 2: Data Augmentation

**User Story:** As a researcher, I want the 2 existing keypoint sequences per class to be expanded through augmentation, so that the Trainer has enough variety to learn discriminative embeddings.

#### Acceptance Criteria

1. THE Augmentor SHALL generate at least 8 augmented sequences per original JSON sample, producing a minimum of 16 sequences per class before training.
2. WHEN generating an augmented sequence, THE Augmentor SHALL apply at least one of the following transforms independently: Gaussian noise injection (σ ≤ 0.02), temporal jitter (random frame drop of up to 10%), uniform time-scaling (speed factor in [0.8, 1.2]), horizontal mirroring of x-coordinates, and random rotation of the signer's body by ±15 degrees in the xy-plane.
3. THE Augmentor SHALL ensure every augmented sequence has exactly SEQ_LEN frames after augmentation, using the same padding strategy as `dataset__creation.py`.
4. THE Augmentor SHALL save augmented sequences as JSON files in `dataset_augmented/{class_name}/` using the naming convention `{original_stem}_aug_{n}.json`.
5. IF an augmented sequence contains fewer than 10 non-zero frames after augmentation, THEN THE Augmentor SHALL discard that sequence and generate a replacement.

---

### Requirement 3: Correct Evaluation with Leave-One-Out Cross-Validation

**User Story:** As a researcher, I want an honest accuracy estimate that does not test on training data, so that I can trust the reported performance numbers.

#### Acceptance Criteria

1. THE Evaluator SHALL implement Leave-One-Out Cross-Validation (LOO-CV) over all available sequences (original + augmented).
2. WHEN running LOO-CV, THE Evaluator SHALL ensure the held-out sample is never present in the Database used for nearest-neighbour classification.
3. THE Evaluator SHALL report per-class accuracy, overall accuracy, and a confusion matrix to stdout and save them to `evaluation_report.txt`.
4. THE Evaluator SHALL report the LOO-CV accuracy as the primary performance metric and SHALL NOT report accuracy computed by querying a sample against a Database that contains that same sample.
5. WHEN the LOO-CV accuracy is below 0.60, THE Evaluator SHALL print a warning: "WARNING: accuracy below 60% — consider collecting more data or tuning augmentation."

---

### Requirement 4: Fixed Inference Pipeline (`test.py`)

**User Story:** As a developer, I want `test.py` to correctly load JSON sequences, match predicted labels to folder names, and apply a confidence threshold, so that offline video testing produces valid results.

#### Acceptance Criteria

1. THE Inferencer SHALL import the `json` module at the top of the file, before any function that calls `json.load`.
2. WHEN building the Database, THE Inferencer SHALL derive the class label directly from `sign_folder.name` without any string splitting or transformation.
3. WHEN classifying a test video, THE Inferencer SHALL extract the true label from the parent folder name of the video file, not from the filename stem.
4. WHEN the Confidence of the nearest-neighbour match is below the Confidence Threshold (default 0.50), THE Inferencer SHALL output "Unknown" as the predicted label.
5. THE Inferencer SHALL print, for each test video: the filename, the true label, the predicted label, the confidence score, and whether the prediction is correct.
6. WHEN no test videos are found in the TEST_DIR, THE Inferencer SHALL print "No test videos found in {TEST_DIR}" and exit with code 0.

---

### Requirement 5: Real-Time Webcam Demo

**User Story:** As a user, I want to perform a sign in front of my webcam and see the recognised sign displayed on screen in real time, so that I can demonstrate the system to others.

#### Acceptance Criteria

1. THE Demo SHALL open the default webcam (device index 0) and display a live video feed in a window titled "PSL Demo".
2. THE Demo SHALL accumulate a rolling buffer of the most recent SEQ_LEN frames of MediaPipe keypoints extracted from the webcam feed.
3. WHEN the rolling buffer contains exactly SEQ_LEN frames, THE Demo SHALL run inference and display the predicted label and confidence score overlaid on the video frame.
4. WHEN the Confidence is below the Confidence Threshold, THE Demo SHALL display "Unknown" instead of a class label.
5. THE Demo SHALL display a visual indicator showing how full the rolling buffer is (e.g., a progress bar), so the user knows when a prediction will be made.
6. WHEN the user presses the 'q' key, THE Demo SHALL release the webcam and close all windows cleanly.
7. IF the webcam cannot be opened, THEN THE Demo SHALL print "Error: cannot open webcam" and exit with code 1.
8. THE Demo SHALL achieve a display frame rate of at least 15 FPS on a CPU-only machine with a modern processor.

---

### Requirement 6: Retrained Model Consistent with Fixed Configuration

**User Story:** As a developer, I want the model to be retrained using the unified configuration and augmented dataset, so that the saved checkpoint is compatible with all inference scripts.

#### Acceptance Criteria

1. THE Trainer SHALL load sequences from both `dataset_final/` (original) and `dataset_augmented/` (augmented) when building the training set.
2. THE Trainer SHALL use MODEL_DIM=128, NUM_LAYERS=2, NUM_HEADS=4, EMBED_DIM=128 as defined in `config.py`.
3. THE Trainer SHALL apply dropout of 0.1 during training (reduced from 0.3 to suit the small dataset size).
4. THE Trainer SHALL save the trained model to `transformer_embedding.pt` upon completion.
5. THE Trainer SHALL log epoch number, triplet loss, and validation accuracy to `training_log.csv` after each epoch.
6. WHEN training completes, THE Trainer SHALL print the final validation accuracy and the path of the saved model.

---

### Requirement 7: Unified Entry-Point Script

**User Story:** As a user, I want a single script that runs the full pipeline end-to-end, so that I can reproduce the demo from scratch with one command.

#### Acceptance Criteria

1. THE System SHALL provide a `run_pipeline.py` script that accepts a `--mode` argument with values: `augment`, `train`, `evaluate`, `test`, `demo`, and `all`.
2. WHEN `--mode all` is specified, THE System SHALL execute the steps in order: augmentation → training → evaluation → demo.
3. WHEN `--mode demo` is specified, THE System SHALL verify that `transformer_embedding.pt` exists and, IF it does not exist, THEN THE System SHALL print "Model not found — run with --mode train first" and exit with code 1.
4. THE System SHALL print the start time, end time, and elapsed time for each pipeline step when running in `--mode all`.
