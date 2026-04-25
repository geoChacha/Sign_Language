# WLASL-100 ASL Recognition Pipeline

Offline, production-ready sign language recognition for 100 words.
Runs on RTX 3070, exports to Android/iOS via TorchScript and ONNX.

## Architecture

Input video -> MediaPipe Holistic -> (64 frames x 225 keypoints) -> Sign Language Transformer -> 100-class prediction

Feature breakdown per frame (225 total):
- Left hand:  21 landmarks x XYZ = 63
- Right hand: 21 landmarks x XYZ = 63
- Pose body:  33 landmarks x XYZ = 99

Model layers:
1. Separate linear projections for each body part
2. Temporal Convolutional Network (local motion features)
3. CLS token + Positional Encoding
4. 4x Transformer Encoder layers (Pre-LN, GELU)
5. Classification MLP head

Why keypoints instead of raw video CNN?
- Robust to lighting, backgrounds, camera quality (perfect for real-world mobile)
- Model is ~3M params vs 30M+ for video CNN
- Runs 30+ FPS on mobile CPU at inference
- No GPU needed on device

## Quick Start

### 1. Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Edit config.py

Change DATASET_ROOT to point to your Kaggle dataset folder.

### 3. Extract keypoints (one-time, ~1-3 hrs on CPU)

```bash
python 1_extract_keypoints.py
```

Reads every video, runs MediaPipe Holistic, saves .npy files.

### 4. Train (~30-60 min on RTX 3070)

```bash
python 2_train.py
```

Expected results: Top-1 ~65-75%, Top-5 ~90%+

### 5. Evaluate

```bash
python 3_evaluate.py
```

### 6. Export for mobile

```bash
python 4_export_mobile.py
```

Produces: model_scripted.ptl (PyTorch Mobile), model.onnx, model_quantized.pt

### 7. Real-time inference

```bash
python 5_inference.py              # webcam
python 5_inference.py --video x.mp4  # video file
```

## Mobile Integration

### Android (PyTorch Mobile)

```kotlin
implementation 'org.pytorch:pytorch_android_lite:2.1.0'

val module = LiteModuleLoader.load("model_scripted.ptl")
val input  = Tensor.fromBlob(floatArray, longArrayOf(1, 64, 225))
val output = module.forward(IValue.from(input)).toTuple()
```

### ONNX Runtime (cross-platform)

```kotlin
val session = OrtSession(env, readBytes("model.onnx"), OrtSession.SessionOptions())
val tensor  = OnnxTensor.createTensor(env, FloatArray(64*225), longArrayOf(1,64,225))
val result  = session.run(mapOf("keypoints" to tensor))
```

## Files

| File | Description |
|------|-------------|
| config.py | All hyperparameters and paths - edit this first |
| model.py | Transformer architecture |
| dataset.py | DataLoader with augmentation |
| 1_extract_keypoints.py | MediaPipe keypoint extraction |
| 2_train.py | Training loop (AMP, warmup, early stopping) |
| 3_evaluate.py | Evaluation + confusion matrix |
| 4_export_mobile.py | TorchScript + ONNX export |
| 5_inference.py | Real-time webcam/video inference |
| utils.py | Training curve plots, dataset stats |

## Tips for Higher Accuracy

- Increase NUM_LAYERS to 6 and D_MODEL to 512 in config.py
- Verify keypoint quality with: python utils.py (reports detection rates)
- WLASL is genuinely hard (many similar signs); Top-5 ~90% is strong
- Consider fine-tuning on your own recorded samples for deployment words
