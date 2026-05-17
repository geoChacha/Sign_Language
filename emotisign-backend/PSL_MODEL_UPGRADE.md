# PSL Model Upgrade: OLD → NEW

## Summary

The PSL alphabet recognition model has been upgraded from the old 23-class model to the NEW 37-class model with significantly improved coverage and accuracy.

## Changes

### Model Specifications

| Aspect | OLD PSL | NEW PSL |
|--------|---------|---------|
| **Classes** | 23 | 37 (full Urdu alphabet) |
| **Accuracy** | Unknown | **99.41%** |
| **Architecture** | 42 → 128 → 64 → 23 | 42 → 128 → 64 → 37 |
| **Format** | PyTorch .pt | sklearn MLP → converted to PyTorch .pt |
| **Training Samples** | Unknown | 5,112 |

### Alphabet Coverage

**OLD PSL (23 letters):**
```
ا ب پ ت ٹ ث ج چ ح خ د ڈ ذ ر ڑ ز ژ س ش ص ض ط ظ
```

**NEW PSL (37 letters — FULL Urdu alphabet):**
```
ء ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ل م ن و
ٹ پ چ ڈ ڑ ژ ک گ ہ ی ے
```

**Added letters (14 new):**
```
ء ع غ ف ق ل م ن و ک گ ہ ی ے
```

## Service Improvements

The `psl_live_service.py` was also upgraded with three critical real-world usability fixes:

### 1. CLAHE Preprocessing
- **What:** Contrast-limited adaptive histogram equalization in LAB color space
- **Why:** MediaPipe hand detection fails under dim/uneven lighting without it
- **Impact:** Hand detection rate improves dramatically in typical indoor lighting

### 2. Lower Confidence Threshold (0.80 → 0.60)
- **What:** Reduced the minimum confidence required to show predictions
- **Why:** With 37 classes, softmax probabilities are naturally diluted; correct predictions often land at 0.65–0.75
- **Impact:** Far fewer "no prediction" moments; users actually see the system responding

### 3. Temporal Smoothing (Majority Vote)
- **What:** Rolling 12-frame buffer with 60% consensus requirement before emitting results
- **Why:** Single-frame predictions flicker due to micro-movements and landmark jitter
- **Impact:** Stable, non-flickering predictions; natural "hold to confirm" behavior

## Files Modified

### Core Service
- `app/ml/psl_live_service.py` — upgraded with CLAHE, smoothing, and lower threshold
- `app/ml/models/psl/alphabet_classifier.pt` — replaced with NEW 37-class model
- `app/ml/models/psl/label_map.json` — updated to 37 Urdu letters

### Scripts
- `scripts/convert_new_psl_model.py` — converts sklearn JSON to PyTorch .pt
- `scripts/test_new_psl_model.py` — verifies model loads and runs correctly

## Migration

No code changes required in:
- WebSocket router (`app/routers/realtime.py`)
- Frontend PSL page
- Database schemas

The service automatically detects the number of classes from the checkpoint at runtime:
```python
num_classes = checkpoint["num_classes"]  # 37 instead of 23
self.model = AlphabetClassifier(num_classes=num_classes)
```

## Testing

Run the test script to verify the upgrade:
```bash
cd emotisign-backend
python scripts/test_new_psl_model.py
```

Expected output:
```
✓ Model has 37 classes
✓ Model loaded successfully
✓ Inference successful (output shape: torch.Size([1, 37]))
✓ All tests passed!
```

## Performance Metrics (NEW PSL)

From `NEW PSL/web_model/metrics.json`:

- **Overall Accuracy:** 99.41%
- **Validation Samples:** 1,023
- **Training Samples:** 5,112
- **Per-class F1-scores:** 0.94–1.00 (most classes achieve perfect 1.00)

### Confusion Matrix Highlights
- Only 6 misclassifications out of 1,023 validation samples
- Most confusion between visually similar letters (ش/م, ص/ا, ض/ٹ, ڈ/م, ی/و)

## Backward Compatibility

The old 23-class model checkpoint is **replaced**, not preserved. If you need to roll back:

1. Restore the old `alphabet_classifier.pt` from backup
2. Restore the old `label_map.json` (23 entries)
3. Revert `psl_live_service.py` to remove CLAHE/smoothing (optional)

## Next Steps

### Recommended
1. Test the live PSL recognition page with real users
2. Collect feedback on the new letters (ع غ ف ق ل م ن و ک گ ہ ی ے)
3. Monitor session statistics for confidence distribution

### Optional Enhancements
1. Add rotation augmentation during training (±15°) for better real-world robustness
2. Collect a small real-webcam fine-tuning dataset (20–30 samples per class)
3. Implement sequence modeling for motion-sensitive letters (if needed)

## References

- NEW PSL model source: `NEW PSL/web_model/model.json`
- Conversion script: `scripts/convert_new_psl_model.py`
- Service implementation: `app/ml/psl_live_service.py`
- Test script: `scripts/test_new_psl_model.py`
