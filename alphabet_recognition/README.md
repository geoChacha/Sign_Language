# PSL Alphabet Recognition for Web

This folder contains a small training/export pipeline for realtime alphabet
recognition in a web app.

The recommended browser pipeline is:

1. Use MediaPipe Hands in the web app to get 21 hand landmarks.
2. Normalize those landmarks with the same logic used during training.
3. Run the exported JSON MLP model with `web/psl_alphabet_classifier.js`.

The model is intentionally tiny: it classifies a 42-value vector
`[x1, y1, ..., x21, y21]`, so it is fast enough to run every webcam frame.

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r alphabet_recognition\requirements.txt
```

## Check The Dataset

```powershell
python alphabet_recognition\inspect_dataset.py
```

The script reads `previous/main_dataset.db` and summarizes the
`rightHandDataset` alphabet table.

## Train And Export

```powershell
python alphabet_recognition\train_export.py
```

Outputs are written to:

- `alphabet_recognition/web_model/model.json`
- `alphabet_recognition/web_model/metrics.json`

`model.json` is the file your web app should load.

## Browser Use

Copy or serve these two files in your web app:

- `alphabet_recognition/web_model/model.json`
- `alphabet_recognition/web/psl_alphabet_classifier.js`
- `alphabet_recognition/web/prediction_smoother.js`

Example:

```html
<script type="module">
  import { PslAlphabetClassifier } from "./psl_alphabet_classifier.js";

  const classifier = await PslAlphabetClassifier.load("./model.json");

  // landmarks must be MediaPipe-style: [{x, y}, ...] with 21 points.
  const result = classifier.predict(landmarks);
  console.log(result.label, result.confidence, result.topK);
</script>
```

For a webcam app, smooth predictions across the last 5-10 frames and show
`Unknown` when confidence is below your chosen threshold.
