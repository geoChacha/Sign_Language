function relu(values) {
  return values.map((value) => Math.max(0, value));
}

function softmax(values) {
  const maxValue = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - maxValue));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function dense(input, weights, bias) {
  const output = new Array(bias.length).fill(0);
  for (let j = 0; j < bias.length; j += 1) {
    let sum = bias[j];
    for (let i = 0; i < input.length; i += 1) {
      sum += input[i] * weights[i][j];
    }
    output[j] = sum;
  }
  return output;
}

function normalizeLandmarks(landmarks) {
  if (!Array.isArray(landmarks) || landmarks.length !== 21) {
    throw new Error("Expected 21 MediaPipe hand landmarks");
  }

  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const scale = Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y);
  const safeScale = scale > 1e-6 ? scale : 1;
  const features = [];

  for (const point of landmarks) {
    features.push((point.x - wrist.x) / safeScale);
    features.push((point.y - wrist.y) / safeScale);
  }

  return features;
}

function topK(probs, labels, k) {
  return probs
    .map((confidence, index) => ({ label: labels[index], confidence }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, k);
}

export class PslAlphabetClassifier {
  static async load(modelUrl) {
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`Failed to load PSL alphabet model: ${response.status}`);
    }
    const model = await response.json();
    return new PslAlphabetClassifier(model);
  }

  constructor(model) {
    if (model.format !== "psl-alphabet-mlp-json-v1") {
      throw new Error(`Unsupported model format: ${model.format}`);
    }
    this.model = model;
  }

  preprocess(landmarks) {
    const features = normalizeLandmarks(landmarks);
    return features.map((value, index) => {
      const std = this.model.featureStd[index] || 1;
      return (value - this.model.featureMean[index]) / std;
    });
  }

  predict(landmarks, options = {}) {
    const threshold =
      options.confidenceThreshold ?? this.model.confidenceThreshold ?? 0.7;
    const k = options.topK ?? 5;

    let x = this.preprocess(landmarks);
    for (let layer = 0; layer < this.model.weights.length; layer += 1) {
      x = dense(x, this.model.weights[layer], this.model.biases[layer]);
      if (layer < this.model.weights.length - 1) {
        x = relu(x);
      }
    }

    const probs = softmax(x);
    const ranked = topK(probs, this.model.labels, k);
    const best = ranked[0];

    return {
      label: best.confidence >= threshold ? best.label : "Unknown",
      rawLabel: best.label,
      confidence: best.confidence,
      topK: ranked,
    };
  }
}

export { normalizeLandmarks };
