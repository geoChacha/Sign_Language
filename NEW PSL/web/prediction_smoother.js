export class PredictionSmoother {
  constructor(options = {}) {
    this.windowSize = options.windowSize ?? 8;
    this.confidenceThreshold = options.confidenceThreshold ?? 0.7;
    this.reset();
  }

  reset() {
    this.frames = [];
  }

  push(prediction) {
    if (!prediction || prediction.label === "Unknown") {
      this.frames.push({ label: "Unknown", confidence: 0 });
    } else {
      this.frames.push({
        label: prediction.rawLabel ?? prediction.label,
        confidence: prediction.confidence ?? 0,
      });
    }

    if (this.frames.length > this.windowSize) {
      this.frames.shift();
    }

    return this.current();
  }

  current() {
    if (this.frames.length === 0) {
      return { label: "Unknown", confidence: 0 };
    }

    const scores = new Map();
    for (const frame of this.frames) {
      if (frame.label === "Unknown") {
        continue;
      }
      scores.set(
        frame.label,
        (scores.get(frame.label) ?? 0) + frame.confidence / this.windowSize,
      );
    }

    let bestLabel = "Unknown";
    let bestScore = 0;
    for (const [label, score] of scores.entries()) {
      if (score > bestScore) {
        bestLabel = label;
        bestScore = score;
      }
    }

    return {
      label: bestScore >= this.confidenceThreshold ? bestLabel : "Unknown",
      confidence: bestScore,
      frames: this.frames.length,
    };
  }
}
