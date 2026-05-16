'use client';

import { useRef, useEffect } from 'react';

// MediaPipe hand connections — 21 landmark pairs
const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm
  [5, 9], [9, 13], [13, 17],
];

interface HandSkeletonProps {
  /** 21 landmarks as [[x_px, y_px], ...]. Pass null to clear the canvas. */
  landmarks: number[][] | null;
  width: number;
  height: number;
  className?: string;
}

export default function HandSkeleton({ landmarks, width, height, className }: HandSkeletonProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Always clear first
    ctx.clearRect(0, 0, width, height);

    if (!landmarks || landmarks.length < 21) return;

    // Draw connections
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';

    for (const [start, end] of HAND_CONNECTIONS) {
      const [x1, y1] = landmarks[start];
      const [x2, y2] = landmarks[end];
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // Draw landmark dots
    for (const [x, y] of landmarks) {
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fillStyle = '#22c55e';
      ctx.fill();
      // White border for visibility
      ctx.strokeStyle = 'rgba(255,255,255,0.6)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }, [landmarks, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className={className}
    />
  );
}
