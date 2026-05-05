'use client';

/**
 * SkeletonCanvas
 *
 * Renders a MediaPipe Holistic skeleton animation onto an HTML5 Canvas element.
 * Receives raw keypoint arrays (N_frames × 75 landmarks × [x, y]) from the
 * text-to-sign API and animates them frame-by-frame using requestAnimationFrame.
 *
 * Coordinate notes:
 *   - X is normalized [0, 1] by MediaPipe
 *   - Y can exceed 1.0 (full-body signs reach ~1.9) because MediaPipe normalizes
 *     by frame width, not height. We normalize per-sequence to fit the canvas.
 *
 * Landmark layout (75 total):
 *   Indices  0–32  → MediaPipe Pose (33 body landmarks)
 *   Indices 33–53  → MediaPipe Left Hand (21 landmarks)
 *   Indices 54–74  → MediaPipe Right Hand (21 landmarks)
 */

import { useEffect, useRef, useMemo } from 'react';

// ── Skeleton topology (mirrored from Featrure_sign_generation/test_two.py) ──

const POSE_CONNECTIONS: [number, number][] = [
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],
  [9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
  [11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[27,31],[23,26],[26,28],[28,32],[24,26],
];
const POSE_CYAN_COUNT = 12; // first 12 → cyan, rest → magenta

const HAND_CONNECTIONS: [number, number][] = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[5,9],[9,10],[10,11],[11,12],
  [0,13],[9,13],[13,14],[14,15],[15,16],
  [0,17],[13,17],[17,18],[18,19],[19,20],
];

// ── Colors ──────────────────────────────────────────────────────────────────
const BG       = '#f8fafc';
const CYAN     = '#0891b2';
const MAGENTA  = '#a21caf';
const RED      = '#dc2626';
const LIME     = '#16a34a';
const BODY_DOT = '#3b82f6';

// ── Types ────────────────────────────────────────────────────────────────────

export interface SkeletonCanvasProps {
  /** Keypoint frames: shape [N_frames][75][2] — X,Y (Y may exceed 1.0) */
  frames: number[][][];
  /** Word label (accessibility) */
  word: string;
  /** Milliseconds per frame (default 50) */
  frameRateMs?: number;
  /** Canvas width in pixels (default 480) */
  width?: number;
  /** Canvas height in pixels (default 480) */
  height?: number;
  className?: string;
}

// ── Coordinate normalization ─────────────────────────────────────────────────

interface Bounds { xMin: number; xMax: number; yMin: number; yMax: number; }

/**
 * Compute the bounding box of all valid (non-zero) landmarks across ALL frames
 * of a sign sequence. Used once per sign to get stable, consistent scaling.
 */
function computeBounds(frames: number[][][]): Bounds {
  let xMin = Infinity, xMax = -Infinity;
  let yMin = Infinity, yMax = -Infinity;

  for (const frame of frames) {
    for (const lm of frame) {
      if (lm[0] === 0 && lm[1] === 0) continue;
      if (lm[0] < xMin) xMin = lm[0];
      if (lm[0] > xMax) xMax = lm[0];
      if (lm[1] < yMin) yMin = lm[1];
      if (lm[1] > yMax) yMax = lm[1];
    }
  }

  // Fallback if no valid landmarks found
  if (!isFinite(xMin)) return { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };

  // Add 8% padding on each side so landmarks aren't clipped at the edge
  const xPad = (xMax - xMin) * 0.08 || 0.05;
  const yPad = (yMax - yMin) * 0.08 || 0.05;
  return {
    xMin: xMin - xPad,
    xMax: xMax + xPad,
    yMin: yMin - yPad,
    yMax: yMax + yPad,
  };
}

/**
 * Map a raw landmark coordinate to canvas pixel space using pre-computed bounds.
 * Maintains aspect ratio by using the same scale factor for X and Y.
 */
function toPixel(
  lm: number[],
  bounds: Bounds,
  canvasW: number,
  canvasH: number,
): [number, number] {
  const xRange = bounds.xMax - bounds.xMin || 1;
  const yRange = bounds.yMax - bounds.yMin || 1;

  // Use uniform scale to preserve aspect ratio
  const scale = Math.min(canvasW / xRange, canvasH / yRange);
  const drawW = xRange * scale;
  const drawH = yRange * scale;

  // Center the drawing in the canvas
  const offsetX = (canvasW - drawW) / 2;
  const offsetY = (canvasH - drawH) / 2;

  const px = (lm[0] - bounds.xMin) * scale + offsetX;
  const py = (lm[1] - bounds.yMin) * scale + offsetY;
  return [px, py];
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function isValid(lm: number[]): boolean {
  return Array.isArray(lm) && lm.length >= 2 && !(lm[0] === 0 && lm[1] === 0);
}

function drawLine(
  ctx: CanvasRenderingContext2D,
  a: number[], b: number[],
  color: string, lw: number,
  bounds: Bounds, w: number, h: number,
): void {
  if (!isValid(a) || !isValid(b)) return;
  const [ax, ay] = toPixel(a, bounds, w, h);
  const [bx, by] = toPixel(b, bounds, w, h);
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx, by);
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.stroke();
}

function drawDot(
  ctx: CanvasRenderingContext2D,
  lm: number[], color: string, r: number,
  bounds: Bounds, w: number, h: number,
): void {
  if (!isValid(lm)) return;
  const [px, py] = toPixel(lm, bounds, w, h);
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

// ── Core draw function (exported for unit tests) ─────────────────────────────

export function drawFrame(
  ctx: CanvasRenderingContext2D,
  frame: number[][],
  width: number,
  height: number,
  bounds: Bounds,
): void {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, width, height);

  if (!frame || frame.length < 75) return;

  const body      = frame.slice(0, 33);
  const leftHand  = frame.slice(33, 54);
  const rightHand = frame.slice(54, 75);

  // Body skeleton
  POSE_CONNECTIONS.forEach(([i, j], idx) => {
    const color = idx < POSE_CYAN_COUNT ? CYAN : MAGENTA;
    if (i < body.length && j < body.length)
      drawLine(ctx, body[i], body[j], color, 3, bounds, width, height);
  });
  body.forEach(lm => drawDot(ctx, lm, BODY_DOT, 3, bounds, width, height));

  // Left hand
  HAND_CONNECTIONS.forEach(([i, j]) => {
    if (i < leftHand.length && j < leftHand.length)
      drawLine(ctx, leftHand[i], leftHand[j], RED, 2.5, bounds, width, height);
  });
  leftHand.forEach(lm => drawDot(ctx, lm, RED, 3, bounds, width, height));

  // Right hand
  HAND_CONNECTIONS.forEach(([i, j]) => {
    if (i < rightHand.length && j < rightHand.length)
      drawLine(ctx, rightHand[i], rightHand[j], LIME, 2.5, bounds, width, height);
  });
  rightHand.forEach(lm => drawDot(ctx, lm, LIME, 3, bounds, width, height));
}

// ── Component ────────────────────────────────────────────────────────────────

export default function SkeletonCanvas({
  frames,
  word,
  frameRateMs = 50,
  width = 480,
  height = 480,
  className,
}: SkeletonCanvasProps) {
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const frameIdxRef = useRef(0);
  const lastTsRef   = useRef(0);
  const rafRef      = useRef(0);

  // Compute bounds once per sign (stable across all frames of this word)
  const bounds = useMemo(() => computeBounds(frames), [frames]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.warn('SkeletonCanvas: cannot get 2D context');
      return;
    }

    // Draw background immediately
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, width, height);

    if (!frames || frames.length === 0) return;

    // Reset animation state
    frameIdxRef.current = 0;
    lastTsRef.current   = 0;
    cancelAnimationFrame(rafRef.current);

    // Draw first frame immediately — no waiting for first RAF tick
    console.log('[SkeletonCanvas] mounting word=', word, 'frames=', frames.length, 'bounds=', bounds);
    drawFrame(ctx, frames[0], width, height, bounds);

    const tick = (ts: number) => {
      if (ts - lastTsRef.current >= frameRateMs) {
        const idx = frameIdxRef.current % frames.length;
        drawFrame(ctx, frames[idx], width, height, bounds);
        frameIdxRef.current += 1;
        lastTsRef.current = ts;
      }
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frames, frameRateMs, width, height, bounds]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className={className}
      style={{ display: 'block' }}
      aria-label={`ASL sign animation: ${word}`}
      role="img"
    />
  );
}
