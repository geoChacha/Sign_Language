'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import { FiCamera, FiStopCircle, FiRefreshCw } from 'react-icons/fi';
import Button from '@/components/ui/Button';
import EmotionBadge from '@/components/ui/EmotionBadge';
import HandSkeleton from '@/components/HandSkeleton';
import type { Emotion } from '@/types';
import {
  usePSLWebSocket,
  type PSLPredictionResult,
  type PSLSessionStats,
} from '@/hooks/usePSLWebSocket';

// ── Helpers ───────────────────────────────────────────────────────────────────

type ConfidenceLevel = 'high' | 'medium' | 'low';

function getConfidenceLevel(confidence: number): ConfidenceLevel {
  if (confidence >= 0.85) return 'high';
  if (confidence >= 0.70) return 'medium';
  return 'low';
}

const CONFIDENCE_COLORS: Record<ConfidenceLevel, string> = {
  high: '#22c55e',    // green
  medium: '#eab308',  // yellow
  low: '#f97316',     // orange
};

const CONFIDENCE_LABELS: Record<ConfidenceLevel, string> = {
  high: 'High confidence',
  medium: 'Medium confidence',
  low: 'Low confidence',
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function PSLAlphabetPage() {
  // ── Refs ──────────────────────────────────────────────────────────────────
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const fpsFrameCountRef = useRef(0);
  const fpsLastTimeRef = useRef(Date.now());

  // ── State ─────────────────────────────────────────────────────────────────
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [prediction, setPrediction] = useState<PSLPredictionResult | null>(null);
  const [noHandDetected, setNoHandDetected] = useState(false);
  const [landmarks, setLandmarks] = useState<number[][] | null>(null);
  const [confidenceLevel, setConfidenceLevel] = useState<ConfidenceLevel | null>(null);
  const [fps, setFps] = useState<number>(0);
  const [sessionStats, setSessionStats] = useState<PSLSessionStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [videoSize, setVideoSize] = useState({ width: 640, height: 480 });

  // ── WebSocket hook ────────────────────────────────────────────────────────
  const { connectionStatus, connect, disconnect, sendFrame } = usePSLWebSocket({
    onConnected: () => {
      setError(null);
    },
    onResult: (result) => {
      setPrediction(result);
      setLandmarks(result.landmarks);
      setNoHandDetected(false);
      setConfidenceLevel(getConfidenceLevel(result.confidence));
    },
    onNoHand: () => {
      setNoHandDetected(true);
      setLandmarks(null);
    },
    onLowConfidence: (data) => {
      setNoHandDetected(false);
      setLandmarks(null);
      setPrediction({
        predicted_label: data.predicted_label,
        urdu_text: data.predicted_label,
        confidence: data.confidence,
        landmarks: [],
        emotion: data.emotion as PSLPredictionResult['emotion'],
        emotion_emoji: data.emotion_emoji,
      });
      setConfidenceLevel('low');
    },
    onError: (msg) => {
      toast.error(msg);
    },
    onSessionEnd: (stats) => {
      setSessionStats(stats);
    },
  });

  // ── Webcam ────────────────────────────────────────────────────────────────
  const startWebcam = useCallback(async (): Promise<boolean> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            setVideoSize({
              width: videoRef.current.videoWidth || 640,
              height: videoRef.current.videoHeight || 480,
            });
          }
        };
      }
      return true;
    } catch (err: unknown) {
      const name = (err as { name?: string }).name;
      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        setError('Camera permission denied. Please allow camera access in your browser settings.');
      } else if (name === 'NotReadableError') {
        setError('Camera is already in use by another application.');
      } else {
        setError('Failed to access webcam. Please check your camera.');
      }
      return false;
    }
  }, []);

  const stopWebcam = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  // ── Frame capture loop ────────────────────────────────────────────────────
  const startFrameLoop = useCallback(() => {
    fpsFrameCountRef.current = 0;
    fpsLastTimeRef.current = Date.now();

    frameIntervalRef.current = setInterval(() => {
      const video = videoRef.current;
      if (!video || video.readyState < 2) return;

      // Capture frame to offscreen canvas
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(video, 0, 0);

      canvas.toBlob((blob) => {
        if (!blob) return;
        const reader = new FileReader();
        reader.onloadend = () => {
          const dataUrl = reader.result as string;
          // Strip the data:image/jpeg;base64, prefix
          const base64 = dataUrl.split(',')[1];
          if (base64) sendFrame(base64);
        };
        reader.readAsDataURL(blob);
      }, 'image/jpeg', 0.8);

      // FPS tracking
      fpsFrameCountRef.current += 1;
      const now = Date.now();
      const elapsed = (now - fpsLastTimeRef.current) / 1000;
      if (elapsed >= 1.0) {
        setFps(Math.round(fpsFrameCountRef.current / elapsed));
        fpsFrameCountRef.current = 0;
        fpsLastTimeRef.current = now;
      }
    }, 33); // ~30 FPS
  }, [sendFrame]);

  const stopFrameLoop = useCallback(() => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    setFps(0);
  }, []);

  // ── Start / Stop recognition ──────────────────────────────────────────────
  const startRecognition = useCallback(async () => {
    setError(null);
    setSessionStats(null);
    setPrediction(null);
    setLandmarks(null);
    setNoHandDetected(false);

    const ok = await startWebcam();
    if (!ok) return;

    connect();
    setIsRecognizing(true);
    // Start frame loop after a short delay to let WS connect
    setTimeout(() => startFrameLoop(), 500);
  }, [startWebcam, connect, startFrameLoop]);

  const stopRecognition = useCallback(() => {
    stopFrameLoop();
    disconnect();
    stopWebcam();
    setIsRecognizing(false);
    setLandmarks(null);
    setNoHandDetected(false);
  }, [stopFrameLoop, disconnect, stopWebcam]);

  // ── Cleanup on unmount ────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      stopFrameLoop();
      stopWebcam();
    };
  }, [stopFrameLoop, stopWebcam]);

  // ── Connection status badge ───────────────────────────────────────────────
  const statusBadge = {
    idle: { label: 'Idle', color: 'bg-gray-100 text-gray-600' },
    connecting: { label: 'Connecting…', color: 'bg-yellow-100 text-yellow-700' },
    connected: { label: 'Live', color: 'bg-green-100 text-green-700' },
    disconnected: { label: 'Disconnected', color: 'bg-red-100 text-red-700' },
    error: { label: 'Error', color: 'bg-red-100 text-red-700' },
  }[connectionStatus];

  const predColor = confidenceLevel ? CONFIDENCE_COLORS[confidenceLevel] : '#9ca3af';

  return (
    <div className="max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        {/* ── Header ── */}
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center gap-3">
            <h1 className="text-2xl font-bold text-white">PSL Alphabet Recognition</h1>
            <span className="text-2xl">🇵🇰</span>
          </div>
          <p className="text-white/70 text-sm max-w-md mx-auto">
            Show a Pakistan Sign Language alphabet hand pose to your webcam for real-time recognition.
          </p>
        </div>

        {/* ── Webcam viewport ── */}
        <div className="relative mx-auto" style={{ maxWidth: 640 }}>
          <div className="relative bg-gray-900 rounded-xl overflow-hidden" style={{ aspectRatio: '4/3' }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
              style={{ transform: 'scaleX(-1)' }} // mirror for natural feel
            />

            {/* Hand skeleton overlay */}
            {isRecognizing && (
              <HandSkeleton
                landmarks={landmarks}
                width={videoSize.width}
                height={videoSize.height}
                className="absolute inset-0 w-full h-full pointer-events-none"
              />
            )}

            {/* FPS badge */}
            {isRecognizing && fps > 0 && (
              <div className="absolute top-3 right-3 bg-black/60 text-white text-xs px-2 py-1 rounded-md font-mono">
                {fps} FPS
              </div>
            )}

            {/* Connection status badge */}
            <div className={`absolute top-3 left-3 text-xs px-2 py-1 rounded-md font-medium ${statusBadge.color}`}>
              {statusBadge.label}
            </div>

            {/* No hand overlay */}
            <AnimatePresence>
              {isRecognizing && noHandDetected && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/70 text-white/80 text-sm px-4 py-2 rounded-full"
                >
                  No hand detected
                </motion.div>
              )}
            </AnimatePresence>

            {/* Idle placeholder */}
            {!isRecognizing && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-gray-900/80">
                <FiCamera size={56} className="text-gray-500" />
                <p className="text-gray-400 text-sm">Click &quot;Start Recognition&quot; to begin</p>
              </div>
            )}
          </div>
        </div>

        {/* ── Prediction display ── */}
        <div className="bg-white rounded-xl p-6 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900">Prediction</h3>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800">
              🇵🇰 PSL Alphabet
            </span>
          </div>

          <div className="min-h-[120px] flex flex-col items-center justify-center gap-4">
            {prediction ? (
              <>
                {/* Large Urdu label + facial emotion */}
                <motion.div className="flex flex-col items-center gap-3">
                  <p
                    dir="rtl"
                    className="font-bold text-center leading-none"
                    style={{
                      fontFamily: 'serif',
                      letterSpacing: '0.05em',
                      fontSize: '4rem',
                      color: predColor,
                      transition: 'color 0.3s ease',
                    }}
                  >
                    {prediction.urdu_text || prediction.predicted_label}
                  </p>
                  {prediction.emotion && (
                    <EmotionBadge emotion={prediction.emotion as Emotion} size="md" />
                  )}
                </motion.div>

                {/* Confidence bar */}
                <div className="w-full max-w-xs space-y-1">
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>{confidenceLevel ? CONFIDENCE_LABELS[confidenceLevel] : ''}</span>
                    <span className="font-medium" style={{ color: predColor }}>
                      {Math.round(prediction.confidence * 100)}%
                    </span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{
                        width: `${prediction.confidence * 100}%`,
                        backgroundColor: predColor,
                      }}
                    />
                  </div>
                </div>
              </>
            ) : (
              <p className="text-gray-400 text-sm">
                {isRecognizing ? 'Waiting for hand pose…' : '---'}
              </p>
            )}
          </div>
        </div>

        {/* ── Controls ── */}
        <div className="flex justify-center gap-4">
          {!isRecognizing ? (
            <Button
              variant="accent"
              onClick={startRecognition}
              leftIcon={<FiCamera />}
              className="min-w-[200px]"
            >
              Start Recognition
            </Button>
          ) : (
            <Button
              variant="danger"
              onClick={stopRecognition}
              leftIcon={<FiStopCircle />}
              className="min-w-[200px]"
            >
              Stop Recognition
            </Button>
          )}

          {!isRecognizing && (prediction || sessionStats) && (
            <Button
              variant="secondary"
              onClick={() => {
                setPrediction(null);
                setSessionStats(null);
                setConfidenceLevel(null);
              }}
              leftIcon={<FiRefreshCw />}
            >
              Reset
            </Button>
          )}
        </div>

        {/* ── Error display ── */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3"
          >
            <span className="text-red-500 text-lg">⚠️</span>
            <div className="flex-1">
              <p className="text-red-700 text-sm">{error}</p>
            </div>
            <button
              onClick={() => { setError(null); startRecognition(); }}
              className="text-xs text-red-600 underline hover:no-underline"
            >
              Retry
            </button>
          </motion.div>
        )}

        {/* ── Session statistics ── */}
        <AnimatePresence>
          {sessionStats && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="bg-white rounded-xl p-6 shadow-lg"
            >
              <h3 className="font-semibold text-gray-900 mb-4">Session Summary</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {[
                  { label: 'Total Frames', value: sessionStats.total_frames },
                  { label: 'Frames with Hand', value: sessionStats.frames_with_hands },
                  { label: 'Predictions', value: sessionStats.predictions_made },
                  {
                    label: 'Avg Confidence',
                    value: `${Math.round(sessionStats.average_confidence * 100)}%`,
                  },
                ].map(({ label, value }) => (
                  <div key={label} className="text-center p-3 bg-gray-50 rounded-lg">
                    <p className="text-2xl font-bold text-primary-600">{value}</p>
                    <p className="text-xs text-gray-500 mt-1">{label}</p>
                  </div>
                ))}
              </div>
              {sessionStats.session_duration_seconds > 0 && (
                <p className="text-xs text-gray-400 text-center mt-3">
                  Session duration: {sessionStats.session_duration_seconds.toFixed(1)}s
                </p>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}
