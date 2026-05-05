'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import {
  FiCamera,
  FiVolume2,
  FiCheckCircle,
  FiXCircle,
  FiAlertCircle,
  FiVideo,
  FiStopCircle,
  FiRefreshCw,
  FiSend,
} from 'react-icons/fi';
import Button from '@/components/ui/Button';
import Select from '@/components/ui/Select';
import api from '@/lib/api';
import { SignLanguage, ValidationState, PredictionResult } from '@/types';

type PageState = 'idle' | 'validating' | 'countdown' | 'recording' | 'processing' | 'review' | 'done';
type InputMode = 'record' | 'upload';

// How many validation checks must pass to consider "ready"
const REQUIRED_CHECKS = 3; // background + body + distance

export default function SignToTextPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const validationIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const countdownIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const [state, setState] = useState<PageState>('idle');
  const [inputMode, setInputMode] = useState<InputMode>('record');
  const [signLanguage, setSignLanguage] = useState<SignLanguage>('ASL');
  const [validationState, setValidationState] = useState<ValidationState | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [countdown, setCountdown] = useState(3);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [thumbnailUrl, setThumbnailUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // How many of the 3 checks are passing right now
  const passedChecks = validationState
    ? [
        validationState.background?.status === 'clear' || validationState.background?.status === 'acceptable',
        validationState.body?.status === 'visible',
        validationState.distance?.status === 'optimal' || validationState.distance?.status === 'acceptable',
      ].filter(Boolean).length
    : 0;

  const allReady = passedChecks === REQUIRED_CHECKS;

  // Border indicator: 0–1 fill based on passed checks
  const indicatorProgress = passedChecks / REQUIRED_CHECKS;

  // Indicator color: red → yellow → green
  const indicatorColor =
    passedChecks === 0 ? '#ef4444'
    : passedChecks === 1 ? '#f97316'
    : passedChecks === 2 ? '#eab308'
    : '#22c55e';

  // ── File upload ──────────────────────────────────────────────────────────

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const validTypes = ['video/mp4', 'video/webm', 'video/avi', 'video/mov'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|webm|avi|mov)$/i)) {
      toast.error('Please select a valid video file (MP4, WEBM, AVI, MOV)');
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      toast.error('Video file is too large (max 50MB)');
      return;
    }
    setSelectedFile(file);
    toast.success('Video selected! Click "Submit Video" to process.');
  }, []);

  const uploadSelectedFile = useCallback(async () => {
    if (!selectedFile) return;
    setState('processing');
    setError(null);
    // Generate thumbnail from uploaded video file
    try {
      const url = URL.createObjectURL(selectedFile);
      const vid = document.createElement('video');
      vid.src = url;
      vid.currentTime = 0.5;
      vid.onloadeddata = () => {
        const canvas = document.createElement('canvas');
        canvas.width = vid.videoWidth || 640;
        canvas.height = vid.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        if (ctx) { ctx.drawImage(vid, 0, 0); setThumbnailUrl(canvas.toDataURL('image/jpeg', 0.8)); }
        URL.revokeObjectURL(url);
      };
    } catch { /* non-fatal */ }
    try {
      const formData = new FormData();
      formData.append('video', selectedFile);
      // PSL endpoint does not use TTA — only append for ASL
      if (signLanguage !== 'PSL') formData.append('use_tta', 'true');
      const endpoint = signLanguage === 'PSL' ? 'psl-sign-to-text' : 'sign-to-text';
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/ml/${endpoint}`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process video');
      }
      const data: PredictionResult = await response.json();
      setResult(data);
      setState('done');
      toast.success('Translation complete!');
    } catch (err: any) {
      setError(err.message || 'Failed to process video');
      toast.error(err.message || 'Failed to process video');
      setState('idle');
    }
  }, [selectedFile, signLanguage]);

  // ── Webcam validation ────────────────────────────────────────────────────

  const startValidation = useCallback(async () => {
    try {
      setError(null);
      setState('validating');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;

      validationIntervalRef.current = setInterval(() => {
        sendFrameForValidation();
      }, 200);
    } catch {
      toast.error('Failed to access webcam. Please grant camera permissions.');
      setState('idle');
    }
  }, []);

  const sendFrameForValidation = useCallback(async () => {
    if (!videoRef.current) return;
    try {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(videoRef.current, 0, 0);
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((b) => resolve(b!), 'image/jpeg', 0.8);
      });
      const formData = new FormData();
      formData.append('frame', blob, 'frame.jpg');
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/ml/validate-environment`, {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        const data: ValidationState = await response.json();
        setValidationState(data);
      }
    } catch {
      // silent — validation errors are non-fatal
    }
  }, []);

  // Watch validationState changes to trigger countdown when all checks pass
  useEffect(() => {
    if (state !== 'validating' && state !== 'countdown') return;

    if (allReady && state === 'validating') {
      // All checks passed — start 3s countdown
      setState('countdown');
      setCountdown(3);
      let c = 3;
      countdownIntervalRef.current = setInterval(() => {
        c -= 1;
        setCountdown(c);
        if (c <= 0) {
          clearInterval(countdownIntervalRef.current!);
          countdownIntervalRef.current = null;
          startRecording();
        }
      }, 1000);
    } else if (!allReady && state === 'countdown') {
      // Requirements dropped — cancel countdown, go back to validating
      if (countdownIntervalRef.current) {
        clearInterval(countdownIntervalRef.current);
        countdownIntervalRef.current = null;
      }
      setState('validating');
    }
  }, [allReady, state]);

  const stopValidation = useCallback(() => {
    if (validationIntervalRef.current) clearInterval(validationIntervalRef.current);
    if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
    validationIntervalRef.current = null;
    countdownIntervalRef.current = null;
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setState('idle');
    setValidationState(null);
  }, []);

  // ── Recording ────────────────────────────────────────────────────────────

  const startRecording = useCallback(() => {
    if (!streamRef.current) return;
    // Stop validation polling while recording
    if (validationIntervalRef.current) {
      clearInterval(validationIntervalRef.current);
      validationIntervalRef.current = null;
    }
    try {
      recordedChunksRef.current = [];
      setRecordingTime(0);
      setThumbnailUrl(null); // clear old thumbnail — new one captured on stop
      const mediaRecorder = new MediaRecorder(streamRef.current, {
        mimeType: 'video/webm;codecs=vp9',
      });
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunksRef.current.push(e.data);
      };
      mediaRecorder.onstop = () => {
        // Go to review state instead of auto-uploading
        setState('review');
      };
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setState('recording');

      const startTime = Date.now();
      const timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        setRecordingTime(elapsed);
        if (elapsed >= 30) {
          stopRecording();
          clearInterval(timerInterval);
        }
      }, 1000);
    } catch {
      toast.error('Failed to start recording');
      setState('validating');
    }
  }, []);

  const stopRecording = useCallback(() => {
    // Capture thumbnail from current video frame before stopping
    if (videoRef.current && videoRef.current.videoWidth > 0) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0);
        setThumbnailUrl(canvas.toDataURL('image/jpeg', 0.8));
      }
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  }, []);

  // ── Review actions ───────────────────────────────────────────────────────

  const submitRecording = useCallback(async () => {
    setState('processing');
    try {
      const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
      if (blob.size > 50 * 1024 * 1024) throw new Error('Video file is too large (max 50MB)');
      const formData = new FormData();
      formData.append('video', blob, 'recording.webm');
      // PSL endpoint does not use TTA — only append for ASL
      if (signLanguage !== 'PSL') formData.append('use_tta', 'true');
      const endpoint = signLanguage === 'PSL' ? 'psl-sign-to-text' : 'sign-to-text';
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/ml/${endpoint}`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process video');
      }
      const data: PredictionResult = await response.json();
      setResult(data);
      setState('done');
      toast.success('Translation complete!');
    } catch (err: any) {
      setError(err.message || 'Failed to process video');
      toast.error(err.message || 'Failed to process video');
      setState('review');
    }
  }, [signLanguage]);

  const reRecord = useCallback(() => {
    recordedChunksRef.current = [];
    setRecordingTime(0);
    setError(null);
    // Restart validation
    setState('validating');
    setValidationState(null);
    validationIntervalRef.current = setInterval(() => {
      sendFrameForValidation();
    }, 200);
  }, [sendFrameForValidation]);

  const recordAgain = useCallback(() => {
    setResult(null);
    setError(null);
    setRecordingTime(0);
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    setState('idle');
  }, []);

  // ── TTS ──────────────────────────────────────────────────────────────────

  const speakText = async () => {
    if (!result?.recognized_text) return;
    setIsSpeaking(true);
    try {
      const response = await api.textToSpeech(result.recognized_text, 'en', 'default');
      const audio = new Audio(api.getAudioUrl(response.audio_url.split('/').pop()!));
      audio.onended = () => setIsSpeaking(false);
      audio.onerror = () => { setIsSpeaking(false); toast.error('Failed to play audio'); };
      audio.play();
    } catch {
      setIsSpeaking(false);
      toast.error('Failed to generate speech');
    }
  };

  // ── Cleanup ──────────────────────────────────────────────────────────────

  useEffect(() => {
    return () => {
      if (validationIntervalRef.current) clearInterval(validationIntervalRef.current);
      if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    };
  }, []);

  // ── Derived UI helpers ───────────────────────────────────────────────────

  const showIndicator = state === 'validating' || state === 'countdown' || state === 'recording';
  const isActive = state === 'validating' || state === 'countdown' || state === 'recording' || state === 'processing';

  return (
    <div className="max-w-4xl mx-auto">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">

        {/* ── Camera viewport with indicator border ── */}
        <div className="relative mx-auto" style={{ maxWidth: 640 }}>

          {/* Animated border indicator */}
          {showIndicator && (
            <div
              className="absolute inset-0 rounded-xl pointer-events-none z-10"
              style={{
                padding: 3,
                borderRadius: '0.75rem',
                background: `conic-gradient(${indicatorColor} ${indicatorProgress * 360}deg, rgba(255,255,255,0.15) ${indicatorProgress * 360}deg)`,
                WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
                WebkitMaskComposite: 'xor',
                maskComposite: 'exclude',
                transition: 'background 0.3s ease',
              }}
            />
          )}

          {/* Recording pulse border */}
          {state === 'recording' && (
            <div
              className="absolute inset-0 rounded-xl pointer-events-none z-10"
              style={{
                border: '3px solid #ef4444',
                borderRadius: '0.75rem',
                animation: 'recordingPulse 1.5s ease-in-out infinite',
              }}
            />
          )}

          <div className="camera-container bg-gray-200 rounded-xl overflow-hidden"
            style={thumbnailUrl && (state === 'idle' || state === 'done' || state === 'review') ? {
              backgroundImage: `url(${thumbnailUrl})`,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
            } : undefined}
          >
            <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />

            {/* Body position guide — shown during validation */}
            {(state === 'validating' || state === 'countdown') && (
              <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                <svg
                  viewBox="0 0 200 280"
                  className="h-4/5 opacity-30"
                  fill="none"
                  stroke={
                    validationState?.body?.status === 'visible' ? '#22c55e'
                    : validationState?.body?.status === 'partial' ? '#eab308'
                    : '#ffffff'
                  }
                  strokeWidth="2"
                  strokeDasharray={validationState?.body?.status === 'visible' ? '0' : '6 4'}
                  style={{ transition: 'stroke 0.4s ease' }}
                >
                  {/* Head */}
                  <circle cx="100" cy="30" r="22" />
                  {/* Neck */}
                  <line x1="100" y1="52" x2="100" y2="70" />
                  {/* Shoulders */}
                  <line x1="40" y1="80" x2="160" y2="80" />
                  {/* Torso */}
                  <line x1="40" y1="80" x2="50" y2="160" />
                  <line x1="160" y1="80" x2="150" y2="160" />
                  <line x1="50" y1="160" x2="150" y2="160" />
                  {/* Left arm */}
                  <line x1="40" y1="80" x2="10" y2="150" />
                  <line x1="10" y1="150" x2="5" y2="210" />
                  {/* Right arm */}
                  <line x1="160" y1="80" x2="190" y2="150" />
                  <line x1="190" y1="150" x2="195" y2="210" />
                  {/* Left hand */}
                  <ellipse cx="5" cy="220" rx="10" ry="14" />
                  {/* Right hand */}
                  <ellipse cx="195" cy="220" rx="10" ry="14" />
                </svg>
              </div>
            )}

            {/* Recording indicator badge */}
            {state === 'recording' && (
              <div className="recording-indicator">
                <div className="recording-dot" />
                <span>Recording {recordingTime}s</span>
              </div>
            )}

            {/* Countdown overlay */}
            {state === 'countdown' && (
              <motion.div
                className="absolute inset-0 flex flex-col items-center justify-center"
                style={{ background: 'rgba(0,0,0,0.45)' }}
              >
                <p className="text-white/80 text-sm uppercase tracking-widest mb-2">Starting in</p>
                <motion.span
                  key={countdown}
                  initial={{ scale: 1.6, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.6, opacity: 0 }}
                  className="text-white font-bold"
                  style={{ fontSize: 96, lineHeight: 1 }}
                >
                  {countdown}
                </motion.span>
              </motion.div>
            )}

            {/* Camera placeholder */}
            {state === 'idle' && (
              <div className="camera-placeholder flex-col gap-2">
                <FiCamera size={64} className="text-gray-400" />
                <p className="mt-2 text-gray-500 text-sm">Click "Start" to begin</p>
              </div>
            )}

            {/* Review overlay */}
            {state === 'review' && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4"
                style={{ background: 'rgba(0,0,0,0.7)' }}>
                <FiCheckCircle size={48} className="text-green-400" />
                <p className="text-white font-semibold text-lg">Recording complete</p>
                <p className="text-white/70 text-sm">{recordingTime}s recorded</p>
                <div className="flex gap-3 mt-2">
                  <button
                    onClick={reRecord}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-white/20 hover:bg-white/30 text-white text-sm font-medium transition-colors"
                  >
                    <FiRefreshCw size={16} /> Re-record
                  </button>
                  <button
                    onClick={submitRecording}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-green-500 hover:bg-green-600 text-white text-sm font-medium transition-colors"
                  >
                    <FiSend size={16} /> Submit
                  </button>
                </div>
              </div>
            )}

            {/* Processing overlay */}
            <AnimatePresence>
              {state === 'processing' && (
                <motion.div
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="absolute inset-0 rounded-xl flex flex-col items-center justify-center overflow-hidden"
                >
                  {/* Thumbnail background */}
                  {thumbnailUrl && (
                    <img
                      src={thumbnailUrl}
                      alt=""
                      className="absolute inset-0 w-full h-full object-cover"
                      style={{ filter: 'blur(6px) brightness(0.6)', transform: 'scale(1.08)' }}
                    />
                  )}
                  {/* Frosted glass overlay */}
                  <div className="absolute inset-0" style={{ background: 'rgba(255,255,255,0.35)', backdropFilter: 'blur(2px)' }} />
                  {/* Content */}
                  <div className="relative z-10 flex flex-col items-center gap-3 px-6 py-8 rounded-2xl"
                    style={{ background: 'rgba(255,255,255,0.55)', backdropFilter: 'blur(12px)',
                             boxShadow: '0 4px 32px rgba(0,0,0,0.12)', border: '1px solid rgba(255,255,255,0.7)' }}>
                    <h3 className="text-primary-700 font-semibold text-lg">Processing Video</h3>
                    <p className="text-gray-600 text-sm">Translating your signs…</p>
                    <div className="loading-dots text-primary-600"><span /><span /><span /></div>
                    <p className="text-gray-500 text-xs mt-1">This may take a few seconds</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Indicator legend */}
          {showIndicator && (
            <div className="flex items-center justify-center gap-2 mt-2">
              <div className="w-2 h-2 rounded-full" style={{ background: indicatorColor }} />
              <span className="text-xs text-gray-500">
                {passedChecks}/{REQUIRED_CHECKS} requirements met
                {state === 'countdown' ? ` — starting in ${countdown}s` : allReady ? ' — all clear!' : ''}
              </span>
            </div>
          )}
        </div>

        {/* ── Validation feedback ── */}
        {(state === 'validating' || state === 'countdown' || state === 'recording') && validationState && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-lg p-4 shadow-md">
            <h4 className="font-semibold text-gray-900 mb-3">Environment Check</h4>
            <div className="space-y-2">
              <ValidationItem
                label="Background"
                status={validationState.background?.status || 'cluttered'}
                message={validationState.background?.message || ''}
                isGood={validationState.background?.status === 'clear' || validationState.background?.status === 'acceptable'}
              />
              <ValidationItem
                label="Body Visibility"
                status={validationState.body?.status || 'not_visible'}
                message={validationState.body?.message || ''}
                isGood={validationState.body?.status === 'visible'}
                detail={
                  validationState.body?.landmarks_detected
                    ? Object.entries(validationState.body.landmarks_detected).map(([k, v]) => ({
                        label: k.replace('_', ' '),
                        ok: v as boolean,
                      }))
                    : undefined
                }
              />
              <ValidationItem
                label="Distance"
                status={validationState.distance?.status || 'out_of_range'}
                message={validationState.distance?.message || ''}
                isGood={validationState.distance?.status === 'optimal' || validationState.distance?.status === 'acceptable'}
              />
            </div>
          </motion.div>
        )}

        {/* ── Controls ── */}
        <div className="space-y-4">
          <div className="flex justify-center">
            <Select
              value={signLanguage}
              onChange={(e) => setSignLanguage(e.target.value as SignLanguage)}
              options={[
                { value: 'ASL', label: 'ASL - American Sign Language' },
                { value: 'PSL', label: 'PSL - Pakistan Sign Language' },
              ]}
              className="w-64"
              disabled={isActive}
            />
          </div>

          {/* PSL mode badge */}
          {signLanguage === 'PSL' && (
            <div className="flex justify-center">
              <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium bg-green-50 text-green-800 border border-green-200">
                🇵🇰 PSL Mode — Recognises 12 Urdu signs
              </span>
            </div>
          )}

          {state === 'idle' && (
            <div className="flex justify-center gap-4">
              <Button variant={inputMode === 'record' ? 'primary' : 'secondary'}
                onClick={() => setInputMode('record')} leftIcon={<FiVideo />}>
                Record Video
              </Button>
              <Button variant={inputMode === 'upload' ? 'primary' : 'secondary'}
                onClick={() => setInputMode('upload')} leftIcon={<FiCamera />}>
                Upload Video
              </Button>
            </div>
          )}

          <div className="flex justify-center gap-4 flex-wrap items-center">
            {state === 'idle' && inputMode === 'record' && (
              <Button variant="accent" onClick={startValidation} leftIcon={<FiCamera />} className="min-w-[180px]">
                Start
              </Button>
            )}

            {state === 'idle' && inputMode === 'upload' && (
              <>
                <input ref={fileInputRef} type="file" accept="video/mp4,video/webm,video/avi,video/mov"
                  onChange={handleFileSelect} className="hidden" />
                <Button variant="secondary" onClick={() => fileInputRef.current?.click()} leftIcon={<FiCamera />}>
                  {selectedFile ? 'Change Video' : 'Select Video'}
                </Button>
                {selectedFile && (
                  <>
                    <span className="text-sm text-gray-600">
                      {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                    </span>
                    <Button variant="accent" onClick={uploadSelectedFile} leftIcon={<FiVideo />}>
                      Submit Video
                    </Button>
                  </>
                )}
              </>
            )}

            {(state === 'validating' || state === 'countdown') && (
              <Button variant="secondary" onClick={stopValidation}>Cancel</Button>
            )}

            {state === 'recording' && (
              <Button variant="danger" onClick={stopRecording} leftIcon={<FiStopCircle />}>
                Stop Recording
              </Button>
            )}

            {state === 'done' && (
              <Button variant="accent" onClick={recordAgain} leftIcon={<FiCamera />}>
                {inputMode === 'record' ? 'Record Again' : 'Upload Another'}
              </Button>
            )}
          </div>
        </div>

        {/* ── Results ── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
          className="bg-white rounded-xl p-6 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900">Translation Output</h3>
            {result?.recognized_text && (
              <button onClick={speakText} disabled={isSpeaking}
                className={`p-2 rounded-lg transition-colors ${isSpeaking ? 'bg-primary-100 text-primary-600' : 'hover:bg-gray-100'}`}
                title="Speak text">
                <FiVolume2 size={20} />
              </button>
            )}
          </div>

          <div className="min-h-[120px]">
            {result ? (
              <>
                {/* PSL mode: show Urdu text with RTL styling */}
                {result.sign_language === 'PSL' ? (
                  <>
                    <div className="flex items-center gap-2 mb-3">
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800">
                        🇵🇰 PSL - Pakistan Sign Language
                      </span>
                    </div>
                    <p
                      className="text-gray-800 text-3xl font-bold mb-4"
                      dir="rtl"
                      style={{ fontFamily: 'serif', letterSpacing: '0.05em' }}
                    >
                      {result.recognized_text}
                    </p>
                  </>
                ) : (
                  <p className="text-gray-800 text-2xl font-medium mb-4">{result.recognized_text}</p>
                )}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <span>Confidence:</span>
                    <div className="flex-1 max-w-xs bg-gray-200 rounded-full h-2">
                      <div className="bg-primary-600 h-2 rounded-full transition-all"
                        style={{ width: `${result.confidence * 100}%` }} />
                    </div>
                    <span className="font-medium">{Math.round(result.confidence * 100)}%</span>
                  </div>
                  <div className="text-sm text-gray-600">
                    {result.sign_language === 'PSL' ? (
                      <>
                        <p>Top Predictions:</p>
                        <ul className="mt-2 space-y-1">
                          {(result.top_predictions ?? []).map((pred: { label: string; confidence: number }, i: number) => (
                            <li key={i} className="flex items-center gap-2">
                              <span className="w-6 text-gray-400">{i + 1}.</span>
                              <span className="flex-1 text-right" dir="rtl" style={{ fontFamily: 'serif' }}>{pred.label}</span>
                              <div className="flex-1 max-w-[200px] bg-gray-200 rounded-full h-1.5">
                                <div className="bg-primary-400 h-1.5 rounded-full" style={{ width: `${pred.confidence * 100}%` }} />
                              </div>
                              <span className="w-12 text-right text-xs">{Math.round(pred.confidence * 100)}%</span>
                            </li>
                          ))}
                        </ul>
                      </>
                    ) : (
                      <>
                        <p>Top 5 Predictions:</p>
                        <ul className="mt-2 space-y-1">
                          {result.top5_predictions.map(([, conf], i) => (
                            <li key={i} className="flex items-center gap-2">
                              <span className="w-6 text-gray-400">{i + 1}.</span>
                              <span className="flex-1">{result.glosses[i]}</span>
                              <div className="flex-1 max-w-[200px] bg-gray-200 rounded-full h-1.5">
                                <div className="bg-primary-400 h-1.5 rounded-full" style={{ width: `${conf * 100}%` }} />
                              </div>
                              <span className="w-12 text-right text-xs">{Math.round(conf * 100)}%</span>
                            </li>
                          ))}
                        </ul>
                      </>
                    )}
                  </div>
                  <div className="text-xs text-gray-500 pt-2 border-t">
                    Processed {result.frame_count} frames in {result.processing_time_ms}ms
                  </div>
                </div>
              </>
            ) : error ? (
              <div className="flex items-center gap-2 text-red-600">
                <FiXCircle size={20} />
                <p>{error}</p>
              </div>
            ) : (
              <p className="text-gray-400">
                {state === 'idle' ? 'Start validation to begin translating sign language'
                  : state === 'validating' ? 'Checking environment — meet all 3 requirements to auto-start'
                  : state === 'countdown' ? 'All requirements met! Recording starts automatically…'
                  : state === 'recording' ? 'Recording your signs…'
                  : state === 'review' ? 'Review your recording above'
                  : state === 'processing' ? 'Processing video…'
                  : ''}
              </p>
            )}
          </div>
        </motion.div>

      </motion.div>

      {/* Inline keyframe for recording pulse */}
      <style>{`
        @keyframes recordingPulse {
          0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
          50% { opacity: 0.7; box-shadow: 0 0 0 8px rgba(239,68,68,0); }
        }
      `}</style>
    </div>
  );
}

function ValidationItem({
  label, status, message, isGood, detail,
}: {
  label: string; status: string; message: string; isGood: boolean;
  detail?: { label: string; ok: boolean }[];
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="mt-0.5">
        {isGood
          ? <FiCheckCircle className="text-green-600" size={20} />
          : <FiAlertCircle className="text-yellow-600" size={20} />}
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-900">{label}</span>
          <span className={`text-xs px-2 py-0.5 rounded-full ${isGood ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
            {status}
          </span>
        </div>
        <p className="text-sm text-gray-600 mt-0.5">{message}</p>
        {detail && (
          <div className="flex flex-wrap gap-2 mt-1.5">
            {detail.map(({ label: dl, ok }) => (
              <span
                key={dl}
                className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full ${
                  ok ? 'bg-green-50 text-green-700' : 'bg-gray-100 text-gray-500'
                }`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${ok ? 'bg-green-500' : 'bg-gray-400'}`} />
                {dl}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
