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
  FiUpload
} from 'react-icons/fi';
import Button from '@/components/ui/Button';
import Select from '@/components/ui/Select';
import api from '@/lib/api';
import { SignLanguage, ValidationState, PredictionResult } from '@/types';

type PageState = 'idle' | 'validating' | 'ready' | 'recording' | 'processing' | 'done';
type InputMode = 'record' | 'upload';

export default function SignToTextPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const validationIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const [state, setState] = useState<PageState>('idle');
  const [inputMode, setInputMode] = useState<InputMode>('record');
  const [signLanguage, setSignLanguage] = useState<SignLanguage>('ASL');
  const [validationState, setValidationState] = useState<ValidationState | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle file selection
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    const validTypes = ['video/mp4', 'video/webm', 'video/avi', 'video/mov'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|webm|avi|mov)$/i)) {
      toast.error('Please select a valid video file (MP4, WEBM, AVI, MOV)');
      return;
    }

    // Validate file size (max 50MB)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      toast.error('Video file is too large (max 50MB)');
      return;
    }

    setSelectedFile(file);
    toast.success('Video selected! Click "Submit Video" to process.');
  }, []);

  // Upload selected file
  const uploadSelectedFile = useCallback(async () => {
    if (!selectedFile) return;

    setState('processing');
    setError(null);

    try {
      const formData = new FormData();
      formData.append('video', selectedFile);
      formData.append('use_tta', 'true');

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/ml/sign-to-text`, {
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
      console.error('Upload error:', err);
      setError(err.message || 'Failed to process video');
      toast.error(err.message || 'Failed to process video');
      setState('idle');
    }
  }, [selectedFile]);

  // Start validation - request webcam and begin validation loop
  const startValidation = useCallback(async () => {
    try {
      setError(null);
      setState('validating');

      // Request webcam access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      // Start validation interval (5 fps = 200ms)
      validationIntervalRef.current = setInterval(() => {
        sendFrameForValidation();
      }, 200);

    } catch (err) {
      console.error('Failed to access webcam:', err);
      toast.error('Failed to access webcam. Please grant camera permissions.');
      setState('idle');
    }
  }, []);

  // Send frame to validation endpoint
  const sendFrameForValidation = useCallback(async () => {
    if (!videoRef.current || state === 'recording' || state === 'processing') {
      return;
    }

    try {
      // Capture frame from video element
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      
      if (!ctx) return;

      ctx.drawImage(videoRef.current, 0, 0);
      
      // Convert to blob
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((b) => resolve(b!), 'image/jpeg', 0.8);
      });

      // Send to validation endpoint
      const formData = new FormData();
      formData.append('frame', blob, 'frame.jpg');

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/ml/validate-environment`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data: ValidationState = await response.json();
        setValidationState(data);

        // Update state based on validation
        if (data.overall === 'ready' && state === 'validating') {
          setState('ready');
        } else if (data.overall === 'not_ready' && state === 'ready') {
          setState('validating');
        }
      }
    } catch (err) {
      console.error('Validation error:', err);
    }
  }, [state]);

  // Stop validation
  const stopValidation = useCallback(() => {
    if (validationIntervalRef.current) {
      clearInterval(validationIntervalRef.current);
      validationIntervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setState('idle');
    setValidationState(null);
  }, []);

  // Start recording
  const startRecording = useCallback(() => {
    if (!streamRef.current) return;

    try {
      recordedChunksRef.current = [];
      setRecordingTime(0);

      const mediaRecorder = new MediaRecorder(streamRef.current, {
        mimeType: 'video/webm;codecs=vp9',
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        uploadVideo();
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setState('recording');

      // Start timer
      const startTime = Date.now();
      const timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        setRecordingTime(elapsed);

        // Auto-stop at 30 seconds
        if (elapsed >= 30) {
          stopRecording();
          clearInterval(timerInterval);
        }
      }, 1000);

    } catch (err) {
      console.error('Failed to start recording:', err);
      toast.error('Failed to start recording');
      setState('ready');
    }
  }, []);

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  }, []);

  // Upload video for processing
  const uploadVideo = useCallback(async () => {
    setState('processing');

    try {
      const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });

      // Validate file size (max 50MB)
      const maxSize = 50 * 1024 * 1024;
      if (blob.size > maxSize) {
        throw new Error('Video file is too large (max 50MB)');
      }

      // Upload to sign-to-text endpoint
      const formData = new FormData();
      formData.append('video', blob, 'recording.webm');
      formData.append('use_tta', 'true');

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/ml/sign-to-text`, {
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
      console.error('Upload error:', err);
      setError(err.message || 'Failed to process video');
      toast.error(err.message || 'Failed to process video');
      setState('ready');
    }
  }, []);

  // Record again
  const recordAgain = useCallback(() => {
    setResult(null);
    setError(null);
    setRecordingTime(0);
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    setState('idle');
  }, []);

  // Speak result text
  const speakText = async () => {
    if (!result?.recognized_text) return;

    setIsSpeaking(true);
    try {
      const response = await api.textToSpeech(result.recognized_text, 'en', 'default');
      const audio = new Audio(api.getAudioUrl(response.audio_url.split('/').pop()!));
      audio.onended = () => setIsSpeaking(false);
      audio.onerror = () => {
        setIsSpeaking(false);
        toast.error('Failed to play audio');
      };
      audio.play();
    } catch {
      setIsSpeaking(false);
      toast.error('Failed to generate speech');
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (validationIntervalRef.current) {
        clearInterval(validationIntervalRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        {/* Video Feed */}
        <div className="relative">
          <div className="camera-container mx-auto bg-gray-200 rounded-xl overflow-hidden">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />

            {/* Recording Indicator */}
            {state === 'recording' && (
              <div className="recording-indicator">
                <div className="recording-dot"></div>
                <span>Recording {recordingTime}s</span>
              </div>
            )}

            {/* Camera Placeholder */}
            {state === 'idle' && (
              <div className="camera-placeholder">
                <FiCamera size={64} className="text-gray-400" />
                <p className="mt-4 text-gray-500">Click "Start Validation" to begin</p>
              </div>
            )}

            {/* Processing Overlay */}
            <AnimatePresence>
              {state === 'processing' && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 bg-white/90 backdrop-blur-sm rounded-xl flex flex-col items-center justify-center"
                >
                  <h3 className="text-primary-600 font-semibold text-lg mb-2">
                    Processing Video
                  </h3>
                  <p className="text-gray-600 mb-4">Translating your signs...</p>
                  <div className="loading-dots text-primary-600">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <p className="text-gray-500 text-sm mt-2">This may take a few seconds</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Validation Feedback */}
          {(state === 'validating' || state === 'ready' || state === 'recording') && validationState && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 bg-white rounded-lg p-4 shadow-md"
            >
              <h4 className="font-semibold text-gray-900 mb-3">Environment Check</h4>
              
              <div className="space-y-2">
                {/* Background */}
                <ValidationItem
                  label="Background"
                  status={validationState.background?.status || 'cluttered'}
                  message={validationState.background?.message || ''}
                  isGood={validationState.background?.status === 'clear' || validationState.background?.status === 'acceptable'}
                />

                {/* Body Visibility */}
                <ValidationItem
                  label="Body Visibility"
                  status={validationState.body?.status || 'not_visible'}
                  message={validationState.body?.message || ''}
                  isGood={validationState.body?.status === 'visible'}
                />

                {/* Distance */}
                <ValidationItem
                  label="Distance"
                  status={validationState.distance?.status || 'out_of_range'}
                  message={validationState.distance?.message || ''}
                  isGood={validationState.distance?.status === 'optimal' || validationState.distance?.status === 'acceptable'}
                />
              </div>
            </motion.div>
          )}
        </div>

        {/* Controls */}
        <div className="space-y-4">
          {/* Sign Language Selection */}
          <div className="flex justify-center">
            <Select
              value={signLanguage}
              onChange={(e) => setSignLanguage(e.target.value as SignLanguage)}
              options={[
                { value: 'ASL', label: 'ASL - American Sign Language' },
                { value: 'PSL', label: 'PSL - Pakistan Sign Language' },
              ]}
              className="w-64"
              disabled={state !== 'idle'}
            />
          </div>

          {/* Input Mode Selection */}
          {state === 'idle' && (
            <div className="flex justify-center gap-4">
              <Button
                variant={inputMode === 'record' ? 'primary' : 'secondary'}
                onClick={() => setInputMode('record')}
                leftIcon={<FiVideo />}
              >
                Record Video
              </Button>
              <Button
                variant={inputMode === 'upload' ? 'primary' : 'secondary'}
                onClick={() => setInputMode('upload')}
                leftIcon={<FiCamera />}
              >
                Upload Video
              </Button>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex justify-center gap-4 flex-wrap items-center">
            {state === 'idle' && inputMode === 'record' && (
              <Button 
                variant="accent" 
                onClick={startValidation} 
                leftIcon={<FiCamera />}
                className="min-w-[180px]"
              >
                Start Validation
              </Button>
            )}

            {state === 'idle' && inputMode === 'upload' && (
              <>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/mp4,video/webm,video/avi,video/mov"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <Button
                  variant="secondary"
                  onClick={() => fileInputRef.current?.click()}
                  leftIcon={<FiCamera />}
                >
                  {selectedFile ? 'Change Video' : 'Select Video'}
                </Button>
                {selectedFile && (
                  <>
                    <span className="text-sm text-gray-600">
                      {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                    </span>
                    <Button
                      variant="accent"
                      onClick={uploadSelectedFile}
                      leftIcon={<FiVideo />}
                    >
                      Submit Video
                    </Button>
                  </>
                )}
              </>
            )}

            {(state === 'validating' || state === 'ready') && (
              <>
                <Button
                  variant="primary"
                  onClick={startRecording}
                  leftIcon={<FiVideo />}
                  disabled={validationState?.overall !== 'ready'}
                  title={validationState?.overall !== 'ready' ? 'Fix validation issues first' : ''}
                >
                  Start Recording
                </Button>
                <Button variant="secondary" onClick={stopValidation}>
                  Cancel
                </Button>
              </>
            )}

            {state === 'recording' && (
              <Button variant="danger" onClick={stopRecording} leftIcon={<FiStopCircle />}>
                Stop Recording
              </Button>
            )}

            {state === 'done' && (
              <>
                <Button variant="accent" onClick={recordAgain} leftIcon={<FiCamera />}>
                  {inputMode === 'record' ? 'Record Again' : 'Upload Another'}
                </Button>
              </>
            )}
          </div>
        </div>

        {/* Results */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-xl p-6 shadow-lg"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900">Translation Output</h3>
            {result?.recognized_text && (
              <button
                onClick={speakText}
                disabled={isSpeaking}
                className={`p-2 rounded-lg transition-colors ${
                  isSpeaking ? 'bg-primary-100 text-primary-600' : 'hover:bg-gray-100'
                }`}
                title="Speak text"
              >
                <FiVolume2 size={20} />
              </button>
            )}
          </div>

          <div className="min-h-[120px]">
            {result ? (
              <>
                <p className="text-gray-800 text-2xl font-medium mb-4">
                  {result.recognized_text}
                </p>

                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <span>Confidence:</span>
                    <div className="flex-1 max-w-xs bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-primary-600 h-2 rounded-full transition-all"
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                    <span className="font-medium">{Math.round(result.confidence * 100)}%</span>
                  </div>

                  <div className="text-sm text-gray-600">
                    <p>Top 5 Predictions:</p>
                    <ul className="mt-2 space-y-1">
                      {result.top5_predictions.map(([idx, conf], i) => (
                        <li key={i} className="flex items-center gap-2">
                          <span className="w-6 text-gray-400">{i + 1}.</span>
                          <span className="flex-1">{result.glosses[i]}</span>
                          <div className="flex-1 max-w-[200px] bg-gray-200 rounded-full h-1.5">
                            <div
                              className="bg-primary-400 h-1.5 rounded-full"
                              style={{ width: `${conf * 100}%` }}
                            />
                          </div>
                          <span className="w-12 text-right text-xs">{Math.round(conf * 100)}%</span>
                        </li>
                      ))}
                    </ul>
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
                {state === 'idle'
                  ? 'Start validation to begin translating sign language'
                  : state === 'validating'
                  ? 'Validating environment...'
                  : state === 'ready'
                  ? 'Environment ready! Click "Start Recording" to begin'
                  : state === 'recording'
                  ? 'Recording your signs...'
                  : state === 'processing'
                  ? 'Processing video...'
                  : ''}
              </p>
            )}
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}

// Validation Item Component
function ValidationItem({ 
  label, 
  status, 
  message, 
  isGood 
}: { 
  label: string; 
  status: string; 
  message: string; 
  isGood: boolean;
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="mt-0.5">
        {isGood ? (
          <FiCheckCircle className="text-green-600" size={20} />
        ) : (
          <FiAlertCircle className="text-yellow-600" size={20} />
        )}
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-900">{label}</span>
          <span className={`text-xs px-2 py-0.5 rounded-full ${
            isGood ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
          }`}>
            {status}
          </span>
        </div>
        <p className="text-sm text-gray-600 mt-0.5">{message}</p>
      </div>
    </div>
  );
}
