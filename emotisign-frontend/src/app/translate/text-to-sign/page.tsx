'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { FiSend, FiMic, FiMicOff, FiInfo } from 'react-icons/fi';
import Button from '@/components/ui/Button';
import Select from '@/components/ui/Select';
import EmotionBadge from '@/components/ui/EmotionBadge';
import SkeletonCanvas from '@/components/ui/SkeletonCanvas';
import api from '@/lib/api';
import { SignLanguage, SignData, EmotionAnalysis } from '@/types';

// Full 100-word ASL vocabulary supported by the sign generator
const ASL_VOCABULARY = [
  'accident','africa','all','apple','basketball','bed','before','bird',
  'birthday','black','blue','book','bowling','brown','but','can','candy',
  'chair','change','cheat','city','clothes','color','computer','cook','cool',
  'corn','cousin','cow','dance','dark','deaf','decide','doctor','dog','drink',
  'eat','enjoy','family','fine','finish','fish','forget','full','give','go',
  'graduate','hat','hearing','help','hot','how','jacket','kiss','language',
  'last','later','letter','like','man','many','medicine','meet','mother',
  'need','no','now','orange','paint','paper','pink','pizza','play','pull',
  'purple','right','same','school','secretary','shirt','short','son','study',
  'table','tall','tell','thanksgiving','thin','thursday','time','walk','want',
  'what','white','who','woman','work','wrong','year','yes',
];

type TranslationState = 'idle' | 'translating' | 'displaying' | 'done';

export default function TextToSignPage() {
  const signTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [state, setState] = useState<TranslationState>('idle');
  const [signLanguage, setSignLanguage] = useState<SignLanguage>('ASL');
  const [inputText, setInputText] = useState('');
  const [displayText, setDisplayText] = useState('');
  const [currentWord, setCurrentWord] = useState('');
  const [currentKeypoints, setCurrentKeypoints] = useState<number[][][] | null>(null);
  const [isFingerspelled, setIsFingerspelled] = useState(false);
  const [signs, setSigns] = useState<SignData[]>([]);
  const [emotion, setEmotion] = useState<EmotionAnalysis | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [showVocab, setShowVocab] = useState(false);
  const [debugLog, setDebugLog] = useState<string[]>([]);

  const log = (msg: string) => {
    console.log('[TTS]', msg);
    setDebugLog(prev => [...prev.slice(-6), msg]);
  };

  // ── Sequential sign playback ─────────────────────────────────────────────
  // Uses a recursive setTimeout chain. Each call schedules the NEXT sign
  // after the current one finishes playing.

  const playSign = useCallback((signList: SignData[], idx: number) => {
    if (idx >= signList.length) {
      setState('done');
      setCurrentWord('');
      setCurrentKeypoints(null);
      return;
    }

    const sign = signList[idx];
    setCurrentWord(sign.word);
    log(`playSign idx=${idx} word="${sign.word}" keypoints=${sign.keypoints?.length ?? 'null'} frames=${sign.frames.length}`);

    if (sign.keypoints && sign.keypoints.length > 0) {
      setIsFingerspelled(false);
      setCurrentKeypoints(sign.keypoints);
      // Play for the full duration of the sign (50ms × frames) + 200ms pause
      const durationMs = sign.keypoints.length * 50 + 200;
      signTimerRef.current = setTimeout(() => playSign(signList, idx + 1), durationMs);
    } else {
      // Fingerspelled — show word text, no canvas
      setIsFingerspelled(true);
      setCurrentKeypoints(null);
      const durationMs = Math.max(sign.frames.length * 250, 600);
      signTimerRef.current = setTimeout(() => playSign(signList, idx + 1), durationMs);
    }
  }, []);

  // ── REST translation ─────────────────────────────────────────────────────

  const translate = async () => {
    if (!inputText.trim()) {
      toast.error('Please enter some text');
      return;
    }

    // Clear previous playback
    if (signTimerRef.current) clearTimeout(signTimerRef.current);
    setState('translating');
    setDisplayText(inputText);
    setSigns([]);
    setCurrentKeypoints(null);
    setCurrentWord('');
    setIsFingerspelled(false);
    setEmotion(null);

    try {
      const response = await api.textToSign({
        text: inputText,
        sign_language: signLanguage,
        include_emotion: true,
      });

      log(`API OK: ${response.signs.length} signs, fingerspelled=[${response.fingerspelled_words}]`);
      response.signs.forEach((s: any) => {
        log(`  sign "${s.word}": keypoints=${s.keypoints?.length ?? 'null'} frames=${s.frames?.length}`);
      });

      setSigns(response.signs);
      if (response.emotion_analysis) setEmotion(response.emotion_analysis);

      if (response.signs.length > 0) {
        setState('displaying');
        playSign(response.signs, 0);
      } else {
        setState('done');
      }
    } catch (err) {
      console.error('Translation error:', err);
      log(`ERROR: ${String(err)}`);
      toast.error('Translation failed — is the backend running?');
      setState('idle');
    }
  };

  // ── Voice input ──────────────────────────────────────────────────────────

  const startVoiceInput = () => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      toast.error('Speech recognition not supported in this browser');
      return;
    }
    const SR = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
    const recognition = new SR();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);
    recognition.onresult = (event: any) => {
      const transcript = Array.from(event.results)
        .map((r: any) => r[0].transcript)
        .join('');
      setInputText(transcript);
    };
    recognition.onerror = () => { setIsListening(false); toast.error('Speech recognition error'); };
    recognition.start();
  };

  // ── Cleanup ──────────────────────────────────────────────────────────────

  useEffect(() => {
    return () => { if (signTimerRef.current) clearTimeout(signTimerRef.current); };
  }, []);

  // ── Render ───────────────────────────────────────────────────────────────

  const isActive = state === 'translating' || state === 'displaying';

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>

        {/* ── Canvas viewport ── */}
        <div
          style={{ background: '#1a1a2e', borderRadius: '0.75rem', position: 'relative',
                   width: '100%', aspectRatio: '1 / 1', maxHeight: 480,
                   display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}
        >
          {/* Skeleton canvas — always mounted when keypoints available */}
          {currentKeypoints && currentKeypoints.length > 0 ? (
            <SkeletonCanvas
              key={currentWord}          /* remount on each new word */
              frames={currentKeypoints}
              word={currentWord}
              frameRateMs={50}
              width={480}
              height={480}
              className="w-full h-full"
            />
          ) : state === 'translating' ? (
            <div className="flex flex-col items-center gap-3">
              <div className="loading-dots text-white/60"><span/><span/><span/></div>
              <p className="text-white/60 text-sm">Generating signs…</p>
            </div>
          ) : isFingerspelled && currentWord ? (
            <div className="text-center px-6">
              <p className="text-white/40 text-xs uppercase tracking-widest mb-2">Fingerspelling</p>
              <p className="text-white text-5xl font-bold tracking-widest uppercase">{currentWord}</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3 text-white/30">
              <svg className="w-14 h-14" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.2}
                  d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
              </svg>
              <p className="text-sm">Enter text below to see ASL signs</p>
            </div>
          )}

          {/* Word label */}
          {currentWord && (
            <div style={{ position: 'absolute', bottom: 12, left: '50%', transform: 'translateX(-50%)',
                          background: 'rgba(0,0,0,0.65)', color: 'white', padding: '4px 16px',
                          borderRadius: 9999, fontSize: 14, fontWeight: 500, whiteSpace: 'nowrap' }}>
              {currentWord}
            </div>
          )}

          {/* Debug overlay — remove after fixing */}
          {debugLog.length > 0 && (
            <div style={{ position: 'absolute', top: 8, left: 8, right: 8,
                          background: 'rgba(0,0,0,0.8)', color: '#0f0', fontFamily: 'monospace',
                          fontSize: 11, padding: 8, borderRadius: 6, pointerEvents: 'none' }}>
              {debugLog.map((l, i) => <div key={i}>{l}</div>)}
            </div>
          )}
        </div>

        {/* ── Language selector ── */}
        <div className="flex justify-center mt-4">
          <Select
            value={signLanguage}
            onChange={(e) => setSignLanguage(e.target.value as SignLanguage)}
            options={[
              { value: 'ASL', label: 'ASL - American Sign Language' },
              { value: 'PSL', label: 'PSL - Pakistan Sign Language' },
            ]}
            className="w-64"
          />
        </div>

        {/* ── Input panel ── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
          className="bg-white rounded-xl p-6 shadow-lg mt-4"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-gray-900">
              {displayText && state !== 'idle' ? `Showing signs for: "${displayText}"` : 'Enter Your Text'}
            </h3>
            <button
              onClick={() => setShowVocab(v => !v)}
              className="flex items-center gap-1 text-xs text-primary-600 hover:text-primary-800 transition-colors"
            >
              <FiInfo size={14} />
              {showVocab ? 'Hide' : 'View'} vocabulary
            </button>
          </div>

          {/* Vocabulary panel */}
          {showVocab && (
            <div className="mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
              <p className="text-xs text-gray-500 mb-2 font-medium">
                {ASL_VOCABULARY.length} supported ASL words — click to use
              </p>
              <div className="flex flex-wrap gap-1.5 max-h-40 overflow-y-auto">
                {ASL_VOCABULARY.map(word => (
                  <button
                    key={word}
                    onClick={() => setInputText(prev => prev ? `${prev} ${word}` : word)}
                    className="px-2 py-0.5 text-xs bg-primary-100 text-primary-700 rounded-full hover:bg-primary-200 transition-colors capitalize"
                  >
                    {word}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="flex gap-3">
            <div className="flex-1">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey && !isActive) { e.preventDefault(); translate(); } }}
                placeholder="Type a word or sentence… (e.g. want help eat basketball)"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-200 focus:border-primary-500 outline-none resize-none text-sm"
                rows={3}
                disabled={isActive}
              />
            </div>
            <div className="flex flex-col gap-2">
              <button
                onClick={startVoiceInput}
                disabled={isListening || isActive}
                className={`p-3 rounded-lg transition-colors ${isListening ? 'bg-red-100 text-red-600' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                title="Voice input"
              >
                {isListening ? <FiMicOff size={20} /> : <FiMic size={20} />}
              </button>
              <Button
                onClick={translate}
                disabled={!inputText.trim() || isActive}
                isLoading={state === 'translating'}
                title="Translate"
              >
                <FiSend size={20} />
              </Button>
            </div>
          </div>

          {emotion && (
            <div className="mt-3">
              <EmotionBadge emotion={emotion.emotion} />
            </div>
          )}
        </motion.div>

        {/* ── Sign preview strip ── */}
        {signs.length > 0 && state === 'done' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-xl p-6 shadow-lg mt-4"
          >
            <h3 className="font-semibold text-gray-900 mb-4">
              Signs ({signs.length} word{signs.length !== 1 ? 's' : ''})
            </h3>
            <div className="flex flex-wrap gap-3">
              {signs.map((sign, i) => (
                <button
                  key={i}
                  onClick={() => {
                    if (signTimerRef.current) clearTimeout(signTimerRef.current);
                    setState('displaying');
                    playSign(signs, i);
                  }}
                  className="text-center group"
                  title={`Replay "${sign.word}"`}
                >
                  <div className="w-20 h-20 rounded-lg overflow-hidden flex items-center justify-center bg-[#1a1a2e] group-hover:ring-2 group-hover:ring-primary-400 transition-all">
                    {sign.keypoints && sign.keypoints.length > 0 ? (
                      <SkeletonCanvas
                        frames={[sign.keypoints[Math.floor(sign.keypoints.length / 2)]]}
                        word={sign.word}
                        frameRateMs={999999}
                        width={80}
                        height={80}
                      />
                    ) : (
                      <span className="text-white text-xl font-bold uppercase">{sign.word[0]}</span>
                    )}
                  </div>
                  <p className="text-xs text-gray-600 mt-1 capitalize">{sign.word}</p>
                  {sign.fingerspelled && (
                    <span className="text-xs text-amber-600">spelled</span>
                  )}
                </button>
              ))}
            </div>
            <p className="text-xs text-gray-400 mt-3">Click any sign to replay it</p>
          </motion.div>
        )}

      </motion.div>
    </div>
  );
}
