'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import { motion, AnimatePresence } from 'framer-motion';
import { format } from 'date-fns';
import toast from 'react-hot-toast';
import fixWebmDuration from 'fix-webm-duration';
import {
  FiArrowLeft, FiSend, FiUser, FiPlus, FiSearch, FiX,
  FiMic, FiMicOff, FiCamera, FiStopCircle, FiVolume2, FiEye,
  FiMessageSquare, FiWifi, FiWifiOff, FiUpload, FiVideo,
} from 'react-icons/fi';
import Link from 'next/link';
import Button from '@/components/ui/Button';
import EmotionBadge from '@/components/ui/EmotionBadge';
import SkeletonCanvas from '@/components/ui/SkeletonCanvas';
import { createChatWS, createSignToTextWS, EmotiSignWebSocket } from '@/lib/websocket';
import { usePSLWebSocket, PSLPredictionResult } from '@/hooks/usePSLWebSocket';
import api from '@/lib/api';
import { useAuthStore } from '@/store/auth';
import { ChatRoom, ChatMessage, User, Emotion, SignLanguage } from '@/types';

export default function ChatPage() {
  const { user, isGuest, fetchUser } = useAuthStore();

  // Refs
  const chatWsRef = useRef<EmotiSignWebSocket | null>(null);
  const signWsRef = useRef<EmotiSignWebSocket | null>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);
  const signAnimRef = useRef<NodeJS.Timeout | null>(null);
  const typingTimerRef = useRef<Record<number, NodeJS.Timeout>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const webcamRef = useRef<Webcam>(null);
  const pslVideoRef = useRef<HTMLVideoElement>(null);
  const pslWebcamRef = useRef<Webcam>(null);
  const pslFrameIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Rooms & messages
  const [rooms, setRooms] = useState<ChatRoom[]>([]);
  const [selectedRoom, setSelectedRoom] = useState<ChatRoom | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoadingRooms, setIsLoadingRooms] = useState(true);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [roomSearchQuery, setRoomSearchQuery] = useState('');

  // Online / typing
  const [onlineUserIds, setOnlineUserIds] = useState<number[]>([]);
  const [typingUsers, setTypingUsers] = useState<{ user_id: number; username: string }[]>([]);

  // New chat modal
  const [showNewChatModal, setShowNewChatModal] = useState(false);
  const [userSearchQuery, setUserSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<User[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isCreatingRoom, setIsCreatingRoom] = useState(false);

  // Sign recorder modal
  const [showSignRecorder, setShowSignRecorder] = useState(false);
  const [signChatMode, setSignChatMode] = useState<'asl' | 'psl'>('asl');
  const [aslSubMode, setAslSubMode] = useState<'live' | 'video'>('live');
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [recognizedText, setRecognizedText] = useState('');
  const [detectedEmotion, setDetectedEmotion] = useState<Emotion | null>(null);
  const [pslLabel, setPslLabel] = useState('');
  const [pslRecordedBlob, setPslRecordedBlob] = useState<Blob | null>(null);
  const [isPslRecording, setIsPslRecording] = useState(false);
  const [pslRecognizedText, setPslRecognizedText] = useState(''); // Accumulated PSL letters
  const [pslLastLetter, setPslLastLetter] = useState(''); // Most recent detected letter
  const [aslRecordedBlob, setAslRecordedBlob] = useState<Blob | null>(null);
  const [aslSelectedFile, setAslSelectedFile] = useState<File | null>(null);
  const [aslFilePreviewUrl, setAslFilePreviewUrl] = useState<string | null>(null);
  const [isAslVideoRecording, setIsAslVideoRecording] = useState(false);
  const [isUploadingSign, setIsUploadingSign] = useState(false);
  const pslMediaRecorderRef = useRef<MediaRecorder | null>(null);
  const aslMediaRecorderRef = useRef<MediaRecorder | null>(null);
  const pslStreamRef = useRef<MediaStream | null>(null);
  const pslChunksRef = useRef<Blob[]>([]);
  const aslChunksRef = useRef<Blob[]>([]);
  const aslVideoFileInputRef = useRef<HTMLInputElement>(null);
  const pslRecordingStartTimeRef = useRef<number>(0); // Track actual recording start time
  const aslRecordingStartTimeRef = useRef<number>(0); // Track ASL recording start time

  // Speech-to-text
  const [isListening, setIsListening] = useState(false);

  // ── PSL WebSocket for live recognition ───────────────────────────────────
  const pslWs = usePSLWebSocket({
    onConnected: useCallback(() => {
      console.log('✅ PSL WebSocket connected');
    }, []),
    onResult: useCallback((result: PSLPredictionResult) => {
      console.log('🎯 PSL letter detected:', result.predicted_label, 'confidence:', result.confidence);
      // Accumulate recognized letters (filter out duplicates)
      setPslLastLetter(result.predicted_label);
      setPslRecognizedText(prev => {
        // Simple deduplication: don't add if it's the same as last character
        const lastChar = prev.slice(-1);
        if (lastChar === result.predicted_label) {
          console.log('⏭️  Skipping duplicate:', result.predicted_label);
          return prev;
        }
        const newText = prev + result.predicted_label;
        console.log('✅ Updated text:', newText);
        return newText;
      });
    }, []),
    onNoHand: useCallback(() => {
      console.log('👋 No hand detected');
    }, []),
    onLowConfidence: useCallback((data: {
      predicted_label: string;
      confidence: number;
      emotion?: string;
      emotion_emoji?: string;
    }) => {
      console.log('❓ Low confidence:', data.predicted_label, data.confidence);
      setPslLastLetter(data.predicted_label + '?');
    }, []),
    onError: useCallback((message: string) => {
      console.error('❌ PSL WS Error:', message);
    }, []),
  });

  // Sign viewer modal
  const [showSignModal, setShowSignModal] = useState(false);
  const [signModalText, setSignModalText] = useState('');
  const [signFrames, setSignFrames] = useState<string[]>([]);
  const [currentSignFrame, setCurrentSignFrame] = useState(0);
  const [isLoadingSigns, setIsLoadingSigns] = useState(false);

  // ── Init ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const init = async () => {
      if (!isGuest) {
        try { await fetchUser(); } catch {}
      }
      setIsLoadingRooms(true);
      try {
        const data = await api.getChatRooms();
        setRooms(data || []);
      } catch (err: any) {
        if (err.response?.status !== 401) {
          toast.error('Failed to load chats. Is the backend running?');
        }
      } finally {
        setIsLoadingRooms(false);
      }
    };
    init();

    return () => { disconnectChat(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Disconnect helper ─────────────────────────────────────────────────
  const disconnectChat = useCallback(() => {
    chatWsRef.current?.disconnect();
    chatWsRef.current = null;
    setWsConnected(false);
    setOnlineUserIds([]);
    setTypingUsers([]);
  }, []);

  // ── Select room ───────────────────────────────────────────────────────
  const selectRoom = useCallback(async (room: ChatRoom) => {
    disconnectChat();
    setSelectedRoom(room);
    setMessages([]);

    // Load history
    setIsLoadingMessages(true);
    try {
      const data = await api.getChatMessages(room.id);
      // API returns newest-first; reverse for chronological display
      setMessages([...(data?.items || [])].reverse());
    } catch (err: any) {
      if (err.response?.status !== 401) toast.error('Failed to load messages');
    } finally {
      setIsLoadingMessages(false);
    }

    api.markMessagesAsRead(room.id).catch(() => {});

    // ── Connect WebSocket ──────────────────────────────────────────────
    const ws = createChatWS(room.id);
    chatWsRef.current = ws;

    ws.on('connected', () => {
      setWsConnected(true);
      ws.send({ type: 'ping' });
    });

    ws.on('disconnected', () => setWsConnected(false));

    // THE KEY FIX: listen on 'message' with full object; dispatch by event name ourselves.
    // The WS class emits 'message' with the complete { event, data, timestamp } object.
    ws.on('message', (fullMsg: any) => {
      const eventName: string = fullMsg.event || fullMsg.type || '';
      const payload = fullMsg.data ?? fullMsg;

      switch (eventName) {
        case 'message': {
          const newMsg = payload as ChatMessage;
          setMessages((prev) => {
            if (prev.find((m) => m.id === newMsg.id)) return prev; // deduplicate
            return [...prev, newMsg];
          });
          api.markMessagesAsRead(room.id).catch(() => {});
          break;
        }

        case 'typing': {
          const td = payload as { user_id: number; username: string };
          // Don't show our own typing indicator
          if (td.user_id === useAuthStore.getState().user?.id) break;

          setTypingUsers((prev) =>
            prev.find((u) => u.user_id === td.user_id) ? prev : [...prev, td]
          );

          // Clear after 3 s
          clearTimeout(typingTimerRef.current[td.user_id]);
          typingTimerRef.current[td.user_id] = setTimeout(() => {
            setTypingUsers((prev) => prev.filter((u) => u.user_id !== td.user_id));
          }, 3000);
          break;
        }

        case 'online_users':
          setOnlineUserIds(payload?.users ?? []);
          break;

        case 'user_joined':
          setOnlineUserIds(payload?.online_users ?? []);
          break;

        case 'user_left':
          setOnlineUserIds(payload?.online_users ?? []);
          break;

        case 'read':
          setMessages((prev) =>
            prev.map((m) =>
              m.sender_id !== useAuthStore.getState().user?.id
                ? { ...m, is_read: true }
                : m
            )
          );
          break;

        case 'pong':
          break;

        case 'error':
          console.error('Chat WS error:', payload?.detail);
          break;

        default:
          break;
      }
    });

    ws.on('error', () => setWsConnected(false));

    try {
      await ws.connect();
    } catch {
      toast.error('Could not connect to chat server');
    }
  }, [disconnectChat]);

  // ── Send message ──────────────────────────────────────────────────────
  const sendMessage = async () => {
    const content = inputMessage.trim();
    if (!content || !chatWsRef.current || !selectedRoom) return;

    if (!chatWsRef.current.isConnected()) {
      toast.error('Not connected — please wait a moment and try again');
      return;
    }

    setIsSending(true);
    try {
      chatWsRef.current.send({ type: 'text', content, auto_translate: true });
      setInputMessage('');
    } catch {
      toast.error('Failed to send message');
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleTypingInput = (value: string) => {
    setInputMessage(value);
    if (chatWsRef.current?.isConnected()) {
      chatWsRef.current.send({ type: 'typing' });
    }
  };

  // ── Auto-scroll ───────────────────────────────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typingUsers]);

  // ── User search ───────────────────────────────────────────────────────
  useEffect(() => {
    const t = setTimeout(() => {
      if (userSearchQuery.trim().length >= 2) {
        setIsSearching(true);
        api.searchUsers(userSearchQuery)
          .then(setSearchResults)
          .catch(() => toast.error('Failed to search users'))
          .finally(() => setIsSearching(false));
      } else {
        setSearchResults([]);
      }
    }, 300);
    return () => clearTimeout(t);
  }, [userSearchQuery]);

  const startNewChat = async (otherUser: User) => {
    setIsCreatingRoom(true);
    try {
      const room = await api.createChatRoom({
        name: otherUser.username,
        member_ids: [otherUser.id],
        is_direct: true,
      });
      setRooms((prev) => (prev.find((r) => r.id === room.id) ? prev : [room, ...prev]));
      setShowNewChatModal(false);
      setUserSearchQuery('');
      setSearchResults([]);
      await selectRoom(room);
      toast.success(`Chat started with ${otherUser.username}`);
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to create chat');
    } finally {
      setIsCreatingRoom(false);
    }
  };

  // ── Sign Recorder ─────────────────────────────────────────────────────
  const openSignRecorder = () => {
    setShowSignRecorder(true);
    setSignChatMode('asl');
    setRecognizedText('');
    setDetectedEmotion(null);
    setRecordingTime(0);
    setIsRecording(false);
    setPslLabel('');
    setPslRecordedBlob(null);
    setIsPslRecording(false);
    setAslSubMode('live');
    setAslRecordedBlob(null);
    setAslSelectedFile(null);
    setIsAslVideoRecording(false);
  };

  const needsCameraPreview =
    showSignRecorder &&
    ((signChatMode === 'psl') || (signChatMode === 'asl' && aslSubMode === 'video' && !aslSelectedFile));

  // PSL / ASL video tab: webcam preview
  useEffect(() => {
    if (!needsCameraPreview) return;
    let stream: MediaStream | null = null;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });
        pslStreamRef.current = stream;
        if (pslVideoRef.current) pslVideoRef.current.srcObject = stream;
      } catch {
        /* preview optional */
      }
    })();
    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, [needsCameraPreview, signChatMode, aslSubMode, aslSelectedFile]);

  useEffect(() => {
    if (!aslSelectedFile) {
      setAslFilePreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(aslSelectedFile);
    setAslFilePreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [aslSelectedFile]);

  const stopAslVideoRecording = useCallback(() => {
    if (aslMediaRecorderRef.current?.state === 'recording') {
      aslMediaRecorderRef.current.stop();
    }
    clearInterval(recordingTimerRef.current ?? undefined);
    recordingTimerRef.current = null;
    setIsAslVideoRecording(false);
  }, []);

  const startAslVideoRecording = async () => {
    if (isAslVideoRecording) return;
    setAslRecordedBlob(null);
    setAslSelectedFile(null);
    setRecordingTime(0);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      pslStreamRef.current = stream;
      if (pslVideoRef.current) pslVideoRef.current.srcObject = stream;
      aslChunksRef.current = [];
      const recorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
          ? 'video/webm;codecs=vp9'
          : 'video/webm',
      });
      aslMediaRecorderRef.current = recorder;
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) aslChunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        const blob = new Blob(aslChunksRef.current, { type: recorder.mimeType || 'video/webm' });
        
        // Fix WebM duration for ASL videos too
        try {
          const actualDuration = Date.now() - aslRecordingStartTimeRef.current;
          const fixedBlob = await fixWebmDuration(blob, actualDuration);
          console.log('✅ ASL video fixed - size:', fixedBlob.size, 'bytes');
          setAslRecordedBlob(fixedBlob);
        } catch (err) {
          console.error('❌ Failed to fix ASL WebM duration:', err);
          setAslRecordedBlob(blob);
        }
        
        stream.getTracks().forEach((t) => t.stop());
        pslStreamRef.current = null;
        if (pslVideoRef.current) pslVideoRef.current.srcObject = null;
      };
      recorder.start(200);
      aslRecordingStartTimeRef.current = Date.now();
      setIsAslVideoRecording(true);
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch {
      toast.error('Could not access camera for recording');
    }
  };

  const handleAslVideoFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const valid = ['video/mp4', 'video/webm', 'video/avi', 'video/quicktime'];
    if (!valid.includes(file.type) && !file.name.match(/\.(mp4|webm|avi|mov)$/i)) {
      toast.error('Use MP4, WEBM, AVI, or MOV');
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      toast.error('Video must be under 50MB');
      return;
    }
    stopAslVideoRecording();
    setAslRecordedBlob(null);
    setAslSelectedFile(file);
    toast.success('Video selected');
  };

  const sendAslSignVideo = async () => {
    const file =
      aslSelectedFile ??
      (aslRecordedBlob
        ? new File([aslRecordedBlob], 'asl-sign.webm', { type: aslRecordedBlob.type || 'video/webm' })
        : null);
    if (!file || !selectedRoom || !chatWsRef.current?.isConnected()) {
      toast.error('Record or upload an ASL video first');
      return;
    }
    setIsUploadingSign(true);
    try {
      const { video_url } = await api.uploadChatSignVideo(selectedRoom.id, file, 'ASL');
      chatWsRef.current.send({
        type: 'sign_video',
        video_url,
        sign_language: 'ASL',
        auto_translate: true,
      });
      toast.success('ASL video sent — translation will appear for your partner');
      cancelSignRecording();
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to send ASL video');
    } finally {
      setIsUploadingSign(false);
    }
  };

  const startSignRecording = async () => {
    if (isRecording) return; // guard against double-start

    setRecognizedText('');
    setDetectedEmotion(null);

    const signWs = createSignToTextWS();
    signWsRef.current = signWs;

    // Listen on 'message' to catch both partial and result events reliably
    signWs.on('message', (fullMsg: any) => {
      const ev = fullMsg.event || fullMsg.type || '';
      const d = fullMsg.data ?? fullMsg;
      if (ev === 'partial') {
        setRecognizedText(d.recognized_text || '');
      } else if (ev === 'result') {
        setRecognizedText(d.recognized_text || '');
        if (d.emotion) setDetectedEmotion(d.emotion as Emotion);
      } else if (ev === 'error') {
        toast.error(d.detail || 'Translation error');
      }
    });

    try {
      await signWs.connect();
      signWs.send({ type: 'config', sign_language: 'ASL', detect_emotion: true });
      setIsRecording(true);

      // Send webcam frames
      frameIntervalRef.current = setInterval(() => {
        if (webcamRef.current && signWsRef.current?.isConnected()) {
          const imageSrc = webcamRef.current.getScreenshot();
          if (imageSrc) {
            signWsRef.current.send({
              type: 'frame',
              data: imageSrc.split(',')[1],
              mime: 'image/jpeg',
            });
          }
        }
      }, 150);

      // Recording timer
      setRecordingTime(0);
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch {
      toast.error('Failed to connect to sign translation service');
    }
  };

  const stopSignRecording = useCallback(() => {
    clearInterval(frameIntervalRef.current ?? undefined);
    frameIntervalRef.current = null;
    clearInterval(recordingTimerRef.current ?? undefined);
    recordingTimerRef.current = null;

    if (signWsRef.current) {
      signWsRef.current.send({ type: 'stop' });
      signWsRef.current.disconnect();
      signWsRef.current = null;
    }
    setIsRecording(false);
  }, []);

  const sendSignMessage = () => {
    if (!recognizedText.trim() || !chatWsRef.current?.isConnected()) return;
    const content = recognizedText.trim();
    chatWsRef.current.send({ type: 'text', content, auto_translate: true });
    cancelSignRecording();
    toast.success('ASL message sent');
  };

  const stopPslRecording = useCallback(() => {
    console.log('⏹️ Stopping PSL recording...');
    
    // Stop MediaRecorder
    if (pslMediaRecorderRef.current?.state === 'recording') {
      console.log('⏹️ MediaRecorder state:', pslMediaRecorderRef.current.state);
      pslMediaRecorderRef.current.stop();
      // Note: blob will be created asynchronously in recorder.onstop callback
    } else {
      console.warn('⚠️ MediaRecorder not in recording state:', pslMediaRecorderRef.current?.state);
    }
    
    // Stop frame streaming
    if (pslFrameIntervalRef.current) {
      clearInterval(pslFrameIntervalRef.current);
      pslFrameIntervalRef.current = null;
      console.log('✅ Stopped frame streaming');
    }
    
    // Stop timer
    clearInterval(recordingTimerRef.current ?? undefined);
    recordingTimerRef.current = null;
    
    // Disconnect PSL WebSocket
    pslWs.disconnect();
    console.log('🔌 Disconnected PSL WebSocket');
    
    setIsPslRecording(false);
  }, [pslWs]);

  const startPslRecording = async () => {
    if (isPslRecording) return;
    
    // Reset state
    setPslRecordedBlob(null);
    setPslRecognizedText('');
    setPslLastLetter('');
    setRecordingTime(0);
    
    try {
      console.log('🚀 Starting PSL recording...');
      console.log('📡 PSL WebSocket current status:', pslWs.connectionStatus);
      console.log('📡 PSL WebSocket object:', pslWs);
      console.log('📡 Calling pslWs.connect()...');
      
      // Connect PSL WebSocket
      pslWs.connect();
      console.log('📡 pslWs.connect() called');
      
      // Wait for connection to establish (increased timeout)
      console.log('⏳ Waiting for WebSocket connection...');
      await new Promise(resolve => setTimeout(resolve, 2000));
      console.log('📡 PSL WebSocket status after connect:', pslWs.connectionStatus);
      
      if (pslWs.connectionStatus !== 'connected') {
        console.warn('⚠️ WebSocket not connected yet, but continuing. Status:', pslWs.connectionStatus);
        console.warn('⚠️ Frames will be sent once connection is established');
      } else {
        // Send a test ping to verify connection is really working
        console.log('📡 Sending test ping to verify connection...');
        try {
          pslWs.updateConfig(0.7); // This sends a message
          console.log('✅ Test message sent successfully');
        } catch (err) {
          console.error('❌ Failed to send test message:', err);
        }
      }
      
      // Give a bit more time after test message
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Start MediaRecorder for video capture
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      
      console.log('📹 Got media stream');
      
      pslStreamRef.current = stream;
      if (pslVideoRef.current) {
        pslVideoRef.current.srcObject = stream;
        console.log('📹 Set video srcObject');
        // Wait for video to be ready
        await new Promise((resolve) => {
          if (pslVideoRef.current) {
            pslVideoRef.current.onloadedmetadata = async () => {
              console.log('📹 Video metadata loaded');
              // Important: Start video playback
              try {
                await pslVideoRef.current!.play();
                console.log('▶️ Video playback started');
              } catch (err) {
                console.error('❌ Video play error:', err);
              }
              resolve(null);
            };
          } else {
            resolve(null);
          }
        });
      }
      
      pslChunksRef.current = [];
      const recorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
          ? 'video/webm;codecs=vp9'
          : 'video/webm',
      });
      
      pslMediaRecorderRef.current = recorder;
      
      recorder.ondataavailable = (e) => {
        console.log('📦 Data chunk received:', e.data.size, 'bytes');
        if (e.data.size > 0) {
          pslChunksRef.current.push(e.data);
        }
      };
      
      recorder.onstop = async () => {
        console.log('⏹️ MediaRecorder stopped, creating blob from chunks:', pslChunksRef.current.length);
        
        // Important: Give MediaRecorder time to finalize the WebM container
        // Without this, the WebM file may not have proper metadata
        await new Promise(resolve => setTimeout(resolve, 500));
        
        const blob = new Blob(pslChunksRef.current, { type: recorder.mimeType || 'video/webm' });
        console.log('📦 Initial blob created - size:', blob.size, 'bytes, type:', blob.type);
        
        if (blob.size === 0) {
          console.error('❌ Empty blob created! No video data recorded.');
          toast.error('Recording failed - no video data captured');
          stream.getTracks().forEach((t) => t.stop());
          pslStreamRef.current = null;
          if (pslVideoRef.current) {
            pslVideoRef.current.srcObject = null;
          }
          return;
        }
        
        // Fix WebM duration metadata using actual recording duration
        // This makes the file readable by OpenCV
        try {
          const actualDuration = Date.now() - pslRecordingStartTimeRef.current;
          console.log('🔧 Fixing WebM duration:', actualDuration, 'ms');
          const fixedBlob = await fixWebmDuration(blob, actualDuration);
          console.log('✅ Fixed blob created - size:', fixedBlob.size, 'bytes');
          
          if (fixedBlob.size < 1000) {
            console.warn('⚠️ Very small blob:', fixedBlob.size, 'bytes - may be incomplete');
            toast('Recording may be too short or incomplete', { icon: '⚠️' });
          } else {
            toast.success(`Video recorded: ${(fixedBlob.size / 1024).toFixed(0)}KB - click Send to upload`);
          }
          
          setPslRecordedBlob(fixedBlob);
        } catch (err) {
          console.error('❌ Failed to fix WebM duration:', err);
          // Fall back to unfixed blob
          toast('Video recorded but may have playback issues', { icon: '⚠️' });
          setPslRecordedBlob(blob);
        }
        
        stream.getTracks().forEach((t) => t.stop());
        pslStreamRef.current = null;
        if (pslVideoRef.current) {
          pslVideoRef.current.srcObject = null;
        }
      };
      
      // Request data more frequently for better reliability
      // Using timeslice of 100ms ensures we get chunks even for short recordings
      recorder.start(100);
      
      // Track recording start time for accurate duration calculation
      pslRecordingStartTimeRef.current = Date.now();
      
      setIsPslRecording(true);
      
      console.log('🎥 PSL Recording started, WebSocket status:', pslWs.connectionStatus);
      console.log('🔄 Starting frame streaming interval after short delay...');
      
      // Wait a bit for WebSocket to be fully ready (like standalone page does)
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Start sending frames to PSL WebSocket
      let frameCount = 0;
      pslFrameIntervalRef.current = setInterval(() => {
        frameCount++;
        
        const video = pslVideoRef.current;
        if (!video || video.readyState < 2) {
          if (frameCount % 20 === 0) {
            console.log('⏳ Video not ready yet, readyState:', video?.readyState, '(frame', frameCount, ')');
          }
          return;
        }
        
        // Capture frame using same method as standalone page
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        ctx.drawImage(video, 0, 0);
        
        // Use toBlob like standalone page
        canvas.toBlob((blob) => {
          if (!blob) return;
          const reader = new FileReader();
          reader.onloadend = () => {
            const dataUrl = reader.result as string;
            const base64 = dataUrl.split(',')[1];
            if (base64 && pslWs.isConnected()) {
              pslWs.sendFrame(base64);
              if (frameCount % 20 === 0) {
                console.log('📤 Frame', frameCount, 'sent to PSL WebSocket');
              }
            } else if (frameCount % 20 === 0 && pslWs.isConnected()) {
              console.log('❌ WebSocket not connected, status:', pslWs.connectionStatus, '(frame', frameCount, ')');
            }
          };
          reader.readAsDataURL(blob);
        }, 'image/jpeg', 0.8);
      }, 150); // ~6-7 FPS (150ms)
      
      // Recording timer with auto-stop at 10 seconds
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime((prev) => {
          const newTime = prev + 1;
          if (newTime >= 10) {
            // Auto-stop at 10 seconds
            stopPslRecording();
          }
          return newTime;
        });
      }, 1000);
      
    } catch (error) {
      console.error('PSL recording error:', error);
      toast.error('Could not access camera for PSL recording');
      pslWs.disconnect();
    }
  };

  const sendPslSignVideo = async () => {
    if (!pslRecordedBlob || !selectedRoom || !chatWsRef.current?.isConnected()) {
      toast.error('Record a PSL video first');
      return;
    }
    
    // Check if blob has actual data
    if (pslRecordedBlob.size === 0) {
      console.error('❌ Cannot send empty video blob');
      toast.error('Video recording is empty - please try recording again');
      return;
    }
    
    console.log('📤 Sending PSL video - size:', pslRecordedBlob.size, 'bytes');
    
    // Give a moment for any final blob finalization
    await new Promise(resolve => setTimeout(resolve, 200));
    
    // Use accumulated recognized text as label (fallback to "PSL Sign" if empty)
    const recognizedLabel = pslRecognizedText.trim() || 'PSL Sign';
    
    // Warn if no letters detected but allow sending
    if (!pslRecognizedText.trim()) {
      console.warn('⚠️ No PSL letters detected, sending with default label');
    } else {
      console.log('✅ Recognized text:', recognizedLabel);
    }
    
    setIsUploadingSign(true);
    try {
      // Normalize MIME type: remove codec info if present
      let mimeType = pslRecordedBlob.type || 'video/webm';
      if (mimeType.includes(';')) {
        mimeType = mimeType.split(';')[0]; // "video/webm;codecs=vp9" → "video/webm"
      }
      
      const file = new File([pslRecordedBlob], 'psl-sign.webm', {
        type: mimeType,
      });
      
      console.log('📦 Created file for upload:', {
        size: file.size,
        type: file.type,
        name: file.name,
        label: recognizedLabel,
        blobSize: pslRecordedBlob.size
      });
      
      if (file.size === 0) {
        throw new Error('File size is 0 bytes - recording may have failed');
      }
      
      const { video_url } = await api.uploadChatSignVideo(selectedRoom.id, file, 'PSL');
      chatWsRef.current.send({
        type: 'sign_video',
        video_url,
        label: recognizedLabel,
        sign_language: 'PSL',
        auto_translate: false,
      });
      toast.success(`PSL sign sent: "${recognizedLabel}"`);
      cancelSignRecording();
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to send PSL video');
    } finally {
      setIsUploadingSign(false);
    }
  };

  const cancelSignRecording = useCallback(() => {
    stopSignRecording();
    stopPslRecording();
    stopAslVideoRecording();
    if (pslStreamRef.current) {
      pslStreamRef.current.getTracks().forEach((t) => t.stop());
      pslStreamRef.current = null;
    }
    setShowSignRecorder(false);
    setRecognizedText('');
    setDetectedEmotion(null);
    setRecordingTime(0);
    setPslLabel('');
    setPslRecordedBlob(null);
    setPslRecognizedText('');
    setPslLastLetter('');
    setIsPslRecording(false);
    setAslSubMode('live');
    setAslRecordedBlob(null);
    setAslSelectedFile(null);
    setIsAslVideoRecording(false);
  }, [stopSignRecording, stopPslRecording, stopAslVideoRecording]);

  // ── Speech-to-Text ────────────────────────────────────────────────────
  const startSpeechToText = () => {
    const SR = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
    if (!SR) { toast.error('Speech recognition not supported in this browser'); return; }

    const recognition = new SR();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);
    recognition.onresult = (event: any) => {
      const transcript = Array.from(event.results)
        .map((r: any) => r[0].transcript)
        .join('');
      setInputMessage(transcript);
    };
    recognition.onerror = () => { setIsListening(false); toast.error('Speech recognition error'); };
    recognition.start();
  };

  // ── Sign Viewer ───────────────────────────────────────────────────────
  const viewSignTranslation = async (text: string) => {
    // Clear any existing animation
    clearInterval(signAnimRef.current ?? undefined);
    setSignModalText(text);
    setSignFrames([]);
    setCurrentSignFrame(0);
    setIsLoadingSigns(true);
    setShowSignModal(true);

    try {
      const response = await api.textToSign({
        text,
        sign_language: 'ASL' as SignLanguage,
        include_emotion: false,
      });

      const allFrames: string[] = [];
      response.signs.forEach((sign: any) => {
        if (sign.frames?.length) allFrames.push(...sign.frames);
      });

      setSignFrames(allFrames);
      setIsLoadingSigns(false);

      if (allFrames.length > 0) {
        let idx = 0;
        signAnimRef.current = setInterval(() => {
          idx = (idx + 1) % allFrames.length;
          setCurrentSignFrame(idx);
        }, 150);
      }
    } catch {
      toast.error('Failed to load sign translation');
      setShowSignModal(false);
    }
  };

  const closeSignModal = () => {
    clearInterval(signAnimRef.current ?? undefined);
    setShowSignModal(false);
  };

  // ── TTS ───────────────────────────────────────────────────────────────
  const speakMessage = (text: string) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
    } else {
      toast.error('Text-to-speech not supported in this browser');
    }
  };

  // ── Cleanup ───────────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      disconnectChat();
      stopSignRecording();
      clearInterval(signAnimRef.current ?? undefined);
    };
  }, [disconnectChat, stopSignRecording]);

  // ── Auth guard ────────────────────────────────────────────────────────
  if (isGuest || !user) {
    return (
      <div className="min-h-screen gradient-bg flex items-center justify-center">
        <div className="bg-white rounded-xl p-8 max-w-md text-center shadow-lg">
          <FiMessageSquare size={48} className="mx-auto mb-4 text-primary-300" />
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Sign in Required</h2>
          <p className="text-gray-600 mb-6">Please sign in to use the chat feature</p>
          <Link href="/login"><Button>Sign In</Button></Link>
        </div>
      </div>
    );
  }

  const filteredRooms = rooms.filter((r) =>
    (r.name ?? '').toLowerCase().includes(roomSearchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen gradient-bg">
      <div className="flex h-screen">

        {/* ── Sidebar ── */}
        <div className={`${selectedRoom ? 'hidden md:flex' : 'flex'} w-full md:w-96 flex-col bg-white border-r`}>
          <div className="p-4 border-b">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Link href="/dashboard" className="text-gray-600 hover:text-gray-900">
                  <FiArrowLeft size={22} />
                </Link>
                <h1 className="text-xl font-bold text-primary-900">Chats</h1>
              </div>
              <button
                onClick={() => setShowNewChatModal(true)}
                className="p-2 bg-primary-100 rounded-lg text-primary-700 hover:bg-primary-200 transition-colors"
                title="New Chat"
              >
                <FiPlus size={20} />
              </button>
            </div>

            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-primary-100 rounded-full flex items-center justify-center flex-shrink-0">
                <FiUser size={18} className="text-primary-700" />
              </div>
              <span className="font-semibold text-gray-900 truncate">{user.username}</span>
            </div>

            <div className="relative">
              <FiSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={15} />
              <input
                type="text"
                placeholder="Search conversations..."
                value={roomSearchQuery}
                onChange={(e) => setRoomSearchQuery(e.target.value)}
                className="w-full pl-9 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary-200 outline-none text-sm"
              />
            </div>
          </div>

          <div className="flex-1 overflow-y-auto">
            {isLoadingRooms ? (
              <div className="flex items-center justify-center h-32">
                <div className="loading-dots text-primary-600"><span /><span /><span /></div>
              </div>
            ) : filteredRooms.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-32 text-gray-500 text-sm gap-1">
                <p>{roomSearchQuery ? 'No results' : 'No conversations yet'}</p>
                {!roomSearchQuery && (
                  <button
                    onClick={() => setShowNewChatModal(true)}
                    className="text-primary-600 flex items-center gap-1 hover:text-primary-700"
                  >
                    <FiPlus size={14} /> Start a new chat
                  </button>
                )}
              </div>
            ) : (
              filteredRooms.map((room) => (
                <motion.button
                  key={room.id}
                  whileHover={{ backgroundColor: 'rgba(139, 92, 246, 0.05)' }}
                  onClick={() => selectRoom(room)}
                  className={`w-full p-4 flex items-start gap-3 border-b text-left ${
                    selectedRoom?.id === room.id ? 'bg-primary-50 border-l-4 border-l-primary-500' : ''
                  }`}
                >
                  <div className="relative flex-shrink-0">
                    <div className="w-11 h-11 bg-primary-100 rounded-full flex items-center justify-center">
                      <FiUser size={18} className="text-primary-700" />
                    </div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex justify-between items-baseline">
                      <h3 className="font-semibold text-gray-900 truncate text-sm">
                        {room.name || `Room #${room.id}`}
                      </h3>
                      <span className="text-xs text-gray-400 ml-2 flex-shrink-0">
                        {format(new Date(room.created_at), 'HH:mm')}
                      </span>
                    </div>
                    <p className="text-xs text-gray-500 mt-0.5">
                      {room.is_direct ? 'Direct message' : `${room.member_count} members`}
                    </p>
                  </div>
                </motion.button>
              ))
            )}
          </div>
        </div>

        {/* ── Chat area ── */}
        <div className={`${selectedRoom ? 'flex' : 'hidden md:flex'} flex-1 flex-col bg-gray-50`}>
          {selectedRoom ? (
            <>
              {/* Header */}
              <div className="bg-white p-4 border-b flex items-center gap-3 shadow-sm">
                <button
                  onClick={() => { disconnectChat(); setSelectedRoom(null); }}
                  className="md:hidden text-gray-600 hover:text-gray-900"
                >
                  <FiArrowLeft size={22} />
                </button>
                <div className="w-9 h-9 bg-primary-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <FiUser size={16} className="text-primary-700" />
                </div>
                <div className="flex-1">
                  <h2 className="font-semibold text-gray-900 text-sm">
                    {selectedRoom.name || `Room #${selectedRoom.id}`}
                  </h2>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    {wsConnected
                      ? <FiWifi size={11} className="text-green-500" />
                      : <FiWifiOff size={11} className="text-gray-400" />
                    }
                    <p className="text-xs text-gray-500">
                      {wsConnected
                        ? onlineUserIds.length > 0 ? `${onlineUserIds.length} online` : 'Connected'
                        : 'Connecting...'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {isLoadingMessages ? (
                  <div className="flex items-center justify-center h-32">
                    <div className="loading-dots text-primary-600"><span /><span /><span /></div>
                  </div>
                ) : messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-40 text-gray-400">
                    <FiMessageSquare size={36} className="mb-3 opacity-40" />
                    <p className="text-sm">No messages yet. Say hello!</p>
                  </div>
                ) : (
                  messages.map((message) => {
                    const isMine = message.sender_id === user.id;
                    return (
                      <motion.div
                        key={message.id}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.15 }}
                        className={`flex ${isMine ? 'justify-end' : 'justify-start'}`}
                      >
                        <div className={`max-w-[75%] ${isMine ? '' : 'flex gap-2'}`}>
                          {!isMine && (
                            <div className="w-7 h-7 bg-gray-200 rounded-full flex-shrink-0 flex items-center justify-center mt-1">
                              <FiUser size={12} />
                            </div>
                          )}
                          <div>
                            {!isMine && (
                              <p className="text-xs text-gray-500 mb-0.5 ml-0.5">
                                {message.sender_username || 'Unknown'}
                              </p>
                            )}
                            <div className={`chat-bubble ${isMine ? 'chat-bubble-sent ml-auto' : 'chat-bubble-received'}`}>
                              {message.message_type === 'sign_video' && message.video_path ? (
                                <div>
                                  <span className="inline-block text-xs font-semibold px-2 py-0.5 rounded-full bg-primary-100 text-primary-800 mb-2">
                                    {(message.sign_language || 'ASL').toString().toUpperCase()}
                                  </span>
                                  <video
                                    src={api.mediaUrl(message.video_path)}
                                    controls
                                    playsInline
                                    className="max-w-full rounded-lg max-h-64 block"
                                  />
                                  {(message.text_content || message.translated_text) && (
                                    <p className="mt-2 text-sm font-medium">
                                      {message.sign_language === 'PSL'
                                        ? `Label: ${message.text_content || message.translated_text}`
                                        : message.translated_text || message.text_content}
                                    </p>
                                  )}
                                </div>
                              ) : (
                                <div>
                                  <p className="text-sm leading-relaxed">{message.text_content}</p>
                                  
                                  {/* ── Inline Sign Animation (if sign_data present) ── */}
                                  {message.sign_data && (() => {
                                    try {
                                      const signs = JSON.parse(message.sign_data);
                                      // Only show first sign for inline preview (to save space)
                                      const firstSignWithKeypoints = signs.find((s: any) => s.keypoints && s.keypoints.length > 0);
                                      if (firstSignWithKeypoints) {
                                        return (
                                          <div className="mt-2 inline-block">
                                            <div className="bg-[#1a1a2e] rounded-lg overflow-hidden w-28 h-28">
                                              <SkeletonCanvas
                                                frames={firstSignWithKeypoints.keypoints}
                                                word={firstSignWithKeypoints.word}
                                                frameRateMs={50}
                                                width={112}
                                                height={112}
                                              />
                                            </div>
                                            <p className="text-xs text-gray-400 mt-1 text-center">
                                              ASL: {firstSignWithKeypoints.word}
                                              {signs.length > 1 && ` +${signs.length - 1} more`}
                                            </p>
                                          </div>
                                        );
                                      }
                                    } catch (e) {
                                      console.error('Failed to parse sign_data:', e);
                                    }
                                    return null;
                                  })()}
                                </div>
                              )}
                            </div>

                            {/* Meta */}
                            <div className={`flex items-center gap-1.5 mt-1 ${isMine ? 'justify-end' : ''}`}>
                              {message.emotion && <EmotionBadge emotion={message.emotion} size="sm" />}
                              <span className="text-xs text-gray-400">
                                {format(new Date(message.created_at), 'HH:mm')}
                              </span>
                              {message.text_content && message.message_type !== 'sign_video' && (
                                <>
                                  <button
                                    onClick={() => viewSignTranslation(message.text_content!)}
                                    className="p-0.5 text-gray-300 hover:text-primary-600 transition-colors"
                                    title="View in Sign Language"
                                  >
                                    <FiEye size={13} />
                                  </button>
                                  <button
                                    onClick={() => speakMessage(message.text_content!)}
                                    className="p-0.5 text-gray-300 hover:text-primary-600 transition-colors"
                                    title="Listen"
                                  >
                                    <FiVolume2 size={13} />
                                  </button>
                                </>
                              )}
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    );
                  })
                )}

                {/* Typing indicator */}
                <AnimatePresence>
                  {typingUsers.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      className="flex items-center gap-2 text-gray-500 text-xs"
                    >
                      <span className="flex gap-1">
                        {[0, 150, 300].map((delay) => (
                          <span
                            key={delay}
                            className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"
                            style={{ animationDelay: `${delay}ms` }}
                          />
                        ))}
                      </span>
                      <span>{typingUsers.map((u) => u.username).join(', ')} typing...</span>
                    </motion.div>
                  )}
                </AnimatePresence>

                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <div className="bg-white p-3 border-t">
                <div className="flex gap-2 items-center">
                  <button
                    onClick={openSignRecorder}
                    className="p-2.5 bg-primary-100 rounded-lg text-primary-700 hover:bg-primary-200 transition-colors flex-shrink-0"
                    title="Record Sign Language"
                  >
                    <FiCamera size={18} />
                  </button>
                  <button
                    onClick={startSpeechToText}
                    disabled={isListening}
                    className={`p-2.5 rounded-lg transition-colors flex-shrink-0 ${
                      isListening ? 'bg-red-100 text-red-600' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                    title={isListening ? 'Listening...' : 'Voice Input'}
                  >
                    {isListening ? <FiMicOff size={18} /> : <FiMic size={18} />}
                  </button>
                  <input
                    type="text"
                    value={inputMessage}
                    onChange={(e) => handleTypingInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={wsConnected ? 'Type a message...' : 'Connecting...'}
                    disabled={!wsConnected}
                    className="flex-1 px-4 py-2.5 border rounded-lg focus:ring-2 focus:ring-primary-200 outline-none text-sm disabled:bg-gray-50 disabled:text-gray-400"
                  />
                  <Button
                    onClick={sendMessage}
                    disabled={!inputMessage.trim() || isSending || !wsConnected}
                    isLoading={isSending}
                  >
                    <FiSend size={18} />
                  </Button>
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <FiMessageSquare size={56} className="mx-auto mb-4 opacity-30" />
                <p className="text-lg font-medium mb-1">No conversation selected</p>
                <p className="text-sm">Choose a chat or start a new one</p>
                <button
                  onClick={() => setShowNewChatModal(true)}
                  className="mt-4 px-4 py-2 bg-primary-600 text-white rounded-lg text-sm hover:bg-primary-700 transition-colors"
                >
                  + New Chat
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── New Chat Modal ── */}
      <AnimatePresence>
        {showNewChatModal && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
            onClick={() => setShowNewChatModal(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-white rounded-xl shadow-xl w-full max-w-md"
            >
              <div className="p-4 border-b flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900">Start New Chat</h2>
                <button onClick={() => setShowNewChatModal(false)} className="p-1 hover:bg-gray-100 rounded-lg">
                  <FiX size={20} />
                </button>
              </div>
              <div className="p-4">
                <div className="relative mb-4">
                  <FiSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={15} />
                  <input
                    type="text"
                    placeholder="Search users by username or email..."
                    value={userSearchQuery}
                    onChange={(e) => setUserSearchQuery(e.target.value)}
                    className="w-full pl-9 pr-4 py-2.5 border rounded-lg focus:ring-2 focus:ring-primary-200 outline-none text-sm"
                    autoFocus
                  />
                </div>
                <div className="max-h-64 overflow-y-auto">
                  {isSearching ? (
                    <div className="flex items-center justify-center py-8">
                      <div className="loading-dots text-primary-600"><span /><span /><span /></div>
                    </div>
                  ) : searchResults.length === 0 ? (
                    <div className="text-center py-8 text-gray-400 text-sm">
                      {userSearchQuery.length >= 2 ? 'No users found' : 'Type at least 2 characters to search'}
                    </div>
                  ) : (
                    <div className="space-y-1">
                      {searchResults.map((u) => (
                        <button
                          key={u.id}
                          onClick={() => startNewChat(u)}
                          disabled={isCreatingRoom}
                          className="w-full p-3 flex items-center gap-3 hover:bg-gray-50 rounded-lg transition-colors text-left"
                        >
                          <div className="w-10 h-10 bg-primary-100 rounded-full flex items-center justify-center flex-shrink-0">
                            <FiUser size={16} className="text-primary-700" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium text-gray-900 text-sm">{u.username}</p>
                            <p className="text-xs text-gray-500 truncate">{u.email}</p>
                          </div>
                          <span className="text-xs text-gray-400 flex-shrink-0 capitalize">
                            {u.role.toLowerCase()}
                          </span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Sign Recorder Modal ── */}
      <AnimatePresence>
        {showSignRecorder && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white rounded-xl shadow-xl w-full max-w-lg"
            >
              <div className="p-4 border-b flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900">Send Sign Language</h2>
                <button onClick={cancelSignRecording} className="p-1 hover:bg-gray-100 rounded-lg">
                  <FiX size={20} />
                </button>
              </div>
              <motion.div className="p-4 space-y-4">
                <motion.div className="flex rounded-lg bg-gray-100 p-1 gap-1">
                  <button type="button" onClick={() => { stopSignRecording(); stopPslRecording(); stopAslVideoRecording(); setSignChatMode('asl'); }} className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${signChatMode === 'asl' ? 'bg-white shadow text-primary-700' : 'text-gray-600'}`}>ASL</button>
                  <button type="button" onClick={() => { stopSignRecording(); stopPslRecording(); stopAslVideoRecording(); setSignChatMode('psl'); }} className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${signChatMode === 'psl' ? 'bg-white shadow text-primary-700' : 'text-gray-600'}`}>PSL</button>
                </motion.div>
                {signChatMode === 'asl' && (
                  <div className="flex rounded-lg border border-gray-200 p-0.5 gap-0.5">
                    <button type="button" onClick={() => { stopAslVideoRecording(); setAslSubMode('live'); setAslRecordedBlob(null); setAslSelectedFile(null); }} className={`flex-1 py-1.5 text-xs font-medium rounded ${aslSubMode === 'live' ? 'bg-primary-50 text-primary-700' : 'text-gray-500'}`}>Live camera</button>
                    <button type="button" onClick={() => { stopSignRecording(); setAslSubMode('video'); }} className={`flex-1 py-1.5 text-xs font-medium rounded ${aslSubMode === 'video' ? 'bg-primary-50 text-primary-700' : 'text-gray-500'}`}>Record / upload video</button>
                  </div>
                )}
                <p className="text-xs text-gray-500">
                  {signChatMode === 'asl' && aslSubMode === 'live' && 'ASL live: sign to the camera, then send recognized text.'}
                  {signChatMode === 'asl' && aslSubMode === 'video' && 'ASL video: record or upload a clip; AI translates signs when you send.'}
                  {signChatMode === 'psl' && 'PSL: record a clip, add a label, and send the video.'}
                </p>
                <motion.div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-video">
                  {signChatMode === 'asl' && aslSubMode === 'live' ? (
                    <Webcam ref={webcamRef} audio={false} screenshotFormat="image/jpeg" videoConstraints={{ facingMode: 'user', width: 640, height: 480 }} className="w-full h-full object-cover" mirrored />
                  ) : signChatMode === 'asl' && aslSubMode === 'video' && aslFilePreviewUrl ? (
                    <video src={aslFilePreviewUrl} controls className="w-full h-full object-contain bg-black" />
                  ) : (
                    <video ref={pslVideoRef} autoPlay playsInline muted className="w-full h-full object-cover" style={{ transform: 'scaleX(-1)' }} />
                  )}
                  {(isRecording || isPslRecording || isAslVideoRecording) && (
                    <motion.div className="absolute top-3 left-3 flex items-center gap-2 bg-red-600/90 text-white px-3 py-1 rounded-full text-sm">
                      <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                      REC {Math.floor(recordingTime / 60)}:{String(recordingTime % 60).padStart(2, '0')}
                    </motion.div>
                  )}
                </motion.div>
                {signChatMode === 'asl' && aslSubMode === 'live' ? (
                  <>
                    <motion.div className="bg-gray-50 rounded-lg p-3 min-h-[60px]">
                      <p className="text-xs text-gray-500 mb-1">Recognized text (ASL)</p>
                      <p className="text-gray-900 text-sm">{recognizedText || (isRecording ? 'Translating signs…' : 'Press Start live')}</p>
                      {detectedEmotion && <motion.div className="mt-2"><EmotionBadge emotion={detectedEmotion} size="sm" /></motion.div>}
                    </motion.div>
                    <motion.div className="flex gap-3">
                      {!isRecording ? (
                        <Button onClick={startSignRecording} className="flex-1" leftIcon={<FiCamera size={16} />}>Start live</Button>
                      ) : (
                        <Button onClick={stopSignRecording} variant="danger" className="flex-1" leftIcon={<FiStopCircle size={16} />}>Stop</Button>
                      )}
                      <Button onClick={sendSignMessage} disabled={!recognizedText.trim() || !chatWsRef.current?.isConnected()} className="flex-1" leftIcon={<FiSend size={16} />}>Send text</Button>
                    </motion.div>
                  </>
                ) : signChatMode === 'asl' && aslSubMode === 'video' ? (
                  <>
                    {(aslRecordedBlob || aslSelectedFile) && !isAslVideoRecording && (
                      <p className="text-xs text-green-700">
                        {aslSelectedFile
                          ? `Selected: ${aslSelectedFile.name}`
                          : `Recording ready (${(aslRecordedBlob!.size / 1024).toFixed(0)} KB)`}
                      </p>
                    )}
                    <input ref={aslVideoFileInputRef} type="file" accept="video/mp4,video/webm,video/avi,video/mov" className="hidden" onChange={handleAslVideoFile} />
                    <motion.div className="flex flex-wrap gap-2">
                      {!isAslVideoRecording ? (
                        <Button onClick={startAslVideoRecording} className="flex-1 min-w-[100px]" leftIcon={<FiCamera size={16} />} disabled={!!aslRecordedBlob || !!aslSelectedFile}>Record</Button>
                      ) : (
                        <Button onClick={stopAslVideoRecording} variant="danger" className="flex-1" leftIcon={<FiStopCircle size={16} />}>Stop</Button>
                      )}
                      <Button variant="secondary" onClick={() => aslVideoFileInputRef.current?.click()} leftIcon={<FiUpload size={16} />} disabled={isAslVideoRecording}>Upload</Button>
                      {(aslRecordedBlob || aslSelectedFile) && !isAslVideoRecording && (
                        <Button variant="secondary" onClick={() => { setAslRecordedBlob(null); setAslSelectedFile(null); setRecordingTime(0); }}>Clear</Button>
                      )}
                      <Button onClick={sendAslSignVideo} disabled={(!aslRecordedBlob && !aslSelectedFile) || !chatWsRef.current?.isConnected() || isUploadingSign} isLoading={isUploadingSign} className="flex-1 min-w-[100px]" leftIcon={<FiVideo size={16} />}>Send video</Button>
                    </motion.div>
                  </>
                ) : (
                  <>
                    {/* Live PSL Recognition Display */}
                    <motion.div className="bg-gray-50 rounded-lg p-3 min-h-[80px]">
                      <p className="text-xs text-gray-500 mb-1">Recognized letters (PSL - Live)</p>
                      <div className="flex items-center gap-2">
                        <p className="text-gray-900 text-lg font-medium" dir="auto">
                          {pslRecognizedText || (isPslRecording ? 'Recording...' : 'Press Record to start')}
                        </p>
                        {isPslRecording && pslLastLetter && (
                          <motion.span
                            initial={{ scale: 0.8, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            className="text-primary-600 text-sm font-bold bg-primary-50 px-2 py-1 rounded"
                          >
                            {pslLastLetter}
                          </motion.span>
                        )}
                      </div>
                      {pslRecognizedText && !isPslRecording && (
                        <p className="text-xs text-green-600 mt-1">
                          Ready to send: {pslRecognizedText.length} letter(s)
                        </p>
                      )}
                      {isPslRecording && (
                        <p className="text-xs text-amber-600 mt-1">
                          Recording... {recordingTime}/10 seconds
                        </p>
                      )}
                    </motion.div>
                    
                    {pslRecordedBlob && !isPslRecording && (
                      <p className="text-xs text-green-700">
                        Video ready ({(pslRecordedBlob.size / 1024).toFixed(0)} KB)
                      </p>
                    )}
                    
                    <motion.div className="flex flex-wrap gap-3">
                      {!isPslRecording ? (
                        <Button 
                          onClick={startPslRecording} 
                          className="flex-1 min-w-[120px]" 
                          leftIcon={<FiCamera size={16} />} 
                          disabled={!!pslRecordedBlob}
                        >
                          {pslRecordedBlob ? 'Recorded' : 'Record (10s)'}
                        </Button>
                      ) : (
                        <Button 
                          onClick={stopPslRecording} 
                          variant="danger" 
                          className="flex-1" 
                          leftIcon={<FiStopCircle size={16} />}
                        >
                          Stop
                        </Button>
                      )}
                      {pslRecordedBlob && !isPslRecording && (
                        <Button 
                          variant="secondary" 
                          onClick={() => { 
                            setPslRecordedBlob(null); 
                            setPslRecognizedText('');
                            setPslLastLetter('');
                            setRecordingTime(0); 
                          }}
                        >
                          Re-record
                        </Button>
                      )}
                      <Button 
                        onClick={sendPslSignVideo} 
                        disabled={!pslRecordedBlob || !chatWsRef.current?.isConnected() || isUploadingSign} 
                        isLoading={isUploadingSign} 
                        className="flex-1 min-w-[120px]" 
                        leftIcon={<FiSend size={16} />}
                      >
                        Send video
                      </Button>
                    </motion.div>
                  </>
                )}
              </motion.div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Sign Viewer Modal ── */}
      <AnimatePresence>
        {showSignModal && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
            onClick={closeSignModal}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-white rounded-xl shadow-xl w-full max-w-lg"
            >
              <div className="p-4 border-b flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900">Sign Language Translation</h2>
                <button onClick={closeSignModal} className="p-1 hover:bg-gray-100 rounded-lg">
                  <FiX size={20} />
                </button>
              </div>
              <div className="p-4 space-y-4">
                <div className="bg-gray-100 rounded-lg aspect-video flex items-center justify-center overflow-hidden">
                  {isLoadingSigns ? (
                    <div className="loading-dots text-primary-600"><span /><span /><span /></div>
                  ) : signFrames.length > 0 ? (
                    <img
                      src={`data:image/png;base64,${signFrames[currentSignFrame]}`}
                      alt="Sign language"
                      className="max-h-full max-w-full object-contain"
                    />
                  ) : (
                    <p className="text-gray-500 text-sm">No sign animation available</p>
                  )}
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">Original text:</p>
                  <p className="text-gray-900 text-sm">{signModalText}</p>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
