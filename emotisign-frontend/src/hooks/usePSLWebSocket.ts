'use client';

import { useRef, useState, useCallback, useEffect } from 'react';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface PSLPredictionResult {
  predicted_label: string;
  urdu_text: string;
  confidence: number;
  landmarks: number[][];  // [[x, y], ...] 21 landmarks in pixel coords
  emotion?: 'happy' | 'sad' | 'neutral';
  emotion_emoji?: string;
  emotion_scores?: Record<string, number>;
  face_detected?: boolean;
}

export interface PSLSessionStats {
  total_frames: number;
  frames_with_hands: number;
  predictions_made: number;
  average_confidence: number;
  session_duration_seconds: number;
}

export interface UsePSLWebSocketOptions {
  onConnected?: () => void;
  onResult: (result: PSLPredictionResult) => void;
  onNoHand: () => void;
  onLowConfidence?: (_data: {
    predicted_label: string;
    confidence: number;
    emotion?: string;
    emotion_emoji?: string;
  }) => void;
  onError?: (_message: string) => void;
  onSessionEnd?: (_stats: PSLSessionStats) => void;
}

export type PSLConnectionStatus = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error';

// ── Hook ──────────────────────────────────────────────────────────────────────

export function usePSLWebSocket(options: UsePSLWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectCountRef = useRef(0);
  const intentionalStopRef = useRef(false);
  const optionsRef = useRef(options);

  const [connectionStatus, setConnectionStatus] = useState<PSLConnectionStatus>('idle');
  const [reconnectCount, setReconnectCount] = useState(0);

  // Keep options ref current without re-creating callbacks
  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  const stopPing = useCallback(() => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
  }, []);

  const startPing = useCallback((ws: WebSocket) => {
    stopPing();
    pingIntervalRef.current = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 10_000);
  }, [stopPing]);

  const connect = useCallback(() => {
    // Derive WS URL from API URL (replace http/https with ws/wss)
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const wsBase = apiUrl.replace(/^http/, 'ws');
    const wsUrl = `${wsBase}/ws/translate/psl-live`;

    intentionalStopRef.current = false;
    setConnectionStatus('connecting');

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      reconnectCountRef.current = 0;
      setReconnectCount(0);
      setConnectionStatus('connected');
      startPing(ws);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        const { event: evtName, data, message } = msg;

        switch (evtName) {
          case 'connected':
            optionsRef.current.onConnected?.();
            break;
          case 'result':
            optionsRef.current.onResult(data as PSLPredictionResult);
            break;
          case 'no_hand':
            optionsRef.current.onNoHand();
            break;
          case 'low_confidence':
            optionsRef.current.onLowConfidence?.(data);
            break;
          case 'error':
            optionsRef.current.onError?.(message || 'Unknown error');
            break;
          case 'session_end':
            optionsRef.current.onSessionEnd?.(data as PSLSessionStats);
            break;
          case 'pong':
            // keepalive — no action needed
            break;
          default:
            break;
        }
      } catch (e) {
        console.error('[PSL WS] Failed to parse message:', e);
      }
    };

    ws.onerror = () => {
      setConnectionStatus('error');
      optionsRef.current.onError?.('WebSocket connection error');
    };

    ws.onclose = () => {
      stopPing();
      setConnectionStatus('disconnected');

      // Auto-reconnect up to 3 times if not intentionally stopped
      if (!intentionalStopRef.current && reconnectCountRef.current < 3) {
        reconnectCountRef.current += 1;
        setReconnectCount(reconnectCountRef.current);
        setTimeout(() => {
          if (!intentionalStopRef.current) {
            connect();
          }
        }, 2000);
      }
    };
  }, [startPing, stopPing]);

  const disconnect = useCallback(() => {
    intentionalStopRef.current = true;
    stopPing();
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'stop' }));
      }
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnectionStatus('idle');
  }, [stopPing]);

  const sendFrame = useCallback((base64: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'frame',
        data: base64,
        mime: 'image/jpeg',
      }));
    }
  }, []);

  const updateConfig = useCallback((confidenceThreshold: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'config',
        confidence_threshold: confidenceThreshold,
      }));
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      intentionalStopRef.current = true;
      stopPing();
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [stopPing]);

  return {
    connectionStatus,
    reconnectCount,
    connect,
    disconnect,
    sendFrame,
    updateConfig,
  };
}
