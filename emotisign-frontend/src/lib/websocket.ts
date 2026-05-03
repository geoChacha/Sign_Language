import Cookies from 'js-cookie';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

export type WebSocketEventHandler = (data: any) => void;

export class EmotiSignWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private handlers: Map<string, WebSocketEventHandler[]> = new Map();
  private isConnecting = false;
  private shouldReconnect = true;

  constructor(path: string, includeAuth: boolean = true) {
    const token = Cookies.get('access_token');
    if (includeAuth && token) {
      // Handle paths that already have query params (e.g. ?language=en)
      const separator = path.includes('?') ? '&' : '?';
      this.url = `${WS_URL}${path}${separator}token=${token}`;
    } else {
      this.url = `${WS_URL}${path}`;
    }
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
        resolve();
        return;
      }

      this.isConnecting = true;
      this.shouldReconnect = true;

      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.emit('connected', {});
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            // Backend sends: { event: "message"|"typing"|"partial"|..., data: {...} }
            // OR for translate WS: { type: "partial"|"result"|..., ... }
            const eventName: string = message.event || message.type || '';

            // Always emit 'message' with the FULL parsed object.
            // Handlers that listen on 'message' inspect fullMsg.event + fullMsg.data themselves.
            this.emit('message', message);

            // Also emit the specific event name with just the data payload.
            // SKIP emitting if eventName === 'message' to avoid a second call with wrong shape.
            if (eventName && eventName !== 'message') {
              this.emit(eventName, message.data ?? message);
            }
          } catch (e) {
            console.error('Failed to parse WebSocket message:', e);
          }
        };

        this.ws.onclose = (event) => {
          this.isConnecting = false;
          this.emit('disconnected', { code: event.code, reason: event.reason });

          if (this.shouldReconnect && !event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          this.isConnecting = false;
          this.emit('error', error);
          reject(error);
        };
      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  private scheduleReconnect() {
    this.reconnectAttempts++;
    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);
    setTimeout(() => {
      if (this.shouldReconnect) {
        this.connect().catch(console.error);
      }
    }, delay);
  }

  send(data: object): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket not connected, dropping message:', data);
    }
  }

  on(event: string, handler: WebSocketEventHandler): void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, []);
    }
    this.handlers.get(event)!.push(handler);
  }

  off(event: string, handler: WebSocketEventHandler): void {
    const eventHandlers = this.handlers.get(event);
    if (eventHandlers) {
      const index = eventHandlers.indexOf(handler);
      if (index !== -1) {
        eventHandlers.splice(index, 1);
      }
    }
  }

  private emit(event: string, data: any): void {
    const eventHandlers = this.handlers.get(event);
    if (eventHandlers) {
      eventHandlers.forEach((handler) => {
        try {
          handler(data);
        } catch (e) {
          console.error(`Error in WebSocket handler for '${event}':`, e);
        }
      });
    }
  }

  disconnect(): void {
    this.shouldReconnect = false;
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.handlers.clear();
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

export function createSignToTextWS(): EmotiSignWebSocket {
  return new EmotiSignWebSocket('/ws/translate/sign-to-text', true);
}

export function createTextToSignWS(): EmotiSignWebSocket {
  return new EmotiSignWebSocket('/ws/translate/text-to-sign', true);
}

export function createChatWS(roomId: number): EmotiSignWebSocket {
  return new EmotiSignWebSocket(`/api/chat/ws/chat/${roomId}`, true);
}

export function createLiveSTTWS(language: string = 'en'): EmotiSignWebSocket {
  // path already has '?' so constructor will append '&token=...'
  return new EmotiSignWebSocket(`/ws/speech/live-stt?language=${language}`, true);
}
