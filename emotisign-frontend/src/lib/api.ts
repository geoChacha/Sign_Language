import axios, { AxiosInstance, AxiosError } from 'axios';
import Cookies from 'js-cookie';
import {
  User,
  TokenResponse,
  RegisterRequest,
  UserUpdateRequest,
  TextToSignRequest,
  TextToSignResponse,
  SignToTextResponse,
  ChatRoom,
  ChatMessage,
  CreateChatRoomRequest,
  TranslationHistoryItem,
  PaginatedResponse,
  SpeechToTextResponse,
  TextToSpeechResponse,
  SignLanguage,
} from '@/types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.client.interceptors.request.use((config) => {
      const token = Cookies.get('access_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          Cookies.remove('access_token');
          // Don't redirect here - let components handle navigation
        }
        return Promise.reject(error);
      }
    );
  }

  async register(data: RegisterRequest): Promise<User> {
    const response = await this.client.post<User>('/api/auth/register', data);
    return response.data;
  }

  async login(username: string, password: string): Promise<TokenResponse> {
    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);

    const response = await this.client.post<TokenResponse>('/api/auth/login', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });

    Cookies.set('access_token', response.data.access_token, {
      expires: response.data.expires_in / 86400,
    });

    return response.data;
  }

  async logout(): Promise<void> {
    try {
      await this.client.post('/api/auth/logout');
    } finally {
      Cookies.remove('access_token');
    }
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get<User>('/api/auth/me');
    return response.data;
  }

  async updateUser(data: UserUpdateRequest): Promise<User> {
    const response = await this.client.patch<User>('/api/auth/me', data);
    return response.data;
  }

  async searchUsers(query?: string): Promise<User[]> {
    const params = query ? `?q=${encodeURIComponent(query)}` : '';
    const response = await this.client.get<User[]>(`/api/auth/users${params}`);
    return response.data;
  }

  async textToSign(data: TextToSignRequest): Promise<TextToSignResponse> {
    const response = await this.client.post<TextToSignResponse>('/api/translate/text-to-sign', data);
    return response.data;
  }

  async signToText(videoFile: File, signLanguage: SignLanguage): Promise<SignToTextResponse> {
    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('sign_language', signLanguage);

    const response = await this.client.post<SignToTextResponse>('/api/translate/sign-to-text', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  async getTranslationHistory(
    page: number = 1,
    pageSize: number = 20,
    mode?: string
  ): Promise<PaginatedResponse<TranslationHistoryItem>> {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (mode) params.append('mode', mode);

    const response = await this.client.get<PaginatedResponse<TranslationHistoryItem>>(
      `/api/translate/history?${params}`
    );
    return response.data;
  }

  async deleteTranslation(id: number): Promise<void> {
    await this.client.delete(`/api/translate/history/${id}`);
  }

  async createChatRoom(data: CreateChatRoomRequest): Promise<ChatRoom> {
    const response = await this.client.post<ChatRoom>('/api/chat/rooms', data);
    return response.data;
  }

  async getChatRooms(): Promise<ChatRoom[]> {
    const response = await this.client.get<ChatRoom[]>('/api/chat/rooms');
    return response.data;
  }

  async getChatMessages(
    roomId: number,
    page: number = 1,
    pageSize: number = 50
  ): Promise<PaginatedResponse<ChatMessage>> {
    const response = await this.client.get<PaginatedResponse<ChatMessage>>(
      `/api/chat/rooms/${roomId}/messages?page=${page}&page_size=${pageSize}`
    );
    return response.data;
  }

  async markMessagesAsRead(roomId: number): Promise<void> {
    await this.client.patch(`/api/chat/rooms/${roomId}/read`);
  }

  async uploadChatSignVideo(roomId: number, videoFile: File): Promise<{ video_url: string }> {
    const formData = new FormData();
    formData.append('video', videoFile);
    const response = await this.client.post<{ video_url: string }>(
      `/api/chat/rooms/${roomId}/sign-video`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    );
    return response.data;
  }

  /** Resolve chat/upload paths to a full URL for <video src>. */
  mediaUrl(path: string | null | undefined): string {
    if (!path) return '';
    if (path.startsWith('http://') || path.startsWith('https://')) return path;
    const base = API_URL.replace(/\/$/, '');
    return path.startsWith('/') ? `${base}${path}` : `${base}/${path}`;
  }

  async speechToText(
    audioFile: File,
    language: string = 'en',
    includeSentiment: boolean = true
  ): Promise<SpeechToTextResponse> {
    const formData = new FormData();
    formData.append('audio', audioFile);
    formData.append('language', language);
    formData.append('include_sentiment', String(includeSentiment));

    const response = await this.client.post<SpeechToTextResponse>('/api/speech/speech-to-text', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  async textToSpeech(
    text: string,
    language: string = 'en',
    voice: string = 'default'
  ): Promise<TextToSpeechResponse> {
    const formData = new FormData();
    formData.append('text', text);
    formData.append('language', language);
    formData.append('voice', voice);

    const response = await this.client.post<TextToSpeechResponse>('/api/speech/text-to-speech', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  getAudioUrl(filename: string): string {
    return `${API_URL}/api/speech/audio/${filename}`;
  }

  getStaticUrl(path: string): string {
    return `${API_URL}/static/${path}`;
  }
}

export const api = new ApiClient();
export default api;
