export type UserRole = 'hearing' | 'deaf' | 'admin';
export type SignLanguage = 'ASL' | 'PSL';
export type TranslationMode = 'text_to_sign' | 'sign_to_text';
export type Emotion = 'happy' | 'sad' | 'angry' | 'surprised' | 'fearful' | 'disgusted' | 'neutral' | 'unknown';
export type MessageType = 'text' | 'sign_video' | 'translation' | 'system';

export interface User {
  id: number;
  username: string;
  email: string;
  full_name: string | null;
  role: UserRole;
  preferred_sign_language: SignLanguage;
  is_active: boolean;
  is_verified: boolean;
  profile_picture_url: string | null;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  full_name?: string;
  role: UserRole;
  preferred_sign_language: SignLanguage;
}

export interface UserUpdateRequest {
  full_name?: string;
  preferred_sign_language?: SignLanguage;
  profile_picture_url?: string;
}

export interface TextToSignRequest {
  text: string;
  sign_language: SignLanguage;
  include_emotion: boolean;
}

export interface SignData {
  word: string;
  /** Raw keypoint frames for canvas rendering: shape (N_frames, 75, 2) — X,Y normalized [0,1] */
  keypoints: number[][][] | null;
  frames: string[];
  gif_url: string | null;
  fingerspelled: boolean;
}

export interface EmotionAnalysis {
  emotion: Emotion;
  sentiment_label: string;
  sentiment_score: number;
  emotion_scores: Record<string, number>;
}

export interface TextToSignResponse {
  translation_id: number;
  input_text: string;
  words: string[];
  signs: SignData[];
  total_duration_ms: number;
  sign_language: string;
  fingerspelled_words: string[];
  emotion_analysis?: EmotionAnalysis;
  processing_time_ms: number;
}

export interface SignToTextResponse {
  translation_id: number;
  recognized_text: string;
  glosses: string[];
  confidence: number;
  sign_language: string;
  frame_count: number;
  processing_time_ms: number;
  emotion_from_video?: EmotionAnalysis;
}

export interface ChatRoom {
  id: number;
  name: string | null;
  is_direct: boolean;
  created_at: string;
  member_count: number;
}

export interface ChatMessage {
  id: number;
  room_id: number;
  sender_id: number | null;
  sender_username: string | null;
  message_type: MessageType;
  text_content: string | null;
  translated_text: string | null;
  sign_data: string | null;
  video_path: string | null;
  emotion: Emotion | null;
  sentiment_label: string | null;
  is_read: boolean;
  created_at: string;
}

export interface CreateChatRoomRequest {
  name?: string;
  member_ids: number[];
  is_direct: boolean;
}

export interface TranslationHistoryItem {
  id: number;
  user_id: number | null;
  mode: TranslationMode;
  sign_language: SignLanguage;
  input_text: string | null;
  output_sign_data: string | null;
  input_video_path: string | null;
  output_text: string | null;
  detected_emotion: Emotion | null;
  sentiment_score: number | null;
  sentiment_label: string | null;
  processing_time_ms: number;
  confidence_score: number | null;
  created_at: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface SpeechToTextResponse {
  translation_id: number;
  recognized_text: string;
  language: string;
  confidence: number;
  duration_seconds: number;
  processing_time_ms: number;
  sentiment?: EmotionAnalysis;
}

export interface TextToSpeechResponse {
  audio_url: string;
  format: string;
  duration_seconds: number;
  text_length: number;
  voice: string;
  language: string;
  processing_time_ms: number;
}

export const EMOTION_EMOJIS: Record<Emotion, string> = {
  happy: '😊',
  sad: '😢',
  angry: '😠',
  surprised: '😮',
  fearful: '😨',
  disgusted: '🤢',
  neutral: '😐',
  unknown: '❓',
};

// WLASL Model Integration Types
export type ValidationStatus = 'clear' | 'acceptable' | 'cluttered' | 'visible' | 'partial' | 'not_visible' | 'optimal' | 'out_of_range' | 'ready' | 'not_ready';

export interface BackgroundValidation {
  status: 'clear' | 'acceptable' | 'cluttered';
  score: number;
  message: string;
}

export interface BodyValidation {
  status: 'visible' | 'partial' | 'not_visible';
  landmarks_detected: {
    left_hand: boolean;
    right_hand: boolean;
    arms: boolean;
    torso: boolean;
  };
  message: string;
}

export interface DistanceValidation {
  status: 'optimal' | 'acceptable' | 'out_of_range';
  shoulder_width_px: number;
  message: string;
}

export interface ValidationState {
  background: BackgroundValidation | null;
  body: BodyValidation | null;
  distance: DistanceValidation | null;
  overall: 'ready' | 'not_ready';
}

export interface PredictionResult {
  recognized_text: string;
  glosses: string[];
  confidence: number;
  top5_predictions: [number, number][];
  frame_count: number;
  processing_time_ms: number;
  sign_language: string;
}
