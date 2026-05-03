'use client';

import { Emotion, EMOTION_EMOJIS } from '@/types';

interface EmotionBadgeProps {
  emotion: Emotion;
  size?: 'sm' | 'md' | 'lg';
}

const emotionColors: Record<Emotion, string> = {
  happy: 'bg-yellow-100 text-yellow-800',
  sad: 'bg-blue-100 text-blue-800',
  angry: 'bg-red-100 text-red-800',
  surprised: 'bg-purple-100 text-purple-800',
  fearful: 'bg-indigo-100 text-indigo-800',
  disgusted: 'bg-green-100 text-green-800',
  neutral: 'bg-gray-100 text-gray-800',
  unknown: 'bg-gray-100 text-gray-500',
};

export default function EmotionBadge({ emotion, size = 'md' }: EmotionBadgeProps) {
  const sizes = {
    sm: 'text-xs px-2 py-1',
    md: 'text-sm px-3 py-1.5',
    lg: 'text-base px-4 py-2',
  };

  const emojiSizes = {
    sm: 'text-sm',
    md: 'text-lg',
    lg: 'text-xl',
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-medium ${emotionColors[emotion]} ${sizes[size]}`}
    >
      <span className={emojiSizes[size]}>{EMOTION_EMOJIS[emotion]}</span>
      <span className="capitalize">{emotion}</span>
    </span>
  );
}
