'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { FiMessageSquare, FiSettings } from 'react-icons/fi';

const connectionOptions = [
  {
    id: 'sign-to-text',
    title: 'Sign to Text/Speech',
    description: 'Translate sign language into spoken or written words in real-time.',
    href: '/translate/sign-to-text',
    icon: (
      <svg viewBox="0 0 80 60" className="w-20 h-16">
        <g fill="#1e40af" stroke="#1e40af" strokeWidth="2">
          <path d="M10 30 Q20 20 25 30 Q30 40 35 30" fill="none" />
          <circle cx="15" cy="25" r="3" />
          <circle cx="25" cy="20" r="3" />
          <path d="M40 25 L50 25" />
          <circle cx="55" cy="25" r="2" />
          <rect x="60" y="15" width="15" height="25" rx="3" fill="none" />
          <path d="M63 25 L72 25 M63 30 L72 30" strokeWidth="1.5" />
        </g>
      </svg>
    ),
    color: 'blue',
  },
  {
    id: 'text-to-sign',
    title: 'Text/Speech to Sign',
    description: 'Convert your speech or text into clear, animated sign language.',
    href: '/translate/text-to-sign',
    icon: (
      <svg viewBox="0 0 80 60" className="w-20 h-16">
        <g fill="#1e40af" stroke="#1e40af" strokeWidth="2">
          <rect x="5" y="15" width="20" height="25" rx="3" fill="none" />
          <path d="M8 22 L22 22 M8 27 L18 27" strokeWidth="1.5" />
          <circle cx="18" cy="18" r="4" fill="none" />
          <path d="M30 25 L40 25" />
          <circle cx="45" cy="25" r="2" />
          <circle cx="55" cy="20" r="8" fill="none" />
          <path d="M53 18 L57 18 M55 20 L55 24" strokeWidth="1.5" />
          <path d="M60 30 Q65 35 70 30" fill="none" />
        </g>
      </svg>
    ),
    color: 'blue',
  },
  {
    id: 'remote-chat',
    title: 'Two-Way Remote Chat',
    description: 'Send sign language videos or text messages to connect with others online.',
    href: '/chat',
    icon: (
      <svg viewBox="0 0 80 60" className="w-20 h-16">
        <g fill="none" stroke="#047857" strokeWidth="2">
          <rect x="10" y="10" width="25" height="20" rx="3" />
          <path d="M15 18 L30 18 M15 23 L27 23" strokeWidth="1.5" />
          <path d="M38 20 L42 20" />
          <rect x="45" y="10" width="25" height="20" rx="3" />
          <circle cx="57" cy="17" r="3" />
          <path d="M53 23 L62 23" strokeWidth="1.5" />
          <path d="M30 35 L50 35 L40 45 Z" fill="#047857" />
        </g>
      </svg>
    ),
    color: 'green',
  },
];

export default function DashboardPage() {
  return (
    <div className="min-h-[calc(100vh-64px)] flex flex-col items-center justify-center px-4 py-12">
      <div className="absolute top-20 right-8 flex gap-4">
        <Link
          href="/chat"
          className="flex flex-col items-center text-white/80 hover:text-white transition-colors"
        >
          <FiMessageSquare size={24} />
          <span className="text-xs mt-1">Chat</span>
        </Link>
        <Link
          href="/settings"
          className="flex flex-col items-center text-white/80 hover:text-white transition-colors"
        >
          <FiSettings size={24} />
          <span className="text-xs mt-1">Settings</span>
        </Link>
      </div>

      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-3xl md:text-4xl font-bold text-white text-center mb-12"
      >
        How do you want to Connect?
      </motion.h1>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl w-full">
        {connectionOptions.map((option, index) => (
          <motion.div
            key={option.id}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Link href={option.href}>
              <div className="bg-white rounded-2xl p-6 shadow-xl hover:shadow-2xl transition-all duration-300 hover:-translate-y-1 h-full">
                <div className="flex justify-center mb-4">{option.icon}</div>
                <h2
                  className={`text-lg font-bold text-center mb-2 ${
                    option.color === 'green' ? 'text-emerald-700' : 'text-blue-700'
                  }`}
                >
                  {option.title}
                </h2>
                <p className="text-gray-600 text-sm text-center">{option.description}</p>
              </div>
            </Link>
          </motion.div>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-12 text-white/60 text-sm text-center"
      >
        Select an option to get started with EmotiSign
      </motion.div>
    </div>
  );
}
