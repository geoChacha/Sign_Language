'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { format } from 'date-fns';
import toast from 'react-hot-toast';
import { FiArrowLeft, FiTrash2, FiClock, FiFilter } from 'react-icons/fi';
import Button from '@/components/ui/Button';
import Select from '@/components/ui/Select';
import EmotionBadge from '@/components/ui/EmotionBadge';
import api from '@/lib/api';
import { useAuthStore } from '@/store/auth';
import { TranslationHistoryItem, PaginatedResponse, Emotion } from '@/types';

export default function HistoryPage() {
  const { isGuest } = useAuthStore();
  const [history, setHistory] = useState<TranslationHistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState<string>('all');
  const [pagination, setPagination] = useState({
    page: 1,
    total: 0,
    totalPages: 1,
  });

  const loadHistory = async (page: number = 1) => {
    setIsLoading(true);
    try {
      const mode = filter === 'all' ? undefined : filter;
      const data = await api.getTranslationHistory(page, 20, mode);
      setHistory(data.items);
      setPagination({
        page: data.page,
        total: data.total,
        totalPages: data.total_pages,
      });
    } catch (error) {
      toast.error('Failed to load history');
    } finally {
      setIsLoading(false);
    }
  };

  const deleteItem = async (id: number) => {
    try {
      await api.deleteTranslation(id);
      setHistory((prev) => prev.filter((item) => item.id !== id));
      toast.success('Translation deleted');
    } catch {
      toast.error('Failed to delete translation');
    }
  };

  useEffect(() => {
    if (!isGuest) {
      loadHistory();
    }
  }, [isGuest, filter]);

  if (isGuest) {
    return (
      <div className="min-h-screen gradient-bg flex items-center justify-center">
        <div className="bg-white rounded-xl p-8 max-w-md text-center shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Sign in Required</h2>
          <p className="text-gray-600 mb-6">
            Please sign in to view your translation history
          </p>
          <Link href="/login">
            <Button>Sign In</Button>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto py-8 px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <Link
                href="/dashboard"
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <FiArrowLeft size={24} />
              </Link>
              <h1 className="text-2xl font-bold text-gray-900">Translation History</h1>
            </div>

            <div className="flex items-center gap-2">
              <FiFilter size={18} className="text-gray-500" />
              <Select
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                options={[
                  { value: 'all', label: 'All' },
                  { value: 'text_to_sign', label: 'Text to Sign' },
                  { value: 'sign_to_text', label: 'Sign to Text' },
                ]}
                className="w-40"
              />
            </div>
          </div>

          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="loading-dots text-primary-600">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          ) : history.length === 0 ? (
            <div className="bg-white rounded-xl shadow-sm p-12 text-center">
              <FiClock size={48} className="mx-auto mb-4 text-gray-300" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                No translations yet
              </h3>
              <p className="text-gray-500 mb-6">
                Start translating to see your history here
              </p>
              <Link href="/dashboard">
                <Button>Start Translating</Button>
              </Link>
            </div>
          ) : (
            <div className="space-y-4">
              {history.map((item, index) => (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="bg-white rounded-xl shadow-sm p-4 hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <span
                          className={`px-2 py-1 text-xs font-medium rounded ${
                            item.mode === 'text_to_sign'
                              ? 'bg-blue-100 text-blue-700'
                              : 'bg-green-100 text-green-700'
                          }`}
                        >
                          {item.mode === 'text_to_sign' ? 'Text → Sign' : 'Sign → Text'}
                        </span>
                        <span className="text-sm text-gray-500">{item.sign_language}</span>
                        <span className="text-sm text-gray-400">
                          {format(new Date(item.created_at), 'MMM d, yyyy HH:mm')}
                        </span>
                      </div>

                      <p className="text-gray-900">
                        {item.mode === 'text_to_sign'
                          ? item.input_text
                          : item.output_text || 'No text recognized'}
                      </p>

                      <div className="flex items-center gap-4 mt-3">
                        {item.detected_emotion && (
                          <EmotionBadge emotion={item.detected_emotion as Emotion} size="sm" />
                        )}
                        {item.confidence_score && (
                          <span className="text-sm text-gray-500">
                            {Math.round(item.confidence_score * 100)}% confidence
                          </span>
                        )}
                        <span className="text-sm text-gray-400">
                          {item.processing_time_ms}ms
                        </span>
                      </div>
                    </div>

                    <button
                      onClick={() => deleteItem(item.id)}
                      className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                    >
                      <FiTrash2 size={18} />
                    </button>
                  </div>
                </motion.div>
              ))}

              {pagination.totalPages > 1 && (
                <div className="flex justify-center gap-2 mt-8">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => loadHistory(pagination.page - 1)}
                    disabled={pagination.page === 1}
                  >
                    Previous
                  </Button>
                  <span className="px-4 py-2 text-gray-600">
                    Page {pagination.page} of {pagination.totalPages}
                  </span>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => loadHistory(pagination.page + 1)}
                    disabled={pagination.page === pagination.totalPages}
                  >
                    Next
                  </Button>
                </div>
              )}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}
