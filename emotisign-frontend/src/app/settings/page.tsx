'use client';

import { useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import {
  FiArrowLeft,
  FiUser,
  FiBell,
  FiGlobe,
  FiShield,
  FiHelpCircle,
  FiInfo,
  FiChevronRight,
  FiLogOut,
  FiTrash2,
} from 'react-icons/fi';
import Button from '@/components/ui/Button';
import Select from '@/components/ui/Select';
import { useAuthStore } from '@/store/auth';

const settingsSections = [
  {
    title: 'Account',
    items: [
      { id: 'profile', label: 'Edit Profile', icon: FiUser, href: '/profile' },
      { id: 'notifications', label: 'Notifications', icon: FiBell, href: null },
      { id: 'language', label: 'Language & Region', icon: FiGlobe, href: null },
    ],
  },
  {
    title: 'Privacy & Security',
    items: [
      { id: 'privacy', label: 'Privacy Settings', icon: FiShield, href: null },
    ],
  },
  {
    title: 'Support',
    items: [
      { id: 'help', label: 'Help Center', icon: FiHelpCircle, href: null },
      { id: 'about', label: 'About EmotiSign', icon: FiInfo, href: null },
    ],
  },
];

export default function SettingsPage() {
  const { user, isGuest, logout } = useAuthStore();
  const [signLanguage, setSignLanguage] = useState(user?.preferred_sign_language || 'ASL');
  const [notifications, setNotifications] = useState(true);

  const handleLogout = async () => {
    await logout();
    window.location.href = '/';
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-2xl mx-auto py-8 px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center gap-4 mb-8">
            <Link
              href="/dashboard"
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <FiArrowLeft size={24} />
            </Link>
            <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
          </div>

          {user && !isGuest && (
            <div className="bg-white rounded-xl shadow-sm p-4 mb-6 flex items-center gap-4">
              <div className="w-14 h-14 bg-primary-100 rounded-full flex items-center justify-center">
                <FiUser size={24} className="text-primary-700" />
              </div>
              <div>
                <h2 className="font-semibold text-gray-900">{user.username}</h2>
                <p className="text-gray-500 text-sm">{user.email}</p>
              </div>
              <Link href="/profile" className="ml-auto">
                <Button variant="ghost" size="sm">Edit</Button>
              </Link>
            </div>
          )}

          <div className="bg-white rounded-xl shadow-sm mb-6">
            <div className="p-4 border-b">
              <h3 className="font-semibold text-gray-900">Preferences</h3>
            </div>
            <div className="p-4 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900">Default Sign Language</p>
                  <p className="text-sm text-gray-500">Choose your preferred sign language</p>
                </div>
                <Select
                  value={signLanguage}
                  onChange={(e) => setSignLanguage(e.target.value as 'ASL' | 'PSL')}
                  options={[
                    { value: 'ASL', label: 'ASL' },
                    { value: 'PSL', label: 'PSL' },
                  ]}
                  className="w-24"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900">Push Notifications</p>
                  <p className="text-sm text-gray-500">Receive notifications for messages</p>
                </div>
                <button
                  onClick={() => setNotifications(!notifications)}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    notifications ? 'bg-primary-600' : 'bg-gray-300'
                  }`}
                >
                  <span
                    className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      notifications ? 'left-7' : 'left-1'
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>

          {settingsSections.map((section) => (
            <div key={section.title} className="bg-white rounded-xl shadow-sm mb-6">
              <div className="p-4 border-b">
                <h3 className="font-semibold text-gray-900">{section.title}</h3>
              </div>
              <div className="divide-y">
                {section.items.map((item) => {
                  const Icon = item.icon;
                  const content = (
                    <div className="flex items-center p-4 hover:bg-gray-50 transition-colors cursor-pointer">
                      <Icon size={20} className="text-gray-500 mr-4" />
                      <span className="flex-1 text-gray-700">{item.label}</span>
                      <FiChevronRight size={20} className="text-gray-400" />
                    </div>
                  );

                  return item.href ? (
                    <Link key={item.id} href={item.href}>
                      {content}
                    </Link>
                  ) : (
                    <div key={item.id} onClick={() => toast.info('Coming soon')}>
                      {content}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}

          <div className="bg-white rounded-xl shadow-sm">
            <button
              onClick={handleLogout}
              className="flex items-center p-4 w-full hover:bg-red-50 transition-colors text-red-600"
            >
              <FiLogOut size={20} className="mr-4" />
              <span>{isGuest ? 'Exit Guest Mode' : 'Log Out'}</span>
            </button>

            {!isGuest && (
              <button
                onClick={() => toast.info('Contact support to delete your account')}
                className="flex items-center p-4 w-full border-t hover:bg-red-50 transition-colors text-red-600"
              >
                <FiTrash2 size={20} className="mr-4" />
                <span>Delete Account</span>
              </button>
            )}
          </div>

          <p className="text-center text-gray-400 text-sm mt-8">
            EmotiSign v1.0.0 | Made with care
          </p>
        </motion.div>
      </div>
    </div>
  );
}
