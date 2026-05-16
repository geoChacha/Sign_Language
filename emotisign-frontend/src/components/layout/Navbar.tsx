'use client';

import { useState } from 'react';
import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { FiMenu, FiX, FiMessageSquare, FiSettings, FiLogOut, FiUser } from 'react-icons/fi';
import Logo from '@/components/ui/Logo';
import Button from '@/components/ui/Button';
import { useAuthStore } from '@/store/auth';

interface NavbarProps {
  variant?: 'landing' | 'app';
}

export default function Navbar({ variant = 'landing' }: NavbarProps) {
  const pathname = usePathname();
  const { isAuthenticated, isGuest, user, logout } = useAuthStore();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);

  const landingLinks = [
    { href: '/', label: 'Home' },
    { href: '/#features', label: 'Features' },
    { href: '/#about', label: 'About' },
  ];

  const appLinks = [
    { href: '/dashboard', label: 'Dashboard', icon: null },
    { href: '/chat', label: 'Chat', icon: <FiMessageSquare /> },
    { href: '/settings', label: 'Settings', icon: <FiSettings /> },
  ];

  const links = variant === 'landing' ? landingLinks : appLinks;

  const handleLogout = async () => {
    await logout();
    window.location.href = '/';
  };

  return (
    <nav className={`sticky top-0 z-50 ${variant === 'landing' ? 'bg-white/80' : 'bg-primary-900'} backdrop-blur-md`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Logo size="md" showText={variant === 'landing'} />

          <div className="hidden md:flex items-center gap-8">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`flex items-center gap-2 text-sm font-medium transition-colors ${
                  variant === 'landing'
                    ? pathname === link.href
                      ? 'text-primary-700'
                      : 'text-gray-600 hover:text-primary-600'
                    : pathname === link.href
                    ? 'text-white'
                    : 'text-gray-300 hover:text-white'
                }`}
              >
                {'icon' in link && (link as { icon: React.ReactNode }).icon}
                {link.label}
              </Link>
            ))}
          </div>

          <div className="hidden md:flex items-center gap-4">
            {isAuthenticated || isGuest ? (
              <div className="relative">
                <button
                  onClick={() => setProfileMenuOpen(!profileMenuOpen)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                    variant === 'landing'
                      ? 'hover:bg-gray-100'
                      : 'hover:bg-primary-800'
                  }`}
                >
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    variant === 'landing' ? 'bg-primary-100' : 'bg-primary-700'
                  }`}>
                    <FiUser className={variant === 'landing' ? 'text-primary-700' : 'text-white'} />
                  </div>
                  <span className={`text-sm font-medium ${variant === 'landing' ? 'text-gray-700' : 'text-white'}`}>
                    {isGuest ? 'Guest' : user?.username}
                  </span>
                </button>

                <AnimatePresence>
                  {profileMenuOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg py-2 border"
                    >
                      {!isGuest && (
                        <Link
                          href="/profile"
                          className="flex items-center gap-2 px-4 py-2 text-gray-700 hover:bg-gray-50"
                          onClick={() => setProfileMenuOpen(false)}
                        >
                          <FiUser size={16} />
                          Profile
                        </Link>
                      )}
                      <Link
                        href="/settings"
                        className="flex items-center gap-2 px-4 py-2 text-gray-700 hover:bg-gray-50"
                        onClick={() => setProfileMenuOpen(false)}
                      >
                        <FiSettings size={16} />
                        Settings
                      </Link>
                      <hr className="my-2" />
                      <button
                        onClick={handleLogout}
                        className="flex items-center gap-2 px-4 py-2 text-red-600 hover:bg-red-50 w-full"
                      >
                        <FiLogOut size={16} />
                        {isGuest ? 'Exit Guest Mode' : 'Logout'}
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ) : (
              <>
                {variant === 'landing' && (
                  <Link href="/login">
                    <Button variant="primary" size="sm">
                      Login/Signup
                    </Button>
                  </Link>
                )}
              </>
            )}
          </div>

          <button
            className="md:hidden p-2"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? (
              <FiX size={24} className={variant === 'landing' ? 'text-gray-700' : 'text-white'} />
            ) : (
              <FiMenu size={24} className={variant === 'landing' ? 'text-gray-700' : 'text-white'} />
            )}
          </button>
        </div>
      </div>

      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className={`md:hidden ${variant === 'landing' ? 'bg-white' : 'bg-primary-900'} border-t`}
          >
            <div className="px-4 py-4 space-y-2">
              {links.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`block px-4 py-2 rounded-lg ${
                    variant === 'landing'
                      ? 'text-gray-700 hover:bg-gray-50'
                      : 'text-white hover:bg-primary-800'
                  }`}
                  onClick={() => setMobileMenuOpen(false)}
                >
                  {link.label}
                </Link>
              ))}
              {!isAuthenticated && !isGuest && (
                <Link href="/login" onClick={() => setMobileMenuOpen(false)}>
                  <Button className="w-full mt-4">Login/Signup</Button>
                </Link>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
