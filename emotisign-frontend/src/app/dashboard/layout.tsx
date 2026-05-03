'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Navbar from '@/components/layout/Navbar';
import { useAuthStore } from '@/store/auth';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const { isAuthenticated, isGuest, fetchUser } = useAuthStore();

  useEffect(() => {
    fetchUser();
  }, [fetchUser]);

  useEffect(() => {
    if (!isAuthenticated && !isGuest) {
      const timer = setTimeout(() => {
        const { isAuthenticated: auth, isGuest: guest } = useAuthStore.getState();
        if (!auth && !guest) {
          router.push('/login');
        }
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [isAuthenticated, isGuest, router]);

  return (
    <div className="min-h-screen gradient-bg">
      <Navbar variant="app" />
      <main>{children}</main>
    </div>
  );
}
