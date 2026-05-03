'use client';

import Link from 'next/link';
import { FiArrowLeft } from 'react-icons/fi';

export default function TranslateLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen gradient-bg">
      <div className="p-4">
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-2 text-white/80 hover:text-white transition-colors"
        >
          <FiArrowLeft size={20} />
          <span>Back to Dashboard</span>
        </Link>
      </div>
      <main className="px-4 pb-8">{children}</main>
    </div>
  );
}
