import type { Metadata } from "next";
import { Toaster } from "react-hot-toast";
import { Noto_Nastaliq_Urdu } from "next/font/google";
import "./globals.css";

const urduFont = Noto_Nastaliq_Urdu({
  weight: ['400', '700'],
  subsets: ['arabic'],
  variable: '--font-urdu',
  display: 'swap',
});

export const metadata: Metadata = {
  title: "EmotiSign - Sign Language Translation",
  description: "Seamless, Emotional, Intelligent Communication - ASL & PSL Translation with Emotion Detection",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={urduFont.variable}>
      <body className="antialiased">
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
            success: {
              style: {
                background: '#22c55e',
              },
            },
            error: {
              style: {
                background: '#ef4444',
              },
            },
          }}
        />
        {children}
      </body>
    </html>
  );
}
