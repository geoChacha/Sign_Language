'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { FiArrowRight, FiPlay, FiCheck } from 'react-icons/fi';
import Navbar from '@/components/layout/Navbar';
import Button from '@/components/ui/Button';

const features = [
  'ASL & PSL Translation',
  'Advanced Emotion Detection',
  'Interactive Learning & Practice',
  'Connect & Share with Ease',
];

const featureSections = [
  {
    title: 'SEE YOUR WORDS APPEAR',
    description: 'Instantly translate a Sign Language (ASL & Pakistan Sign Language) into clear, readable text. Break down the communication barriers to connect.',
    cta: 'Try Sign-to-Text',
    href: '/translate/sign-to-text',
    image: '🤟',
    reverse: false,
  },
  {
    title: 'A Visual Voice',
    description: 'Type your message and watch our friendly 3D avatar translate it accurate sign language. Communicate freely with the deaf community.',
    cta: 'Try Text-to-Sign',
    href: '/translate/text-to-sign',
    image: '✋',
    reverse: true,
  },
];

const languages = [
  { name: 'American Sign Language', code: 'ASL', rating: 5 },
  { name: 'Pakistan Sign Language', code: 'PSL', rating: 4 },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-violet-50 to-indigo-50">
      <Navbar variant="landing" />

      <section className="relative overflow-hidden py-20 lg:py-28">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-100/50 to-indigo-100/50"></div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
            >
              <h1 className="text-4xl lg:text-5xl font-bold text-primary-900 mb-4">
                EMOTISIGN:
                <br />
                <span className="text-primary-700">THE FUTURE OF CONNECTION IS HERE</span>
              </h1>
              <p className="text-xl text-gray-600 mb-6">
                Seamless, Emotional, Intelligent Communication
              </p>

              <ul className="space-y-3 mb-8">
                {features.map((feature, index) => (
                  <motion.li
                    key={feature}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 + index * 0.1 }}
                    className="flex items-center gap-3"
                  >
                    <span className="text-primary-600">→</span>
                    <span className="text-gray-700">{feature}</span>
                  </motion.li>
                ))}
              </ul>

              <div className="flex flex-wrap gap-4">
                <Link href="/signup">
                  <Button size="lg" rightIcon={<FiArrowRight />}>
                    Get Started
                  </Button>
                </Link>
                <Link href="/demo">
                  <Button variant="secondary" size="lg" leftIcon={<FiPlay />}>
                    Watch Demo
                  </Button>
                </Link>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="relative"
            >
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-4">
                  <div className="bg-white rounded-2xl shadow-lg p-6 transform hover:scale-105 transition-transform">
                    <div className="text-4xl mb-2">👋</div>
                    <p className="text-sm text-gray-600">See your words. Feel the Message.</p>
                  </div>
                  <div className="bg-white rounded-2xl shadow-lg p-6 transform hover:scale-105 transition-transform">
                    <div className="text-4xl mb-2">✋</div>
                    <p className="text-sm text-gray-600">Type & Sign. A visual Voice</p>
                  </div>
                </div>
                <div className="space-y-4 mt-8">
                  <div className="bg-white rounded-2xl shadow-lg p-6 transform hover:scale-105 transition-transform">
                    <div className="text-4xl mb-2">📝</div>
                    <p className="text-sm text-gray-600">Text the Sign</p>
                  </div>
                  <div className="bg-primary-700 rounded-2xl shadow-lg p-6 text-white transform hover:scale-105 transition-transform">
                    <div className="text-4xl mb-2">💬</div>
                    <p className="text-sm">SHARE YOUR STORY</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      <section id="features" className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {featureSections.map((section, index) => (
            <motion.div
              key={section.title}
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className={`flex flex-col ${
                section.reverse ? 'lg:flex-row-reverse' : 'lg:flex-row'
              } items-center gap-12 mb-20`}
            >
              <div className="flex-1">
                <div className="bg-gradient-to-br from-purple-100 to-indigo-100 rounded-3xl p-8 shadow-lg">
                  <div className="aspect-video bg-white rounded-2xl flex items-center justify-center">
                    <span className="text-8xl">{section.image}</span>
                  </div>
                </div>
              </div>
              <div className="flex-1">
                <h2 className="text-3xl font-bold text-gray-900 mb-4">{section.title}</h2>
                <p className="text-gray-600 mb-6 text-lg">{section.description}</p>
                <Link href={section.href}>
                  <Button variant="accent">{section.cta}</Button>
                </Link>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="py-20 bg-gradient-to-r from-primary-900 to-primary-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl font-bold text-white mb-4">Language Support</h2>
            <p className="text-purple-200 max-w-2xl mx-auto">
              Connect in your world, on your terms. EmotiSign offers comprehensive support for multiple sign languages
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8 max-w-2xl mx-auto">
            {languages.map((lang) => (
              <motion.div
                key={lang.code}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                className="bg-white/10 backdrop-blur-md rounded-2xl p-6 text-white"
              >
                <h3 className="text-xl font-semibold mb-2">{lang.name}</h3>
                <p className="text-purple-200 text-sm mb-3">{lang.code}</p>
                <div className="flex gap-1">
                  {[...Array(5)].map((_, i) => (
                    <span key={i} className={i < lang.rating ? 'text-yellow-400' : 'text-gray-500'}>
                      ★
                    </span>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="bg-gradient-to-r from-primary-700 to-primary-900 rounded-3xl p-12 text-center text-white"
          >
            <h2 className="text-3xl font-bold mb-4">JOIN US ANYWHERE</h2>
            <p className="text-purple-200 mb-8 max-w-2xl mx-auto">
              Experience seamless communication across all your devices. Download EmotiSign and connect on the go!
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <Button variant="accent" size="lg">
                Download App
              </Button>
              <Link href="/dashboard">
                <Button variant="secondary" size="lg" className="!bg-white/10 !text-white !border-white/30 hover:!bg-white/20">
                  Try Live Demo
                </Button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <h3 className="text-lg font-bold mb-4">EmotiSign</h3>
              <p className="text-gray-400 text-sm">
                Breaking communication barriers with AI-powered sign language translation.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Features</h4>
              <ul className="space-y-2 text-gray-400 text-sm">
                <li><Link href="/translate/sign-to-text" className="hover:text-white">Sign to Text</Link></li>
                <li><Link href="/translate/text-to-sign" className="hover:text-white">Text to Sign</Link></li>
                <li><Link href="/chat" className="hover:text-white">Remote Chat</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-gray-400 text-sm">
                <li><Link href="/about" className="hover:text-white">About Us</Link></li>
                <li><Link href="/contact" className="hover:text-white">Contact</Link></li>
                <li><Link href="/privacy" className="hover:text-white">Privacy Policy</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Support</h4>
              <ul className="space-y-2 text-gray-400 text-sm">
                <li><Link href="/help" className="hover:text-white">Help Center</Link></li>
                <li><Link href="/docs" className="hover:text-white">Documentation</Link></li>
                <li><Link href="/faq" className="hover:text-white">FAQ</Link></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400 text-sm">
            © 2026 EmotiSign. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
