/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  // Disable ESLint during build — run it separately in CI
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: false,
  },
  images: {
    domains: ['localhost', '127.0.0.1'],
  },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
  },

  // ── API / WebSocket proxy rewrites ──────────────────────────────────────
  // Proxying through Next.js avoids cross-origin requests entirely when the
  // backend URL is not the same host as the frontend (e.g. Docker Compose,
  // any deployed environment). The browser talks to /api/* on the same origin;
  // Next.js forwards it to the backend — no CORS preflight needed at all.
  //
  // To enable: set NEXT_PUBLIC_USE_PROXY=true in your .env.local
  // When disabled (default in dev with both servers on localhost): the frontend
  // calls the backend directly using NEXT_PUBLIC_API_URL.
  async rewrites() {
    if (process.env.NEXT_PUBLIC_USE_PROXY !== 'true') {
      return [];
    }

    const backendUrl = process.env.BACKEND_INTERNAL_URL || 'http://backend:8000';

    return [
      // REST API
      {
        source: '/proxy/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
      // WebSocket endpoints — Next.js rewrites work for WS too
      {
        source: '/proxy/ws/:path*',
        destination: `${backendUrl}/ws/:path*`,
      },
      // Static files served by the backend (uploaded videos, signs, etc.)
      {
        source: '/proxy/uploads/:path*',
        destination: `${backendUrl}/uploads/:path*`,
      },
      {
        source: '/proxy/static/:path*',
        destination: `${backendUrl}/static/:path*`,
      },
    ];
  },
};

export default nextConfig;
