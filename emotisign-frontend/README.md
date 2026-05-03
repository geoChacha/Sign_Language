# EmotiSign Frontend

A Next.js web application for EmotiSign - a bidirectional sign language communication platform with emotion detection.

## Features

- **Sign to Text/Speech**: Real-time webcam-based sign language recognition
- **Text/Speech to Sign**: Convert text or speech to animated sign language
- **Two-Way Chat**: Real-time messaging with sign language video support
- **Emotion Detection**: AI-powered emotion analysis for all translations
- **Multi-language Support**: ASL (American Sign Language) & PSL (Pakistan Sign Language)

## Tech Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Animations**: Framer Motion
- **Icons**: React Icons
- **HTTP Client**: Axios
- **WebSocket**: Native WebSocket API with custom wrapper
- **Camera**: react-webcam

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- EmotiSign Backend running on `http://localhost:8000`

### Installation

1. Install dependencies:

```bash
npm install
```

2. Configure environment variables:

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

3. Start the development server:

```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
src/
├── app/                    # Next.js App Router pages
│   ├── page.tsx           # Landing page
│   ├── login/             # Login page
│   ├── signup/            # Registration page
│   ├── dashboard/         # Main dashboard
│   ├── translate/         # Translation pages
│   │   ├── sign-to-text/  # Sign language to text
│   │   └── text-to-sign/  # Text to sign language
│   ├── chat/              # Chat interface
│   ├── profile/           # User profile
│   └── settings/          # App settings
├── components/
│   ├── ui/                # Reusable UI components
│   └── layout/            # Layout components
├── lib/
│   ├── api.ts             # API client
│   └── websocket.ts       # WebSocket utilities
├── store/
│   └── auth.ts            # Authentication state
└── types/
    └── index.ts           # TypeScript types
```

## API Integration

The frontend connects to the EmotiSign FastAPI backend:

- **REST API**: `/api/auth/*`, `/api/translate/*`, `/api/chat/*`, `/api/speech/*`
- **WebSocket**: Real-time translation and chat

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API URL |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000` | WebSocket URL |

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

Note: WebRTC (camera access) requires HTTPS in production.

## License

MIT
