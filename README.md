# EmotiSign

Bidirectional sign language communication system with real-time ASL recognition and text-to-sign animation.

## Features

- **Text → Sign** — Type any text and see real ASL skeleton animations for 100 supported words
- **Sign → Text** — Upload a video of sign language and get text back (WLASL-100 model)
- **Speech → Text** — Voice input support
- **Chat** — Real-time chat with sign language support
- **Emotion detection** — Sentiment analysis on text and facial expressions from video

## Quick Start — Docker (Recommended)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd emotisign
cp .env.docker .env
# Edit .env and set a strong SECRET_KEY
```

### 2. Build and start

```bash
docker compose up --build
```

- **Frontend** → http://localhost:3000
- **Backend API** → http://localhost:8000
- **API Docs** → http://localhost:8000/docs

### 3. Stop

```bash
docker compose down
```

Data (database, uploads) is persisted in Docker named volumes across restarts.

---

## Quick Start — Native (Windows)

### Prerequisites

- **Python 3.10.9** — the setup script will download and install it automatically if missing
- [Node.js 18+](https://nodejs.org/) — must be installed manually

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd emotisign
```

### 2. Run setup (once)

```bat
setup.bat
```

### 3. Start the app

```bat
start.bat
```

Opens two terminal windows — backend at http://localhost:8000, frontend at http://localhost:3000.

---

## Manual Setup

If you prefer to run things manually:

### Backend

```bash
cd emotisign-backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd emotisign-frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
```

---

## Project Structure

```
emotisign/
├── emotisign-backend/          # FastAPI backend
│   ├── app/
│   │   ├── ml/
│   │   │   ├── sign_generator.py   # Text-to-sign keypoint service
│   │   │   ├── wlasl_service.py    # Sign-to-text WLASL-100 model
│   │   │   └── models/wlasl100/    # Model weights + vocab
│   │   ├── routers/                # API endpoints
│   │   └── services/               # WebSocket, frame processing
│   ├── main.py
│   └── requirements.txt
│
├── emotisign-frontend/         # Next.js frontend
│   └── src/
│       ├── app/translate/      # Translation pages
│       └── components/ui/
│           └── SkeletonCanvas.tsx  # ASL skeleton animation
│
├── Featrure_sign_generation/   # ASL keypoint data
│   └── keypoints_best/         # 100 .npy files (one per word)
│
├── setup.bat                   # One-time setup script
└── start.bat                   # Start both servers
```

---

## ASL Vocabulary (100 words)

The text-to-sign feature supports these words. Any other word is fingerspelled automatically.

`accident` `africa` `all` `apple` `basketball` `bed` `before` `bird` `birthday` `black` `blue` `book` `bowling` `brown` `but` `can` `candy` `chair` `change` `cheat` `city` `clothes` `color` `computer` `cook` `cool` `corn` `cousin` `cow` `dance` `dark` `deaf` `decide` `doctor` `dog` `drink` `eat` `enjoy` `family` `fine` `finish` `fish` `forget` `full` `give` `go` `graduate` `hat` `hearing` `help` `hot` `how` `jacket` `kiss` `language` `last` `later` `letter` `like` `man` `many` `medicine` `meet` `mother` `need` `no` `now` `orange` `paint` `paper` `pink` `pizza` `play` `pull` `purple` `right` `same` `school` `secretary` `shirt` `short` `son` `study` `table` `tall` `tell` `thanksgiving` `thin` `thursday` `time` `walk` `want` `what` `white` `who` `woman` `work` `wrong` `year` `yes`

---

## Environment Variables

Copy `emotisign-backend/.env.example` to `emotisign-backend/.env` and adjust as needed.

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | *(required)* | JWT signing secret |
| `ML_DEVICE` | `cpu` | `cpu` or `cuda` for GPU inference |
| `KEYPOINTS_DIR` | *(auto-detected)* | Path to `keypoints_best/` folder |
| `SIGN_FRAME_RATE_MS` | `50` | Animation speed in ms per frame |

---

## API

Interactive docs available at **http://localhost:8000/docs** when the backend is running.

Key endpoints:
- `POST /api/translate/text-to-sign` — Convert text to ASL keypoints
- `POST /api/translate/sign-to-text` — Upload video, get text
- `POST /api/auth/register` — Create account
- `POST /api/auth/login` — Get JWT token
