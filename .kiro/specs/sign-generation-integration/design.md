# Design Document: sign-generation-integration

## Overview

This feature replaces the placeholder `text_to_sign()` stub in the EmotiSign backend with a real implementation that loads pre-recorded ASL keypoint data from `.npy` files and returns raw landmark arrays to the frontend. The frontend replaces its `<img>` tag animation with an HTML5 Canvas renderer that draws a MediaPipe Holistic skeleton frame-by-frame.

The key architectural shift is moving rendering from the server to the client. Instead of generating PNG frames server-side and base64-encoding them, the backend returns compact `(N_frames, 75, 2)` float arrays. The browser draws each frame directly onto a `<canvas>` element using the same skeleton topology defined in `test_two.py`. This eliminates server-side image generation overhead entirely and reduces payload size significantly.

**Scope summary:**
- Backend: new `SignGenerator` class, updated `ml_service.py` delegation, lifespan initialization, schema update
- Frontend: new `SkeletonCanvas` component, updated `text-to-sign/page.tsx`, updated `SignData` type
- No database schema changes; no new API routes; no Redis or external caching

---

## Architecture

### Data Flow

```
.npy files on disk
      │
      ▼ (startup: load() called once)
┌─────────────────────┐
│   SignGenerator     │  ← process-level in-memory cache
│  (sign_generator.py)│    { word → List[List[List[float]]] }
└─────────────────────┘
      │ get_sign(word) / text_to_sign(text)
      ▼
┌─────────────────────┐
│   ml_service.py     │  ← thin delegation layer
│  text_to_sign()     │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  translation.py     │  ← existing FastAPI router (unchanged logic)
│  POST /text-to-sign │
└─────────────────────┘
      │ JSON response: { signs: [{ word, keypoints, frames:[], ... }] }
      ▼
┌─────────────────────┐
│  text-to-sign/      │  ← Next.js page
│  page.tsx           │
└─────────────────────┘
      │ keypoints array per word
      ▼
┌─────────────────────┐
│  SkeletonCanvas.tsx │  ← new reusable component
│  (HTML5 Canvas)     │    draws skeleton frame-by-frame
└─────────────────────┘
```

### Component Diagram

```mermaid
graph TD
    A[".npy files\nFeatrure_sign_generation/keypoints_best/"] -->|startup load| B[SignGenerator\nsign_generator.py]
    B -->|in-memory cache| B
    B -->|delegate| C[ml_service.py\ntext_to_sign()]
    C -->|sign result dict| D[translation.py\nPOST /api/translate/text-to-sign]
    D -->|JSON response| E[text-to-sign/page.tsx]
    E -->|keypoints prop| F[SkeletonCanvas.tsx]
    F -->|drawLine / drawArc| G[HTML5 Canvas Element]
```

### Startup Initialization Sequence

```
Application start (lifespan)
  │
  ├─ init_db()
  ├─ get_wlasl_service()   ← existing sign-to-text model
  └─ SignGenerator.load()
       │
       ├─ Read KEYPOINTS_DIR env var (default: Featrure_sign_generation/keypoints_best)
       ├─ Glob *.npy files
       ├─ For each file:
       │    ├─ np.load(path)                    → shape (N, 75, 2+)
       │    ├─ Strip Z: array[:, :, :2]         → shape (N, 75, 2)
       │    ├─ Convert to Python list           → JSON-serializable
       │    └─ Store in self._cache[word]
       └─ Log vocabulary size (or error if 0 files)
```

---

## Components and Interfaces

### Backend: `SignGenerator` (sign_generator.py)

```python
class SignGenerator:
    """
    Loads ASL keypoint .npy files at startup and serves them on demand.
    Thread-safe for read access; cache is populated once at load() time.
    """

    def __init__(self) -> None:
        self._cache: dict[str, list[list[list[float]]]] = {}
        # shape: word → [frame_0, frame_1, ...] where frame_i = [[x,y]*75]
        self._vocabulary: set[str] = set()
        self._frame_rate_ms: int = 50  # default; overridden by SIGN_FRAME_RATE_MS

    def load(self, keypoints_dir: str | None = None) -> None:
        """
        Load all .npy files from keypoints_dir into the in-memory cache.
        Called once during application lifespan startup.

        Args:
            keypoints_dir: Override path (used in tests). If None, reads
                           KEYPOINTS_DIR env var, falling back to
                           'Featrure_sign_generation/keypoints_best'.
        Raises:
            Nothing — all errors are logged; missing dir falls back to empty vocab.
        """

    def get_sign(self, word: str) -> list[list[list[float]]] | None:
        """
        Return cached keypoint frames for a vocabulary word.

        Args:
            word: Lowercase word string.
        Returns:
            List of frames, each frame being a list of 75 [x, y] pairs.
            None if word is not in vocabulary.
        """

    async def text_to_sign(self, text: str, sign_language: str = "ASL") -> dict:
        """
        Tokenize text and assemble sign data for the API response.

        Tokenization: split on whitespace, strip non-alpha, lowercase.
        Vocabulary hit  → keypoints from cache, frames=[], fingerspelled=False
        Vocabulary miss → fingerspelling chars in frames, keypoints=None, fingerspelled=True

        Returns dict matching the existing ml_service.text_to_sign() contract:
        {
            "words": List[str],
            "signs": List[SignDict],
            "total_duration_ms": int,
            "sign_language": str,
            "fingerspelled_words": List[str],
        }
        where SignDict = {
            "word": str,
            "keypoints": List[List[List[float]]] | None,
            "frames": List[str],
            "gif_url": None,
            "fingerspelled": bool,
        }
        """

    @property
    def vocabulary(self) -> set[str]:
        """Read-only view of loaded word set."""
        return frozenset(self._vocabulary)
```

**Key implementation notes:**
- `load()` uses `numpy.load(path, allow_pickle=True)` then slices `[:, :, :2]` to strip Z/visibility channels
- Each array is converted to a plain Python `list` via `.tolist()` so it is JSON-serializable without a custom encoder
- Files with unexpected shapes (not `(N, 75, *)` where `*` >= 2) are logged as errors and skipped
- `_frame_rate_ms` is read from `os.environ.get("SIGN_FRAME_RATE_MS", "50")` with a fallback on non-numeric values

### Backend: `ml_service.py` (modified)

The existing `text_to_sign()` async function is replaced with a thin delegation:

```python
# Module-level singleton (set by main.py lifespan)
_sign_generator: SignGenerator | None = None

def set_sign_generator(sg: SignGenerator) -> None:
    global _sign_generator
    _sign_generator = sg

async def text_to_sign(text: str, sign_language: str = "ASL") -> dict:
    if _sign_generator is None:
        # Fallback: return empty result if generator not initialized
        return {"words": [], "signs": [], "total_duration_ms": 0,
                "sign_language": sign_language, "fingerspelled_words": []}
    return await _sign_generator.text_to_sign(text, sign_language)
```

### Backend: `main.py` (modified lifespan)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing setup ...
    
    # Initialize SignGenerator
    try:
        from app.ml.sign_generator import SignGenerator
        from app.ml.ml_service import set_sign_generator
        sg = SignGenerator()
        sg.load()
        set_sign_generator(sg)
        print(f"✅ SignGenerator loaded {len(sg.vocabulary)} signs")
    except Exception as e:
        print(f"⚠️  Warning: SignGenerator initialization failed: {e}")
    
    # ... existing WLASL init ...
    yield
    # ... existing cleanup ...
```

### Backend: `schemas.py` (modified `SignUnit`)

```python
class SignUnit(BaseModel):
    word: str
    keypoints: Optional[List[List[List[float]]]] = None  # (N_frames, 75, 2) — NEW
    frames: List[str] = []
    gif_url: Optional[str] = None
    fingerspelled: bool = False
```

The `TextToSignResponse.signs` field remains `List[Dict[str, Any]]` to avoid breaking the existing router, but the dict shape now includes `keypoints`.

### Frontend: `SkeletonCanvas.tsx` (new component)

```typescript
interface SkeletonCanvasProps {
  /** All frames for the current word: shape [N_frames][75][2] */
  frames: number[][][];
  /** Word label shown below the canvas */
  word: string;
  /** Milliseconds per frame (default 50) */
  frameRateMs?: number;
  width?: number;
  height?: number;
  className?: string;
}

export default function SkeletonCanvas({
  frames,
  word,
  frameRateMs = 50,
  width = 400,
  height = 400,
  className,
}: SkeletonCanvasProps): JSX.Element
```

**Internal state:**
- `canvasRef: RefObject<HTMLCanvasElement>`
- `frameIndexRef: RefObject<number>` — current frame index (mutable ref, not state, to avoid re-renders)
- `rafRef: RefObject<number>` — `requestAnimationFrame` handle for cleanup

**Rendering loop:**
```
useEffect (triggered when frames or frameRateMs changes):
  cancel previous RAF
  frameIndex = 0
  lastTimestamp = 0

  function tick(timestamp):
    if timestamp - lastTimestamp >= frameRateMs:
      drawFrame(canvas, frames[frameIndex % frames.length])
      frameIndex++
      lastTimestamp = timestamp
    rafRef.current = requestAnimationFrame(tick)

  rafRef.current = requestAnimationFrame(tick)
  return () => cancelAnimationFrame(rafRef.current)
```

**`drawFrame(ctx, frame)` algorithm:**
```
ctx.clearRect(0, 0, width, height)
ctx.fillStyle = '#1a1a2e'  // dark background
ctx.fillRect(0, 0, width, height)

body    = frame[0..32]    // indices 0–32
leftHand  = frame[33..53] // indices 33–53
rightHand = frame[54..74] // indices 54–74

function isValid(lm): return !(lm[0] === 0 && lm[1] === 0)
function toPixel(lm): return [lm[0] * width, lm[1] * height]

// Draw body connections (POSE_CONNECTIONS)
//   first 12 entries → cyan (#00FFFF)
//   remaining entries → magenta (#FF00FF)
for each [i, j] in POSE_CONNECTIONS:
  if isValid(body[i]) && isValid(body[j]):
    drawLine(toPixel(body[i]), toPixel(body[j]), color, lineWidth=3)

// Draw left hand (HAND_CONNECTIONS) → red (#FF4444)
for each [i, j] in HAND_CONNECTIONS:
  if isValid(leftHand[i]) && isValid(leftHand[j]):
    drawLine(toPixel(leftHand[i]), toPixel(leftHand[j]), '#FF4444', lineWidth=2)

// Draw right hand (HAND_CONNECTIONS) → lime-green (#00FF00)
for each [i, j] in HAND_CONNECTIONS:
  if isValid(rightHand[i]) && isValid(rightHand[j]):
    drawLine(toPixel(rightHand[i]), toPixel(rightHand[j]), '#00FF00', lineWidth=2)

// Draw landmark dots (optional, for clarity)
// body → blue dots, left hand → red dots, right hand → lime dots
```

**Topology constants (mirrored from test_two.py):**
```typescript
const POSE_CONNECTIONS: [number, number][] = [
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],
  [9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
  [11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[27,31],[23,26],[26,28],[28,32],[24,26]
];

const HAND_CONNECTIONS: [number, number][] = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[5,9],[9,10],[10,11],[11,12],
  [0,13],[9,13],[13,14],[14,15],[15,16],
  [0,17],[13,17],[17,18],[18,19],[19,20],
];
```

### Frontend: `types/index.ts` (modified `SignData`)

```typescript
export interface SignData {
  word: string;
  keypoints: number[][][] | null;  // (N_frames, 75, 2) — NEW
  frames: string[];
  gif_url: string | null;
  fingerspelled: boolean;
}
```

### Frontend: `text-to-sign/page.tsx` (modified)

The page replaces the `<motion.img>` animation block with `<SkeletonCanvas>`. Key changes:

1. Remove `currentFrame: string | null` state
2. Add `currentKeypoints: number[][][] | null` state
3. In `translateViaRest()`, when iterating signs, set `currentKeypoints` to `sign.keypoints` for vocabulary words (or null for fingerspelled)
4. Replace the `<AnimatePresence>` / `<motion.img>` block with:

```tsx
{currentKeypoints ? (
  <SkeletonCanvas
    frames={currentKeypoints}
    word={currentWord}
    frameRateMs={50}
    width={400}
    height={400}
    className="w-full h-full"
  />
) : (
  // existing placeholder / loading state
)}
```

5. The "Sign Preview" thumbnails at the bottom are updated similarly — each thumbnail renders the first frame of `sign.keypoints` onto a small canvas instead of an `<img>` tag.

---

## Data Models

### Keypoint Array Shape

```
Raw .npy file:   (N_frames, 75, D)   where D ≥ 2 (may include Z, visibility)
After stripping: (N_frames, 75, 2)   X and Y only, normalized [0.0, 1.0]

Landmark layout:
  Indices  0–32  → MediaPipe Pose (33 body landmarks)
  Indices 33–53  → MediaPipe Left Hand (21 landmarks)
  Indices 54–74  → MediaPipe Right Hand (21 landmarks)
```

### API Response Shape

```json
{
  "translation_id": 42,
  "input_text": "basketball",
  "words": ["basketball"],
  "signs": [
    {
      "word": "basketball",
      "keypoints": [
        [[0.50, 0.30], [0.48, 0.31], ...],
        ...
      ],
      "frames": [],
      "gif_url": null,
      "fingerspelled": false
    }
  ],
  "total_duration_ms": 1500,
  "sign_language": "ASL",
  "fingerspelled_words": [],
  "emotion_analysis": { ... },
  "processing_time_ms": 12
}
```

For a fingerspelled word (e.g., "xyz"):
```json
{
  "word": "xyz",
  "keypoints": null,
  "frames": ["x", "y", "z"],
  "gif_url": null,
  "fingerspelled": true
}
```

### In-Memory Cache Structure

```python
# SignGenerator._cache
{
    "basketball": [                    # word (str)
        [[0.50, 0.30], [0.48, 0.31], ...],  # frame 0: 75 [x,y] pairs
        [[0.51, 0.29], [0.47, 0.32], ...],  # frame 1
        ...                                  # N_frames total
    ],
    "apple": [...],
    # ... 100 words total
}
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Vocabulary Filename Mapping

*For any* set of `.npy` files in the keypoints directory, the vocabulary exposed by `SignGenerator` after `load()` SHALL equal the set of lowercase filename stems (filenames without the `.npy` extension), with no additions or omissions.

**Validates: Requirements 1.3**

---

### Property 2: Tokenization Strips Non-Alphabetic Characters

*For any* input string passed to `text_to_sign()`, every token in the resulting `words` list SHALL consist exclusively of lowercase alphabetic characters (matching `[a-z]+`). No digits, punctuation, or whitespace SHALL appear in any token.

**Validates: Requirements 2.1**

---

### Property 3: Vocabulary Word Response Shape

*For any* word present in the vocabulary, the corresponding sign object returned by `text_to_sign()` SHALL satisfy all of the following simultaneously:
- `fingerspelled` is `False`
- `keypoints` is a non-empty list of frames
- Each frame contains exactly 75 landmark pairs
- Each landmark pair contains exactly 2 float values (X and Y)
- `frames` is an empty list `[]`
- `gif_url` is `None`

**Validates: Requirements 2.2, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2**

---

### Property 4: Non-Vocabulary Word Fingerspelling

*For any* word absent from the vocabulary, the corresponding sign object returned by `text_to_sign()` SHALL satisfy all of the following simultaneously:
- `fingerspelled` is `True`
- `keypoints` is `None`
- `frames` is a non-empty list of single-character strings matching the letters of the word
- The word appears in the `fingerspelled_words` list of the response

**Validates: Requirements 2.3, 4.3, 4.4**

---

### Property 5: Word Order Preservation

*For any* input text containing a sequence of words, the order of sign objects in the `signs` list returned by `text_to_sign()` SHALL match the left-to-right order of the corresponding tokens in the input text.

**Validates: Requirements 2.4**

---

### Property 6: Duration Calculation

*For any* input text containing only vocabulary words, the `total_duration_ms` value in the response SHALL equal the sum of frame counts across all sign objects multiplied by the configured frame rate (default 50 ms per frame).

Formally: `total_duration_ms == sum(len(sign["keypoints"]) for sign in signs) * frame_rate_ms`

**Validates: Requirements 4.5, 5.1**

---

### Property 7: Canvas Renders All Valid Connections

*For any* keypoint frame where a pair of landmarks `(i, j)` are both valid (neither has both X and Y equal to zero), the `drawFrame()` function SHALL invoke a canvas line-drawing call for that connection using the correct color for the landmark group (cyan/magenta for body, red for left hand, lime-green for right hand). Conversely, *for any* landmark that is invalid (both X and Y are zero), no canvas line-drawing call SHALL be made for any connection involving that landmark index.

**Validates: Requirements 3.5, 3.6**

---

### Property 8: Coordinate Scaling

*For any* normalized landmark coordinate `(x, y)` in `[0.0, 1.0]` and canvas dimensions `(W, H)`, the pixel position used in the canvas drawing call SHALL be `(x * W, y * H)`.

**Validates: Requirements 3.7**

---

### Property 9: Cache Idempotence

*For any* vocabulary word, calling `get_sign(word)` multiple times SHALL return arrays with identical values on every call. The word SHALL be present in `SignGenerator._cache` after the first call, and subsequent calls SHALL not perform any disk I/O (the cache entry is reused).

**Validates: Requirements 7.2, 7.3**

---

## Error Handling

### Backend Error Scenarios

| Scenario | Detection | Response |
|---|---|---|
| `KEYPOINTS_DIR` does not exist | `Path.exists()` check in `load()` | Log error with resolved path; vocabulary = empty set; all words fingerspelled |
| `.npy` file has wrong shape `(N, M, D)` where `M ≠ 75` or `D < 2` | Shape assertion after `np.load()` | Log warning with filename and actual shape; skip word; word is fingerspelled |
| `.npy` file is corrupted / unreadable | `try/except` around `np.load()` | Log warning with filename and exception; skip word; word is fingerspelled |
| `SignGenerator` not initialized when request arrives | `_sign_generator is None` check in `ml_service.text_to_sign()` | Return empty result dict (no crash); all words fingerspelled |
| Unhandled exception in `/api/translate/text-to-sign` | FastAPI exception handler | HTTP 500 with `{"detail": "Internal server error"}` — no stack trace in response body |

### Frontend Error Scenarios

| Scenario | Detection | Response |
|---|---|---|
| `sign.keypoints` is `null` (fingerspelled word) | Null check before passing to `SkeletonCanvas` | Render existing placeholder / fingerspelling display; do not mount `SkeletonCanvas` |
| Canvas context unavailable (`getContext` returns null) | Null check in `useEffect` | Log warning; component renders empty canvas element |
| `frames` array is empty | Length check before starting animation loop | No animation started; canvas shows blank dark background |
| Frame data has wrong landmark count | Runtime length check in `drawFrame` | Skip connections that reference out-of-bounds indices; log warning once |

### Logging Strategy

- `ERROR` level: directory not found, zero files loaded, all frames for a word failed
- `WARNING` level: individual file load failure, individual frame skip
- `INFO` level: successful load with vocabulary count, frame rate configuration

---

## Testing Strategy

### Unit Tests (Backend — pytest)

**`tests/test_sign_generator.py`**

- `test_load_populates_vocabulary`: Load from a temp directory with known `.npy` files; assert vocabulary size and word membership.
- `test_load_skips_corrupted_file`: Place one corrupted file among valid ones; assert only valid words are in vocabulary.
- `test_load_missing_directory`: Pass nonexistent path; assert vocabulary is empty.
- `test_get_sign_returns_correct_shape`: For a loaded word, assert `get_sign(word)` returns `(N, 75, 2)` shaped data.
- `test_get_sign_unknown_word_returns_none`: Assert `get_sign("notaword")` returns `None`.
- `test_text_to_sign_empty_input`: Pass whitespace-only string; assert `signs == []` and `words == []`.
- `test_text_to_sign_fingerspelled_word_in_list`: Pass a non-vocabulary word; assert it appears in `fingerspelled_words`.
- `test_frame_rate_env_var`: Set `SIGN_FRAME_RATE_MS=100`; assert `_frame_rate_ms == 100`.
- `test_frame_rate_default`: Unset env var; assert `_frame_rate_ms == 50`.

**`tests/test_translation_endpoint.py`** (additions)

- `test_text_to_sign_vocabulary_word_has_keypoints`: POST a vocabulary word; assert response sign has `keypoints` non-null and `frames == []`.
- `test_text_to_sign_unknown_word_fingerspelled`: POST a non-vocabulary word; assert `fingerspelled == True` and word in `fingerspelled_words`.
- `test_text_to_sign_500_on_exception`: Mock `text_to_sign` to raise; assert HTTP 500 with no traceback.

### Property-Based Tests (Backend — pytest + Hypothesis)

Property-based testing is appropriate here because `SignGenerator` contains pure data-transformation logic (tokenization, shape normalization, cache lookup) where input variation meaningfully exercises edge cases.

**Library:** [Hypothesis](https://hypothesis.readthedocs.io/) (`pip install hypothesis`)

**`tests/test_sign_generator_properties.py`**

Each test runs a minimum of 100 iterations (Hypothesis default).

```python
# Feature: sign-generation-integration, Property 1: Vocabulary filename mapping
@given(st.lists(st.text(alphabet=st.characters(whitelist_categories=('Ll',)), min_size=1), min_size=1))
def test_vocabulary_equals_filename_stems(word_list): ...

# Feature: sign-generation-integration, Property 2: Tokenization strips non-alpha
@given(st.text())
def test_tokenization_strips_non_alpha(text): ...

# Feature: sign-generation-integration, Property 3: Vocabulary word response shape
@given(st.sampled_from(list(VOCABULARY)))
def test_vocabulary_word_response_shape(word): ...

# Feature: sign-generation-integration, Property 4: Non-vocabulary word fingerspelling
@given(st.text(alphabet=st.characters(whitelist_categories=('Ll',)), min_size=1)
       .filter(lambda w: w not in VOCABULARY))
def test_non_vocabulary_word_is_fingerspelled(word): ...

# Feature: sign-generation-integration, Property 5: Word order preservation
@given(st.lists(st.text(alphabet=st.characters(whitelist_categories=('Ll',)), min_size=1), min_size=1, max_size=20))
def test_word_order_preserved(words): ...

# Feature: sign-generation-integration, Property 6: Duration calculation
@given(st.lists(st.sampled_from(list(VOCABULARY)), min_size=1, max_size=10))
def test_duration_calculation(vocab_words): ...

# Feature: sign-generation-integration, Property 9: Cache idempotence
@given(st.sampled_from(list(VOCABULARY)))
def test_cache_idempotence(word): ...
```

### Property-Based Tests (Frontend — Vitest + fast-check)

**Library:** [fast-check](https://fast-check.dev/) (`npm install --save-dev fast-check`)

**`src/components/ui/__tests__/SkeletonCanvas.test.ts`**

```typescript
// Feature: sign-generation-integration, Property 7: Canvas renders all valid connections
fc.assert(fc.property(
  fc.array(fc.tuple(fc.float({min:0,max:1}), fc.float({min:0,max:1})), {minLength:75, maxLength:75}),
  (landmarks) => { /* verify drawLine called for all valid connections */ }
));

// Feature: sign-generation-integration, Property 7 (zero landmark): No draw for (0,0) landmarks
fc.assert(fc.property(
  fc.integer({min:0, max:32}),  // random body landmark index to zero out
  (zeroIdx) => { /* verify no drawLine calls involving zeroIdx */ }
));

// Feature: sign-generation-integration, Property 8: Coordinate scaling
fc.assert(fc.property(
  fc.float({min:0, max:1}),
  fc.float({min:0, max:1}),
  fc.integer({min:100, max:1000}),
  fc.integer({min:100, max:1000}),
  (x, y, W, H) => { /* verify pixel = (x*W, y*H) */ }
));
```

### Integration Tests

- Start the backend with the real keypoints directory; POST "basketball" to `/api/translate/text-to-sign`; assert `keypoints` is a non-empty 3D array and `frames == []`.
- Assert response time < 500 ms for a single vocabulary word.

### Testing Notes

- Backend property tests use a fixture that calls `SignGenerator.load()` once with the real `keypoints_best/` directory, then reuses the loaded instance across all property test runs to avoid repeated disk I/O.
- Frontend canvas tests mock `CanvasRenderingContext2D` using `jest-canvas-mock` or a manual spy to capture draw calls without a real DOM.
- The `SkeletonCanvas` component is tested in isolation (not mounted in the full page) to keep tests fast and focused.
