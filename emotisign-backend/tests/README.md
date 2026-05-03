# WLASL-100 Model Tests

This directory contains unit tests and integration tests for the WLASL-100 ASL recognition system.

## Setup

Install test dependencies:

```bash
pip install pytest pytest-asyncio pytest-cov requests
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Running Tests

### Run all tests:

```bash
pytest
```

### Run with verbose output:

```bash
pytest -v
```

### Run specific test file:

```bash
pytest tests/test_model.py
pytest tests/test_ml_service.py
```

### Run with coverage report:

```bash
pytest --cov=app/ml --cov=app/services --cov-report=html --cov-report=term
```

### Run only unit tests:

```bash
pytest -m unit
```

### Run system verification script:

```bash
python test_system.py
```

This script tests:
1. Model loading
2. Keypoint extraction
3. Inference with dummy data
4. Environment validation
5. API endpoints (if server is running)

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── fixtures/                # Test data (videos, images)
│   └── README.md
├── test_model.py            # Model architecture tests
├── test_ml_service.py       # ML service tests
└── README.md                # This file
```

## Test Coverage

Current test coverage:

- **Model Architecture**: ✅ Initialization, forward pass, dropout, parameters
- **Sequence Resampling**: ✅ Upsampling, downsampling, padding
- **TTA Variants**: ✅ 4 variants generation, horizontal mirror
- **Statistics**: ✅ Initial stats, updates after prediction

## Writing New Tests

### Example unit test:

```python
import pytest
from app.ml.wlasl_service import WLASLModelService

def test_my_feature():
    """Test description."""
    service = WLASLModelService(
        model_path="dummy",
        vocab_path="dummy",
        device="cpu"
    )
    
    # Test logic here
    assert True
```

### Example async test:

```python
import pytest

@pytest.mark.asyncio
async def test_async_feature():
    """Test async functionality."""
    result = await some_async_function()
    assert result is not None
```

### Using fixtures:

```python
def test_with_fixture(dummy_keypoints):
    """Test using fixture from conftest.py."""
    assert dummy_keypoints.shape == (100, 126)
```

## Continuous Integration

To run tests in CI/CD:

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests with coverage
pytest --cov=app/ml --cov=app/services --cov-report=xml

# Upload coverage to codecov (optional)
# codecov -f coverage.xml
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'app'"

Make sure you're running pytest from the `emotisign-backend` directory:

```bash
cd emotisign-backend
pytest
```

### "Model file not found"

Some tests require the actual model files. Make sure they exist:

```bash
ls app/ml/models/wlasl100/
# Should show: best_model.pth, vocab.json, temperature.json
```

### "CUDA out of memory"

Tests use CPU by default. If you see CUDA errors, make sure tests are using `device="cpu"`.

### "MediaPipe initialization failed"

Ensure mediapipe==0.9.3 is installed (not 0.10+):

```bash
pip install mediapipe==0.9.3
```

## Performance Benchmarks

To run performance benchmarks:

```bash
pytest tests/test_performance.py -v
```

Expected performance (with TTA):
- GPU: 200-500ms per video
- CPU: 1-2s per video

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Ensure tests pass: `pytest`
3. Check coverage: `pytest --cov`
4. Update this README if needed

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
