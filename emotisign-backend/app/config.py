from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Authentication
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./emotisign.db"
    
    # Upload
    UPLOAD_DIR: str = "uploads/videos"
    MAX_VIDEO_SIZE_MB: int = 50
    
    # ML Model Configuration
    MODEL_PATH: str = "app/ml/models/wlasl100/best_model.pth"
    VOCAB_PATH: str = "app/ml/models/wlasl100/vocab.json"
    TEMPERATURE_PATH: str = "app/ml/models/wlasl100/temperature.json"
    ML_DEVICE: str = "cpu"
    USE_TTA: bool = True
    CONFIDENCE_THRESHOLD: float = 0.25
    MAX_VIDEO_DURATION_SEC: int = 30

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from .env file


@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
