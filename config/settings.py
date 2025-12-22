# config/settings.py
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # App
    APP_NAME: str = "CryptixProj"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # API Keys
    ANTHROPIC_API_KEY: str
    
    # Database
    DATABASE_URL: str = "postgresql://user:pass@localhost/cryptix"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Supported Cryptos
    SUPPORTED_CRYPTOS: List[str] = ["BTC", "ETH", "USDT", "USDC"]
    
    # Model Config
    LSTM_LOOK_BACK: int = 60
    LSTM_UNITS: List[int] = [128, 64, 32]
    LSTM_DROPOUT: float = 0.2
    LSTM_LEARNING_RATE: float = 0.001
    
    # Training
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    TEST_SIZE: float = 0.2
    
    # Continuous Learning
    ONLINE_LEARNING_LR: float = 0.0001
    REPLAY_BUFFER_SIZE: int = 1000
    RETRAIN_SCHEDULE: str = "0 2 * * *"  # 2 AM daily
    
    # Performance Thresholds
    MIN_ACCURACY: float = 0.75
    MAX_DRIFT_SCORE: float = 0.6
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()