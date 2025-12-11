from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Security
    API_KEY: str

    # Wasp Backend
    WASP_WEBHOOK_URL: str

    # AI Providers
    # Prompt Enhancement (GPT-4o via OpenRouter)
    OPENROUTER_API_KEY: str
    OPENROUTER_MODEL: str = "openai/gpt-4o"

    # Stage 1 & 2: Person Generation + Compositing
    SEEDREAM_API_KEY: str

    # Stage 3: Video Generation (Veo3)
    KIE_API_KEY: str  # Primary provider (cheaper)
    FAL_KEY: str  # Fallback provider (more reliable)

    # AWS S3 Storage
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_S3_BUCKET: str = "ugcvideo-assets"
    AWS_REGION: str = "us-east-1"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
