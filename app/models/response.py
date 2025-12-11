from pydantic import BaseModel, Field
from typing import Optional


class GenerateResponse(BaseModel):
    """Response model for successful generation request"""

    success: bool = True
    job_id: str = Field(..., description="External job ID from AI provider")
    provider: str = Field(..., description="AI provider name that succeeded (e.g., 'kie', 'fal')")
    estimated_time_seconds: int = Field(..., description="Estimated generation time")
    generation_id: str = Field(..., description="Original generation ID from Wasp")
    fallback_triggered: Optional[bool] = Field(default=False, description="Whether fallback was triggered")
    primary_provider: Optional[str] = Field(default=None, description="Primary provider attempted (if fallback occurred)")


class ErrorResponse(BaseModel):
    """Response model for errors"""

    success: bool = False
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for debugging")


class GenerateUGCVideoResponse(BaseModel):
    """Response model for complete UGC video generation pipeline"""

    success: bool = True
    generation_id: str = Field(..., description="Original generation ID from Wasp")

    # Stage outputs
    person_url: str = Field(..., description="S3 URL of generated person image")
    person_s3_key: str = Field(..., description="S3 key for person image")
    composite_url: str = Field(..., description="S3 URL of composite image")
    composite_s3_key: str = Field(..., description="S3 key for composite image")
    video_url: str = Field(..., description="S3 URL of final video")
    video_s3_key: str = Field(..., description="S3 key for final video")

    # Provider info
    provider_used: str = Field(..., description="Video generation provider ('kie' or 'fal')")
    fallback_triggered: bool = Field(..., description="Whether Fal.ai fallback was used")

    # Performance
    total_time: float = Field(..., description="Total pipeline execution time in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "generation_id": "123e4567-e89b-12d3-a456-426614174000",
                "person_url": "https://s3.amazonaws.com/ugcvideo-assets/person-images/user-abc-123/gen-123.png",
                "person_s3_key": "person-images/user-abc-123/gen-123.png",
                "composite_url": "https://s3.amazonaws.com/ugcvideo-assets/composites/user-abc-123/gen-123.png",
                "composite_s3_key": "composites/user-abc-123/gen-123.png",
                "video_url": "https://s3.amazonaws.com/ugcvideo-assets/videos/user-abc-123/gen-123.mp4",
                "video_s3_key": "videos/user-abc-123/gen-123.mp4",
                "provider_used": "kie",
                "fallback_triggered": False,
                "total_time": 245.3
            }
        }
