from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class GenerateVideoRequest(BaseModel):
    """Request model for video generation"""

    generation_id: str = Field(..., description="Unique ID from Wasp backend")
    model: str = Field(..., description="AI model to use (e.g., 'wan-t2v')")
    prompt: str = Field(..., description="User's text prompt")
    system_prompt: str = Field(..., description="System/template prompt")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model-specific parameters")
    webhook_url: str = Field(..., description="URL to call when generation completes")

    class Config:
        json_schema_extra = {
            "example": {
                "generation_id": "123e4567-e89b-12d3-a456-426614174000",
                "model": "wan-t2v",
                "prompt": "A woman trying on a cozy sweater",
                "system_prompt": "Generate a UGC-style video...",
                "parameters": {
                    "aspect_ratio": "9:16",
                    "resolution": "480p",
                    "num_frames": 81
                },
                "webhook_url": "https://your-app.com/api/webhooks/ai"
            }
        }


class GenerateImageToVideoRequest(BaseModel):
    """Request model for image-to-video generation (Minimax Hailuo) - supports provider fallback"""

    generation_id: str = Field(..., description="Unique ID from Wasp backend")
    model: str = Field(..., description="AI model to use (e.g., 'minimax-hailuo-i2v')")
    image_url: str = Field(..., description="S3 URL of the uploaded reference image")
    prompt: str = Field(..., description="Motion description prompt")
    system_prompt: str = Field(..., description="System/template prompt")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model-specific parameters")
    webhook_url: str = Field(..., description="URL to call when generation completes")

    # Provider fallback configuration
    preferred_provider: str = Field(default="kie", description="Primary provider to try first")
    fallback_timeout: int = Field(default=60, description="Seconds before fallback")
    fallback_provider: str = Field(default="fal", description="Fallback provider")

    class Config:
        json_schema_extra = {
            "example": {
                "generation_id": "123e4567-e89b-12d3-a456-426614174000",
                "model": "minimax-hailuo-i2v",
                "image_url": "https://ecomassets-generations.s3.eu-north-1.amazonaws.com/uploads/product.jpg",
                "prompt": "The product rotates 360 degrees smoothly",
                "system_prompt": "Generate a professional product video...",
                "parameters": {
                    "prompt_optimizer": True
                },
                "webhook_url": "https://your-app.com/api/webhooks/ai"
            }
        }


class GenerateImageRequest(BaseModel):
    """Request model for image editing (SeeDream V4) - supports multiple images"""

    generation_id: str = Field(..., description="Unique ID from Wasp backend")
    model: str = Field(..., description="AI model to use (e.g., 'seedream-v4')")
    image_urls: List[str] = Field(..., description="List of S3 URLs for images to combine/enhance")
    prompt: str = Field(..., description="Editing/enhancement instructions prompt")
    system_prompt: str = Field(..., description="System/template prompt")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model-specific parameters")
    webhook_url: str = Field(..., description="URL to call when generation completes")

    # Provider fallback configuration
    preferred_provider: str = Field(default="kie", description="Primary provider to try first")
    fallback_timeout: int = Field(default=45, description="Seconds before fallback")
    fallback_provider: str = Field(default="fal", description="Fallback provider")

    class Config:
        json_schema_extra = {
            "example": {
                "generation_id": "123e4567-e89b-12d3-a456-426614174000",
                "model": "seedream-v4",
                "image_urls": [
                    "https://ecomassets-generations.s3.eu-north-1.amazonaws.com/uploads/product.jpg",
                    "https://ecomassets-generations.s3.eu-north-1.amazonaws.com/uploads/background.jpg"
                ],
                "prompt": "Combine the product with the background, add professional studio lighting",
                "system_prompt": "Enhance and combine the uploaded images...",
                "parameters": {
                    "num_images": 1,
                    "enable_safety_checker": False
                },
                "webhook_url": "https://your-app.com/api/webhooks/ai"
            }
        }


class GenerateTextToImageRequest(BaseModel):
    """Request model for text-to-image generation (SeeDream V4 Text-to-Image) - NO reference images"""

    generation_id: str = Field(..., description="Unique ID from Wasp backend")
    model: str = Field(..., description="AI model to use (e.g., 'seedream-v4-text')")
    prompt: str = Field(..., description="Text prompt to generate image from")
    system_prompt: str = Field(..., description="System/template prompt")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model-specific parameters")
    webhook_url: str = Field(..., description="URL to call when generation completes")

    class Config:
        json_schema_extra = {
            "example": {
                "generation_id": "123e4567-e89b-12d3-a456-426614174000",
                "model": "seedream-v4-text",
                "prompt": "A futuristic city skyline at sunset, cyberpunk style with neon lights",
                "system_prompt": "Generate a high-quality, detailed image...",
                "parameters": {
                    "image_size": "square_hd",
                    "num_images": 1,
                    "enhance_prompt_mode": "standard"
                },
                "webhook_url": "https://your-app.com/api/webhooks/ai"
            }
        }


class PersonFieldsRequest(BaseModel):
    """Form fields for Easy Mode person generation"""

    gender: str = Field(..., description="Person gender: 'male', 'female', 'non-binary'")
    age: str = Field(..., description="Age range: '18-25', '26-35', '36-45', '46-60', '60+'")
    ethnicity: str = Field(..., description="Ethnicity: 'Caucasian', 'African', 'Asian', 'Hispanic', etc.")
    clothing: str = Field(..., description="Clothing style: 'casual', 'business', 'athletic', 'formal'")
    expression: str = Field(..., description="Facial expression: 'smiling', 'neutral', 'excited', etc.")
    background: str = Field(..., description="Background type: 'white', 'outdoor', 'home', 'studio'")


class GenerateUGCVideoRequest(BaseModel):
    """
    Request model for complete 3-stage UGC video generation pipeline

    This endpoint orchestrates:
    - Stage 1: Generate AI person (Easy or Advanced mode)
    - Stage 2: Composite person with product
    - Stage 3: Generate video animation (Veo3 with fallback)
    """

    generation_id: str = Field(..., description="Unique ID from Wasp backend (VideoGeneration record)")
    user_id: str = Field(..., description="User UUID")

    # Stage 1: Person Generation
    stage1_mode: str = Field(..., description="'EASY' (form fields) or 'ADVANCED' (custom prompt)")
    person_prompt: Optional[str] = Field(None, description="Custom prompt for ADVANCED mode")
    person_fields: Optional[PersonFieldsRequest] = Field(None, description="Form fields for EASY mode")

    # Stage 2: Compositing
    product_image_url: str = Field(..., description="S3 URL of uploaded product image")
    composite_prompt: Optional[str] = Field(None, description="Optional custom compositing instructions")

    # Stage 3: Video Generation
    video_prompt: str = Field(..., description="Prompt for video animation")
    veo3_mode: str = Field(default="STANDARD", description="'FAST' (2-3 min) or 'STANDARD' (4-6 min)")

    # Options
    duration: int = Field(default=8, description="Video duration in seconds")
    aspect_ratio: str = Field(default="9:16", description="Video aspect ratio (default portrait)")
    webhook_url: Optional[str] = Field(None, description="Optional webhook for progress updates")

    class Config:
        json_schema_extra = {
            "example": {
                "generation_id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": "user-abc-123",
                "stage1_mode": "EASY",
                "person_fields": {
                    "gender": "female",
                    "age": "26-35",
                    "ethnicity": "Asian",
                    "clothing": "casual",
                    "expression": "smiling",
                    "background": "white"
                },
                "product_image_url": "https://s3.amazonaws.com/ugcvideo-assets/products/user-abc-123/product-xyz.png",
                "composite_prompt": None,
                "video_prompt": "Woman enthusiastically presents the skincare product to camera, smiling and gesturing naturally",
                "veo3_mode": "STANDARD",
                "duration": 8,
                "aspect_ratio": "9:16",
                "webhook_url": "https://your-app.com/api/webhooks/video-progress"
            }
        }


# ===== NEW REQUEST MODELS FOR TAB-BASED ARCHITECTURE =====

class GeneratePersonRequest(BaseModel):
    """
    Request model for Stage 1 only: Person Generation

    Stage 1 generates ONLY the person image - no product included.
    The person will have hand positioned naturally, ready for Stage 2 compositing.
    """

    generation_id: str = Field(..., description="Unique ID from Wasp backend")
    user_id: str = Field(..., description="User UUID")
    stage1_mode: str = Field(..., description="'EASY' (form fields) or 'ADVANCED' (custom prompt)")
    person_prompt: Optional[str] = Field(None, description="Custom prompt for ADVANCED mode")
    person_fields: Optional[PersonFieldsRequest] = Field(None, description="Form fields for EASY mode")

    class Config:
        json_schema_extra = {
            "example": {
                "generation_id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": "user-abc-123",
                "stage1_mode": "EASY",
                "person_fields": {
                    "gender": "female",
                    "age": "26-35",
                    "ethnicity": "Asian",
                    "clothing": "casual",
                    "expression": "smiling",
                    "background": "white"
                }
            }
        }


class GenerateCompositeRequest(BaseModel):
    """
    Request model for Stage 2 only: Product Compositing

    This endpoint composites a person image with a product image independently.
    Users can upload their own person image or use one from their library.
    """

    generation_id: str = Field(..., description="Unique ID from Wasp backend")
    user_id: str = Field(..., description="User UUID")
    person_image_url: str = Field(..., description="S3 URL of person image (from Stage 1 or uploaded)")
    product_image_url: str = Field(..., description="S3 URL of product image")
    composite_prompt: Optional[str] = Field(None, description="Optional compositing instructions")

    class Config:
        json_schema_extra = {
            "example": {
                "generation_id": "456e7890-e89b-12d3-a456-426614174001",
                "user_id": "user-abc-123",
                "person_image_url": "https://s3.amazonaws.com/ugcvideo-assets/person-images/user-abc-123/person-xyz.jpg",
                "product_image_url": "https://s3.amazonaws.com/ugcvideo-assets/products/user-abc-123/product-xyz.png",
                "composite_prompt": "Place the product naturally in the person's hand with realistic lighting"
            }
        }


class GenerateVideoOnlyRequest(BaseModel):
    """
    Request model for Stage 3 only: Video Generation

    This endpoint generates video from a composite image independently.
    Users can upload their own composite or use one from their library.
    """

    generation_id: str = Field(..., description="Unique ID from Wasp backend")
    user_id: str = Field(..., description="User UUID")
    composite_image_url: str = Field(..., description="S3 URL of composite image (from Stage 2 or uploaded)")
    video_prompt: str = Field(..., description="Prompt for video animation")
    veo3_mode: str = Field(default="STANDARD", description="'FAST' (2-3 min) or 'STANDARD' (4-6 min)")
    duration: int = Field(default=8, description="Video duration in seconds")
    aspect_ratio: str = Field(default="9:16", description="Video aspect ratio")
    product_image_url: Optional[str] = Field(None, description="Product image URL (optional, for GPT-4o prompt enhancement)")

    class Config:
        json_schema_extra = {
            "example": {
                "generation_id": "789e0123-e89b-12d3-a456-426614174002",
                "user_id": "user-abc-123",
                "composite_image_url": "https://s3.amazonaws.com/ugcvideo-assets/composites/user-abc-123/composite-xyz.jpg",
                "video_prompt": "Woman enthusiastically presents the product to camera, smiling and gesturing naturally",
                "veo3_mode": "STANDARD",
                "duration": 8,
                "aspect_ratio": "9:16"
            }
        }
