"""
Seedream Provider for Stage 1 (Person Generation) and Stage 2 (Compositing)

Seedream provides high-quality image generation and editing capabilities:
- Stage 1: Text-to-image for generating AI persons
- Stage 2: Image-to-image editing for compositing products with persons
"""

import httpx
import asyncio
import logging
from typing import Dict, Any, Optional

from .base import BaseProvider

logger = logging.getLogger(__name__)


class SeedreamProvider(BaseProvider):
    """Provider for Seedream API - handles person generation and compositing"""

    BASE_URL = "https://api.seedream.ai/v1"

    def __init__(self, api_key: str):
        """
        Initialize Seedream provider

        Args:
            api_key: Seedream API key
        """
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def submit_text_to_image_job(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1080,
        height: int = 1440,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        webhook_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stage 1: Generate person image from text prompt

        Args:
            prompt: Text description of person to generate
            negative_prompt: What to avoid in generation
            width: Image width in pixels (default: 1080)
            height: Image height in pixels (default: 1440 for 9:16)
            num_inference_steps: Quality vs speed tradeoff (default: 50)
            guidance_scale: How closely to follow prompt (default: 7.5)
            webhook_url: Optional webhook for completion notification

        Returns:
            {
                "job_id": "seed_abc123",
                "provider": "seedream",
                "status": "processing"
            }
        """
        try:
            logger.info(f"🎨 [Seedream] Generating person image...")
            logger.info(f"📝 Prompt: {prompt[:100]}...")

            payload = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "style": "photorealistic",
                "negative_prompt": negative_prompt or "cartoon, anime, illustration, low quality, blurry, distorted",
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale
            }

            if webhook_url:
                payload["webhook_url"] = webhook_url

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/text-to-image",
                    headers=self.headers,
                    json=payload
                )

                response.raise_for_status()
                data = response.json()

                logger.info(f"✅ [Seedream] Job submitted: {data.get('task_id')}")

                return {
                    "job_id": data["task_id"],
                    "provider": "seedream",
                    "status": data.get("status", "processing"),
                    "estimated_time": data.get("estimated_time", 45)
                }

        except httpx.HTTPError as e:
            logger.error(f"❌ [Seedream] HTTP error: {str(e)}")
            raise Exception(f"Seedream text-to-image failed: {str(e)}")

        except Exception as e:
            logger.error(f"❌ [Seedream] Unexpected error: {str(e)}")
            raise

    async def submit_image_edit_job(
        self,
        base_image_url: str,
        prompt: str,
        overlay_image_url: Optional[str] = None,
        strength: float = 0.8,
        edit_mode: str = "inpaint_and_blend",
        webhook_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stage 2: Composite product with person image

        Args:
            base_image_url: URL of person image (from Stage 1)
            prompt: Instructions for compositing (e.g., "Place skincare bottle in right hand")
            overlay_image_url: Optional product image to composite
            strength: Edit strength (0.0 to 1.0, default: 0.8)
            edit_mode: "inpaint_and_blend" or "edit" (default: inpaint_and_blend)
            webhook_url: Optional webhook for completion notification

        Returns:
            {
                "job_id": "seed_xyz789",
                "provider": "seedream",
                "status": "processing"
            }
        """
        try:
            logger.info(f"🎨 [Seedream] Compositing product with person...")
            logger.info(f"📝 Prompt: {prompt[:100]}...")
            logger.info(f"🖼️  Base image: {base_image_url}")

            payload = {
                "base_image_url": base_image_url,
                "prompt": prompt,
                "edit_mode": edit_mode,
                "strength": strength
            }

            if overlay_image_url:
                payload["overlay_image_url"] = overlay_image_url
                logger.info(f"🛍️  Product image: {overlay_image_url}")

            if webhook_url:
                payload["webhook_url"] = webhook_url

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/image-edit",
                    headers=self.headers,
                    json=payload
                )

                response.raise_for_status()
                data = response.json()

                logger.info(f"✅ [Seedream] Job submitted: {data.get('task_id')}")

                return {
                    "job_id": data["task_id"],
                    "provider": "seedream",
                    "status": data.get("status", "processing"),
                    "estimated_time": data.get("estimated_time", 30)
                }

        except httpx.HTTPError as e:
            logger.error(f"❌ [Seedream] HTTP error: {str(e)}")
            raise Exception(f"Seedream image-edit failed: {str(e)}")

        except Exception as e:
            logger.error(f"❌ [Seedream] Unexpected error: {str(e)}")
            raise

    async def get_status(self, job_id: str, model: str = "") -> Dict[str, Any]:
        """
        Check status of Seedream generation job

        Args:
            job_id: Seedream task ID
            model: Model name (not used for Seedream, kept for interface compatibility)

        Returns:
            {
                "status": "COMPLETED" | "PROCESSING" | "FAILED",
                "result_url": "https://cdn.seedream.../image.png" (if completed),
                "error": "Error message" (if failed),
                "progress": 75 (optional, percentage)
            }
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.BASE_URL}/status/{job_id}",
                    headers=self.headers
                )

                response.raise_for_status()
                data = response.json()

                # Map Seedream status to standard format
                status_map = {
                    'pending': 'PROCESSING',
                    'processing': 'PROCESSING',
                    'completed': 'COMPLETED',
                    'failed': 'FAILED'
                }

                status = status_map.get(data.get('status', 'processing'), 'PROCESSING')

                result = {
                    "status": status,
                    "result_url": None,
                    "error": None,
                    "progress": data.get('progress', 0)
                }

                # Extract result URL if completed
                if status == 'COMPLETED' and data.get('image_url'):
                    result["result_url"] = data["image_url"]
                    logger.info(f"✅ [Seedream] Job {job_id} completed: {result['result_url']}")

                elif status == 'FAILED':
                    result["error"] = data.get('error', 'Unknown error')
                    logger.error(f"❌ [Seedream] Job {job_id} failed: {result['error']}")

                else:
                    logger.info(f"⏳ [Seedream] Job {job_id} still processing ({result['progress']}%)")

                return result

        except httpx.HTTPError as e:
            logger.error(f"❌ [Seedream] Status check failed: {str(e)}")
            raise Exception(f"Seedream status check failed: {str(e)}")

        except Exception as e:
            logger.error(f"❌ [Seedream] Unexpected error: {str(e)}")
            raise

    async def wait_for_completion(
        self,
        job_id: str,
        timeout: int = 120,
        poll_interval: int = 5
    ) -> str:
        """
        Poll job status until completion or timeout

        Args:
            job_id: Seedream task ID
            timeout: Maximum wait time in seconds (default: 120)
            poll_interval: Seconds between status checks (default: 5)

        Returns:
            Result URL of generated image

        Raises:
            Exception: If job fails or times out
        """
        logger.info(f"⏳ [Seedream] Waiting for job {job_id} to complete...")

        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed > timeout:
                raise Exception(f"Seedream job {job_id} timed out after {timeout}s")

            status_data = await self.get_status(job_id)

            if status_data["status"] == "COMPLETED":
                if not status_data["result_url"]:
                    raise Exception("Job completed but no result URL found")
                return status_data["result_url"]

            elif status_data["status"] == "FAILED":
                raise Exception(f"Seedream job failed: {status_data.get('error', 'Unknown error')}")

            # Still processing, wait and retry
            await asyncio.sleep(poll_interval)

    async def submit_job(self, request) -> Dict[str, Any]:
        """
        Generic job submission (implements BaseProvider interface)

        This method routes to the appropriate Seedream endpoint based on request type.
        For more control, use submit_text_to_image_job() or submit_image_edit_job() directly.
        """
        # Determine if this is text-to-image or image-edit based on request fields
        if hasattr(request, 'base_image_url') and request.base_image_url:
            # Stage 2: Image editing/compositing
            return await self.submit_image_edit_job(
                base_image_url=request.base_image_url,
                prompt=request.prompt,
                overlay_image_url=getattr(request, 'overlay_image_url', None),
                webhook_url=getattr(request, 'webhook_url', None)
            )
        else:
            # Stage 1: Text-to-image person generation
            return await self.submit_text_to_image_job(
                prompt=request.prompt,
                negative_prompt=getattr(request, 'negative_prompt', ''),
                webhook_url=getattr(request, 'webhook_url', None)
            )


# Helper function to build person prompt from Easy Mode form fields
def build_person_prompt_from_fields(
    gender: str,
    age: str,
    ethnicity: str,
    clothing: str,
    expression: str,
    background: str
) -> str:
    """
    Build Seedream-optimized prompt from Easy Mode form fields

    Args:
        gender: "male", "female", "non-binary"
        age: "18-25", "26-35", "36-45", "46-60", "60+"
        ethnicity: "Caucasian", "African", "Asian", "Hispanic", etc.
        clothing: "casual", "business", "athletic", etc.
        expression: "smiling", "neutral", "excited", etc.
        background: "white", "outdoor", "home", etc.

    Returns:
        Optimized prompt string for Seedream
    """
    age_map = {
        "18-25": "young adult",
        "26-35": "adult",
        "36-45": "middle-aged adult",
        "46-60": "mature adult",
        "60+": "senior"
    }

    clothing_map = {
        "casual": "wearing casual t-shirt and jeans",
        "business": "wearing professional business attire",
        "athletic": "wearing athletic sportswear",
        "formal": "wearing formal dress clothing"
    }

    background_map = {
        "white": "on plain white background",
        "outdoor": "outdoors in natural lighting",
        "home": "in modern home interior",
        "studio": "in professional photo studio"
    }

    # Build UGC-style prompt with amateur aesthetic (matching n8n workflow)
    base_description = [
        f"Medium shot portrait of {age_map.get(age, age)}",  # Medium shot = good framing
        f"{ethnicity} {gender}",
        f"{expression} expression",
        clothing_map.get(clothing, f"wearing {clothing}"),
        background_map.get(background, f"{background} background"),
        "hand raised naturally in front of body, palm open and visible"  # Ready for Stage 2 compositing
    ]

    # Critical UGC camera keywords (from n8n workflow)
    camera_keywords = [
        "unremarkable amateur iPhone photo",
        "reddit image",
        "snapchat photo",
        "Casual iPhone selfie",
        "Authentic share",
        "slightly blurry",
        "amateur quality phone photo"
    ]

    # UGC realism characteristics
    ugc_style = [
        "everyday realism",
        "authentic and relatable setting",
        "candid pose",
        "genuine expression",
        "visible imperfections",
        "blemishes",
        "messy hair",
        "uneven skin texture",
        "natural lighting with slight imperfections"
    ]

    # Combine all parts
    prompt = ", ".join(base_description + camera_keywords + ugc_style)

    logger.info(f"📝 Built UGC-style person prompt: {prompt[:100]}...")

    return prompt
