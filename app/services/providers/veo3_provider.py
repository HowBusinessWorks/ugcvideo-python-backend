"""
Veo3 Provider for Stage 3 (Video Generation)

Unified provider for Veo3 video generation with automatic fallback:
- Primary: Kie.ai (cheaper, ~50% cost savings)
- Fallback: Fal.ai (more reliable)

Supports both veo3-fast (2-3 min) and veo3-standard (4-6 min) generation modes.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from .base import BaseProvider
from .kie_provider import KieProvider
from .fal_provider import FalProvider

logger = logging.getLogger(__name__)


class Veo3Provider(BaseProvider):
    """
    Unified Veo3 provider with intelligent fallback

    Automatically tries Kie.ai first for cost savings, falls back to Fal.ai if needed.
    """

    def __init__(self, kie_api_key: str = None, fal_api_key: str = None):
        """
        Initialize Veo3 provider with both Kie.ai and Fal.ai

        Args:
            kie_api_key: Not used - kept for backwards compatibility
            fal_api_key: Not used - kept for backwards compatibility

        Note: Both providers get their API keys from settings
        """
        self.kie_provider = KieProvider()
        self.fal_provider = FalProvider()

    async def submit_veo3_job(
        self,
        image_url: str,
        prompt: str,
        use_fast: bool = False,
        duration: int = 8,
        aspect_ratio: str = "9:16",
        webhook_url: Optional[str] = None,
        preferred_provider: str = "kie",
        fallback_timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Generate video from image using Veo3 with automatic fallback

        Args:
            image_url: URL of composite image (from Stage 2)
            prompt: Video animation prompt
            use_fast: Use veo3-fast (2-3 min) vs veo3-standard (4-6 min)
            duration: Video duration in seconds (default: 8)
            aspect_ratio: Video aspect ratio (default: "9:16")
            webhook_url: Optional webhook for completion notification
            preferred_provider: "kie" or "fal" (default: "kie")
            fallback_timeout: Seconds to wait before fallback (default: 60)

        Returns:
            {
                "job_id": "kie_abc123" or "fal_xyz789",
                "provider": "kie" or "fal",
                "fallback_triggered": True/False,
                "primary_provider": "kie",
                "estimated_time": 180
            }
        """
        logger.info(f"🎥 [Veo3] Starting video generation...")
        logger.info(f"📝 Prompt: {prompt[:100]}...")
        logger.info(f"🖼️  Image: {image_url}")
        logger.info(f"⚡ Mode: {'veo3-fast' if use_fast else 'veo3-standard'}")

        # If preferred provider is fal, use it directly (no fallback needed)
        if preferred_provider == "fal":
            logger.info(f"🔄 [Veo3] Using Fal.ai directly (preferred provider)")
            result = await self._submit_to_fal(
                image_url=image_url,
                prompt=prompt,
                duration=duration,
                aspect_ratio=aspect_ratio,
                webhook_url=webhook_url,
                use_fast=use_fast
            )

            return {
                **result,
                "fallback_triggered": False,
                "primary_provider": "fal"
            }

        # Try Kie.ai first with timeout
        primary_error = None

        try:
            logger.info(f"⚡ [Veo3] Attempting Kie.ai (primary, cheaper)...")

            result = await asyncio.wait_for(
                self._submit_to_kie(
                    image_url=image_url,
                    prompt=prompt,
                    use_fast=use_fast,
                    duration=duration,
                    aspect_ratio=aspect_ratio,
                    webhook_url=webhook_url
                ),
                timeout=fallback_timeout
            )

            logger.info(f"✅ [Veo3] Kie.ai succeeded: {result['job_id']}")

            return {
                **result,
                "fallback_triggered": False,
                "primary_provider": "kie"
            }

        except asyncio.TimeoutError:
            primary_error = f"Kie.ai timed out after {fallback_timeout}s"
            logger.warning(f"⏱️ [Veo3] {primary_error}")

        except Exception as e:
            primary_error = f"Kie.ai failed: {str(e)}"
            logger.warning(f"❌ [Veo3] {primary_error}")

        # Fallback to Fal.ai
        try:
            logger.info(f"🔄 [Veo3] Falling back to Fal.ai...")

            result = await self._submit_to_fal(
                image_url=image_url,
                prompt=prompt,
                duration=duration,
                aspect_ratio=aspect_ratio,
                webhook_url=webhook_url,
                use_fast=use_fast
            )

            logger.info(f"✅ [Veo3] Fal.ai succeeded: {result['job_id']}")

            # Send alert about fallback usage
            logger.warning(f"⚠️ [Veo3] Fallback triggered - Primary: {primary_error}")

            return {
                **result,
                "fallback_triggered": True,
                "primary_provider": "kie",
                "primary_error": primary_error
            }

        except Exception as fal_error:
            # Both providers failed - critical error
            logger.error(f"❌ [Veo3] Both providers failed!")
            logger.error(f"   Kie.ai: {primary_error}")
            logger.error(f"   Fal.ai: {str(fal_error)}")

            raise Exception(
                f"Veo3 video generation failed on all providers.\n"
                f"Primary (Kie.ai): {primary_error}\n"
                f"Fallback (Fal.ai): {str(fal_error)}"
            )

    async def _submit_to_kie(
        self,
        image_url: str,
        prompt: str,
        use_fast: bool,
        duration: int,
        aspect_ratio: str,
        webhook_url: Optional[str]
    ) -> Dict[str, Any]:
        """Submit video generation job to Kie.ai using Veo 3.1 API"""

        # Build Kie.ai request for Veo 3.1
        # Kie.ai now supports actual Veo3 models via /api/v1/veo/generate endpoint
        # Supports both veo3 (standard) and veo3_fast models

        class KieRequest:
            def __init__(self):
                self.image_url = image_url
                self.prompt = prompt
                self.system_prompt = ""
                self.parameters = {
                    "use_fast": use_fast,  # veo3_fast or veo3
                    "aspect_ratio": aspect_ratio,  # "16:9" or "9:16"
                    "enable_translation": True  # Auto-translate prompts to English
                }
                self.webhook_url = webhook_url

        request = KieRequest()

        result = await self.kie_provider.submit_image_to_video_job(request)

        return {
            "job_id": result["job_id"],
            "provider": "kie",
            "estimated_time": result.get("estimated_time", 120 if use_fast else 240)
        }

    async def _submit_to_fal(
        self,
        image_url: str,
        prompt: str,
        duration: int,
        aspect_ratio: str,
        webhook_url: Optional[str],
        use_fast: bool = False
    ) -> Dict[str, Any]:
        """Submit video generation job to Fal.ai using Veo3"""

        # Build Fal.ai request
        class FalRequest:
            def __init__(self):
                self.image_url = image_url
                self.prompt = prompt
                self.system_prompt = ""
                self.parameters = {
                    "duration": duration,
                    "aspect_ratio": aspect_ratio,
                    "use_fast": use_fast
                }
                self.webhook_url = webhook_url

        request = FalRequest()

        result = await self.fal_provider.submit_image_to_video_job(request)

        return {
            "job_id": result["job_id"],
            "provider": "fal",
            "estimated_time": result.get("estimated_time", 120 if use_fast else 240)
        }

    async def get_status(self, job_id: str, provider: str) -> Dict[str, Any]:
        """
        Check status of Veo3 generation job

        Args:
            job_id: Task ID from Kie.ai or Fal.ai
            provider: "kie" or "fal"

        Returns:
            {
                "status": "COMPLETED" | "PROCESSING" | "FAILED",
                "result_url": "https://cdn.../video.mp4" (if completed),
                "error": "Error message" (if failed)
            }
        """
        if provider == "kie":
            return await self.kie_provider.get_status(
                job_id,
                model="veo3"  # Use veo3 to trigger Veo3 status endpoint
            )
        elif provider == "fal":
            return await self.fal_provider.get_status(
                job_id,
                model="fal-ai/veo3.1/image-to-video"  # Veo3 endpoint
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def wait_for_completion(
        self,
        job_id: str,
        provider: str,
        timeout: int = 600,
        poll_interval: int = 30
    ) -> str:
        """
        Poll job status until completion or timeout

        Args:
            job_id: Task ID
            provider: "kie" or "fal"
            timeout: Maximum wait time in seconds (default: 600 = 10 min)
            poll_interval: Seconds between status checks (default: 30)

        Returns:
            Result URL of generated video

        Raises:
            Exception: If job fails or times out
        """
        logger.info(f"⏳ [Veo3] Waiting for {provider} job {job_id} to complete...")

        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed > timeout:
                raise Exception(f"Veo3 job {job_id} timed out after {timeout}s")

            status_data = await self.get_status(job_id, provider)

            if status_data["status"] == "COMPLETED":
                if not status_data["result_url"]:
                    raise Exception("Job completed but no result URL found")
                return status_data["result_url"]

            elif status_data["status"] == "FAILED":
                raise Exception(
                    f"Veo3 job failed: {status_data.get('error', 'Unknown error')}"
                )

            # Still processing, wait and retry
            logger.info(f"⏳ [Veo3] Still processing... ({int(elapsed)}s elapsed)")
            await asyncio.sleep(poll_interval)

    def format_prompt(self, user_prompt: str, system_prompt: str = "") -> str:
        """
        Format prompt for Veo3 video generation

        Args:
            user_prompt: User's animation prompt
            system_prompt: Optional system prompt (rarely used for Veo3)

        Returns:
            Formatted prompt string
        """
        if system_prompt:
            return f"{system_prompt}\n\n{user_prompt}"
        return user_prompt

    async def submit_job(self, request) -> Dict[str, Any]:
        """
        Generic job submission (implements BaseProvider interface)

        Routes to submit_veo3_job with request parameters
        """
        return await self.submit_veo3_job(
            image_url=request.image_url,
            prompt=request.prompt,
            use_fast=getattr(request, 'use_fast', False),
            duration=getattr(request, 'duration', 8),
            aspect_ratio=getattr(request, 'aspect_ratio', '9:16'),
            webhook_url=getattr(request, 'webhook_url', None),
            preferred_provider=getattr(request, 'preferred_provider', 'kie'),
            fallback_timeout=getattr(request, 'fallback_timeout', 60)
        )


# Cost tracking
VEO3_COSTS = {
    "kie_fast": 1.50,      # Kie.ai veo3-fast
    "kie_standard": 3.00,  # Kie.ai veo3-standard
    "fal": 3.00            # Fal.ai (fallback)
}


def get_veo3_cost(provider: str, use_fast: bool = False) -> float:
    """
    Get cost estimate for Veo3 generation

    Args:
        provider: "kie" or "fal"
        use_fast: Whether veo3-fast mode was used

    Returns:
        Estimated cost in USD
    """
    if provider == "kie":
        return VEO3_COSTS["kie_fast"] if use_fast else VEO3_COSTS["kie_standard"]
    elif provider == "fal":
        return VEO3_COSTS["fal"]
    else:
        return 0.0
