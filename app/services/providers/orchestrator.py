import asyncio
from typing import Dict, Any, Optional
from .kie_provider import KieProvider
from .fal_provider import FalProvider


class ProviderOrchestrator:
    """
    Smart orchestrator that tries primary provider with timeout,
    then falls back to secondary provider if needed.

    This ensures:
    - Cost savings by using cheaper providers first
    - Reliability through automatic fallback
    - Configurable timeout per template
    """

    def __init__(self):
        self.kie = KieProvider()
        self.fal = FalProvider()

    async def submit_image_job_with_fallback(
        self,
        request: Any,
        primary_provider: str = "kie",
        fallback_provider: str = "fal",
        timeout_seconds: int = 45
    ) -> Dict[str, Any]:
        """
        Submit image generation job with smart fallback

        Args:
            request: GenerateImageRequest
            primary_provider: Provider to try first ('kie' or 'fal')
            fallback_provider: Provider to use if primary fails/times out
            timeout_seconds: How long to wait for primary before fallback

        Returns:
            Dict with:
                - job_id: The job ID from whichever provider succeeded
                - estimated_time: Estimated completion time
                - provider_used: Which provider was actually used
                - fallback_triggered: Whether fallback was triggered
        """

        print(f"🎯 Starting generation with primary: {primary_provider}, fallback: {fallback_provider}, timeout: {timeout_seconds}s")

        # Get provider instances
        primary = self._get_provider(primary_provider)
        fallback_prov = self._get_provider(fallback_provider)

        # Try primary provider with timeout
        try:
            print(f"⚡ Trying {primary_provider}...")

            # Use asyncio.wait_for to enforce timeout
            result = await asyncio.wait_for(
                primary.submit_image_job(request),
                timeout=timeout_seconds
            )

            print(f"✅ {primary_provider} succeeded!")

            # Add provider metadata to result
            result["provider_used"] = primary_provider
            result["fallback_triggered"] = False

            return result

        except asyncio.TimeoutError:
            print(f"⏱️  {primary_provider} timed out after {timeout_seconds}s, falling back to {fallback_provider}...")

        except Exception as e:
            print(f"❌ {primary_provider} failed with error: {str(e)}")
            print(f"🔄 Falling back to {fallback_provider}...")

        # Fallback to secondary provider (no timeout - let it complete)
        try:
            print(f"⚡ Trying fallback provider {fallback_provider}...")

            result = await fallback_prov.submit_image_job(request)

            print(f"✅ Fallback provider {fallback_provider} succeeded!")

            # Add provider metadata
            result["provider_used"] = fallback_provider
            result["fallback_triggered"] = True
            result["primary_provider"] = primary_provider  # Track what we tried first

            return result

        except Exception as e:
            # Both providers failed
            print(f"💥 Both providers failed! Primary: {primary_provider}, Fallback: {fallback_provider}")
            raise Exception(
                f"All providers failed. Primary ({primary_provider}) and "
                f"fallback ({fallback_provider}) both errored: {str(e)}"
            )

    async def submit_image_to_video_job_with_fallback(
        self,
        request: Any,
        primary_provider: str = "kie",
        fallback_provider: str = "fal",
        timeout_seconds: int = 60
    ) -> Dict[str, Any]:
        """
        Submit image-to-video generation job with smart fallback

        Args:
            request: GenerateImageToVideoRequest object
            primary_provider: Primary provider to try ('kie' or 'fal')
            fallback_provider: Fallback provider if primary fails
            timeout_seconds: Timeout for primary provider (default: 60s for video)

        Returns:
            Dict with job_id, provider_used, estimated_time, fallback_triggered, primary_provider
        """
        print(f"🎬 Starting video generation with primary: {primary_provider}, timeout: {timeout_seconds}s")

        primary = self._get_provider(primary_provider)
        fallback_prov = self._get_provider(fallback_provider)

        # Try primary provider with timeout
        try:
            print(f"⚡ Trying {primary_provider} for image-to-video...")

            result = await asyncio.wait_for(
                primary.submit_image_to_video_job(request),
                timeout=timeout_seconds
            )

            print(f"✅ {primary_provider} succeeded for video generation!")

            # Add provider metadata
            result["provider_used"] = primary_provider
            result["fallback_triggered"] = False

            return result

        except asyncio.TimeoutError:
            print(f"⏱️  {primary_provider} timed out after {timeout_seconds}s, falling back to {fallback_provider}...")

        except Exception as e:
            print(f"❌ {primary_provider} failed with error: {str(e)}")
            print(f"🔄 Falling back to {fallback_provider}...")

        # Fallback to secondary provider (no timeout - let it complete)
        try:
            print(f"⚡ Trying fallback provider {fallback_provider} for video...")

            result = await fallback_prov.submit_image_to_video_job(request)

            print(f"✅ Fallback provider {fallback_provider} succeeded for video!")

            # Add provider metadata
            result["provider_used"] = fallback_provider
            result["fallback_triggered"] = True
            result["primary_provider"] = primary_provider  # Track what we tried first

            return result

        except Exception as e:
            # Both providers failed
            print(f"💥 Both providers failed for video! Primary: {primary_provider}, Fallback: {fallback_provider}")
            raise Exception(
                f"All providers failed for video generation. Primary ({primary_provider}) and "
                f"fallback ({fallback_provider}) both errored: {str(e)}"
            )

    async def check_status_any_provider(
        self,
        job_id: str,
        provider: str,
        model: str
    ) -> Dict[str, Any]:
        """
        Check status on the appropriate provider

        Args:
            job_id: Job/task ID
            provider: Which provider to check ('kie' or 'fal')
            model: Model identifier

        Returns:
            Dict with status and result_url
        """
        provider_instance = self._get_provider(provider)
        return await provider_instance.check_status(job_id, model)

    def _get_provider(self, name: str):
        """Get provider instance by name"""
        if name.lower() == "kie":
            return self.kie
        elif name.lower() == "fal":
            return self.fal
        else:
            raise ValueError(f"Unknown provider: {name}. Must be 'kie' or 'fal'")
