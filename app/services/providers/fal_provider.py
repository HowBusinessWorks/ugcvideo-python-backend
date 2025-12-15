import fal_client
import asyncio
import logging
from typing import Dict, Any, Optional
from app.services.providers.base import BaseProvider
from app.core.config import settings

logger = logging.getLogger(__name__)


class FalProvider(BaseProvider):
    """Fal.ai provider implementation"""

    def __init__(self):
        # Initialize Fal client with API key
        self.client = fal_client.AsyncClient(key=settings.FAL_KEY)

    def format_prompt(self, user_prompt: str, system_prompt: str) -> str:
        """
        Combine system and user prompts

        For Wan T2V, we'll combine both prompts into one
        """
        return f"{system_prompt}\n\n{user_prompt}"

    async def submit_job(self, request: Any) -> Dict[str, Any]:
        """
        Submit video generation job to Fal.ai

        Args:
            request: GenerateVideoRequest object

        Returns:
            Dict with job_id and estimated_time
        """
        # Format the full prompt
        full_prompt = self.format_prompt(request.prompt, request.system_prompt)

        # Prepare arguments for Fal.ai
        fal_arguments = {
            "prompt": full_prompt,
            **request.parameters  # Spread all parameters (aspect_ratio, resolution, etc.)
        }

        # Submit job to Fal.ai with webhook
        result = await self.client.submit(
            "fal-ai/wan-t2v",  # Model endpoint
            arguments=fal_arguments,
            webhook_url=request.webhook_url
        )

        # Return standardized response
        return {
            "job_id": result.request_id,
            "estimated_time": 90  # Wan T2V typically takes 60-120 seconds
        }

    async def submit_image_to_video_job(self, request: Any) -> Dict[str, Any]:
        """
        Submit image-to-video generation job to Fal.ai (Veo3)

        Args:
            request: GenerateImageToVideoRequest object

        Returns:
            Dict with job_id and estimated_time
        """
        # Format the full prompt (combine system + user prompt)
        full_prompt = self.format_prompt(request.prompt, request.system_prompt)

        # Check if fast mode is requested
        use_fast = request.parameters.get("use_fast", False)

        # Get aspect ratio and duration from parameters
        aspect_ratio = request.parameters.get("aspect_ratio", "9:16")
        duration = request.parameters.get("duration", 8)

        # Prepare arguments for Veo3
        fal_arguments = {
            "prompt": full_prompt,
            "image_url": request.image_url,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
        }

        # Choose Veo3 endpoint based on speed mode
        model_endpoint = "fal-ai/veo3.1/fast/image-to-video" if use_fast else "fal-ai/veo3.1/image-to-video"

        # Submit job to Fal.ai with webhook
        result = await self.client.submit(
            model_endpoint,  # Veo3 fast or standard
            arguments=fal_arguments,
            webhook_url=request.webhook_url
        )

        # Return standardized response
        return {
            "job_id": result.request_id,
            "estimated_time": 120 if use_fast else 240  # Veo3 fast: 2 min, standard: 4 min
        }

    async def submit_image_job(self, request: Any) -> Dict[str, Any]:
        """
        Submit image editing job to Fal.ai (SeeDream V4)
        Supports multiple images that will be combined into one result

        Args:
            request: GenerateImageRequest object with image_urls array

        Returns:
            Dict with job_id and estimated_time
        """
        # Format the full prompt (combine system + user prompt)
        full_prompt = self.format_prompt(request.prompt, request.system_prompt)

        # Prepare arguments for SeeDream V4
        fal_arguments = {
            "prompt": full_prompt,
            "image_urls": request.image_urls,  # Pass all images (SeeDream will combine them)
            **request.parameters  # Spread all parameters (num_images, enable_safety_checker, etc.)
        }

        # Submit job to Fal.ai with webhook
        result = await self.client.submit(
            "fal-ai/bytedance/seedream/v4/edit",  # SeeDream V4 model
            arguments=fal_arguments,
            webhook_url=request.webhook_url
        )

        # Return standardized response
        return {
            "job_id": result.request_id,
            "estimated_time": 30  # Image editing is fast (typically 20-40 seconds)
        }

    async def submit_text_to_image_job(self, request: Any) -> Dict[str, Any]:
        """
        Submit text-to-image generation job to Fal.ai (SeeDream V4 Text-to-Image)
        Pure text prompt, no reference images needed

        Args:
            request: GenerateTextToImageRequest object with prompt only

        Returns:
            Dict with job_id and estimated_time
        """
        # Format the full prompt (combine system + user prompt)
        full_prompt = self.format_prompt(request.prompt, request.system_prompt)

        # Prepare arguments for SeeDream V4 Text-to-Image
        fal_arguments = {
            "prompt": full_prompt,
            **request.parameters  # Spread all parameters (image_size, num_images, etc.)
        }

        # Submit job to Fal.ai with webhook
        result = await self.client.submit(
            "fal-ai/bytedance/seedream/v4/text-to-image",  # SeeDream V4 Text-to-Image model
            arguments=fal_arguments,
            webhook_url=request.webhook_url
        )

        # Return standardized response
        return {
            "job_id": result.request_id,
            "estimated_time": 25  # Text-to-image is fast (typically 15-35 seconds)
        }

    async def check_status(self, job_id: str, model: str = "fal-ai/wan-t2v") -> Dict[str, Any]:
        """
        Check the status of a job on Fal.ai

        Args:
            job_id: The request_id returned from submit_job
            model: The Fal.ai model endpoint (default: "fal-ai/wan-t2v")

        Returns:
            Dict with status, result_url (if completed), and error (if failed)
        """
        try:
            # Get job status from Fal.ai
            status_obj = await self.client.status(
                model,
                job_id,
                with_logs=False
            )

            # Fal.ai returns an object, check its type
            status_type = type(status_obj).__name__

            print(f"Fal.ai status type: {status_type}")

            # Handle Completed status
            if status_type == "Completed":
                # Status is completed, now fetch the actual result
                print(f"🎬 Fetching result for completed job {job_id}...")

                result = await self.client.result(
                    model,
                    job_id
                )

                print(f"DEBUG - result type: {type(result)}")
                print(f"DEBUG - result: {result}")

                # Extract result URL (video or image)
                result_url = None

                # First, try dict format (most common with Fal.ai SDK)
                if isinstance(result, dict):
                    # Try images array first (SeeDream V4, Flux, etc.)
                    images_data = result.get('images', [])
                    if isinstance(images_data, list) and len(images_data) > 0:
                        if isinstance(images_data[0], dict):
                            result_url = images_data[0].get('url')
                        else:
                            result_url = str(images_data[0])

                    # Try single image field
                    if not result_url:
                        image_data = result.get('image', {})
                        if isinstance(image_data, dict) and image_data.get('url'):
                            result_url = image_data.get('url')

                    # Try video field
                    if not result_url:
                        video_data = result.get('video', {})
                        if isinstance(video_data, dict) and video_data.get('url'):
                            result_url = video_data.get('url')

                    # Try direct url field
                    if not result_url:
                        result_url = result.get('url')

                # Fallback to object attribute access
                if not result_url:
                    # Try images array (object format)
                    if hasattr(result, 'images') and result.images and len(result.images) > 0:
                        if hasattr(result.images[0], 'url'):
                            result_url = result.images[0].url
                        else:
                            result_url = str(result.images[0])
                    # Try single image
                    elif hasattr(result, 'image') and result.image:
                        if hasattr(result.image, 'url'):
                            result_url = result.image.url
                        else:
                            result_url = str(result.image)
                    # Try video
                    elif hasattr(result, 'video') and result.video:
                        if hasattr(result.video, 'url'):
                            result_url = result.video.url
                        else:
                            result_url = str(result.video)

                print(f"✅ Generation completed! Result URL: {result_url}")

                return {
                    "status": "COMPLETED",
                    "result_url": result_url,
                    "error": None
                }

            # Handle InProgress or InQueue status
            elif status_type in ["InProgress", "InQueue"]:
                print(f"⏳ Generation in progress...")
                return {
                    "status": "PROCESSING",
                    "result_url": None,
                    "error": None
                }

            # Handle Failed status
            elif status_type == "Failed":
                error_msg = "Generation failed"
                if hasattr(status_obj, 'error'):
                    error_msg = str(status_obj.error)
                elif hasattr(status_obj, 'message'):
                    error_msg = str(status_obj.message)

                print(f"❌ Generation failed: {error_msg}")

                return {
                    "status": "FAILED",
                    "result_url": None,
                    "error": error_msg
                }

            # Unknown status
            else:
                print(f"Unknown status type: {status_type}, treating as processing")
                return {
                    "status": "PROCESSING",
                    "result_url": None,
                    "error": None
                }

        except Exception as e:
            # If we can't check status, assume still processing
            print(f"Error checking Fal.ai status: {e}")
            return {
                "status": "PROCESSING",
                "result_url": None,
                "error": None
            }

    # ===== Seedream-compatible interface for Stage 1 & 2 =====
    # These methods provide the same interface as SeedreamProvider
    # so FalProvider can be used as a drop-in replacement

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
        Stage 1: Generate person image from text prompt (Seedream-compatible)
        Uses fal-ai/bytedance/seedream/v4/text-to-image

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
                "job_id": "fal_request_id",
                "provider": "fal",
                "status": "processing"
            }
        """
        try:
            logger.info(f"🎨 [Fal/Seedream] Generating person image...")
            logger.info(f"📝 Prompt: {prompt[:100]}...")

            # Prepare arguments for Fal.ai Seedream model
            fal_arguments = {
                "prompt": prompt,
                "image_size": {
                    "width": width,
                    "height": height
                },
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images": 1,
                "enable_safety_checker": True
            }

            if negative_prompt:
                fal_arguments["negative_prompt"] = negative_prompt

            # Submit job to Fal.ai
            result = await self.client.submit(
                "fal-ai/bytedance/seedream/v4/text-to-image",
                arguments=fal_arguments,
                webhook_url=webhook_url
            )

            logger.info(f"✅ [Fal/Seedream] Job submitted: {result.request_id}")

            return {
                "job_id": result.request_id,
                "provider": "fal",
                "status": "processing",
                "estimated_time": 25
            }

        except Exception as e:
            logger.error(f"❌ [Fal/Seedream] Text-to-image error: {str(e)}")
            raise Exception(f"Fal Seedream text-to-image failed: {str(e)}")

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
        Stage 2: Composite product with person image (Seedream-compatible)
        Uses fal-ai/bytedance/seedream/v4/edit

        Args:
            base_image_url: URL of person image (from Stage 1)
            prompt: Instructions for compositing
            overlay_image_url: Optional product image to composite
            strength: Edit strength (0.0 to 1.0, default: 0.8)
            edit_mode: Not used by Fal.ai, kept for compatibility
            webhook_url: Optional webhook for completion notification

        Returns:
            {
                "job_id": "fal_request_id",
                "provider": "fal",
                "status": "processing"
            }
        """
        try:
            logger.info(f"🎨 [Fal/Seedream] Compositing product with person...")
            logger.info(f"📝 Prompt: {prompt[:100]}...")
            logger.info(f"🖼️  Base image: {base_image_url}")

            # Build image_urls array for Seedream edit
            image_urls = [base_image_url]
            if overlay_image_url:
                image_urls.append(overlay_image_url)
                logger.info(f"🛍️  Product image: {overlay_image_url}")

            # Prepare arguments for Fal.ai Seedream edit
            fal_arguments = {
                "prompt": prompt,
                "image_urls": image_urls,  # Seedream will combine these
                "num_images": 1,
                "enable_safety_checker": True
            }

            # Submit job to Fal.ai
            result = await self.client.submit(
                "fal-ai/bytedance/seedream/v4/edit",
                arguments=fal_arguments,
                webhook_url=webhook_url
            )

            logger.info(f"✅ [Fal/Seedream] Job submitted: {result.request_id}")

            return {
                "job_id": result.request_id,
                "provider": "fal",
                "status": "processing",
                "estimated_time": 30
            }

        except Exception as e:
            logger.error(f"❌ [Fal/Seedream] Image edit error: {str(e)}")
            raise Exception(f"Fal Seedream image-edit failed: {str(e)}")

    async def get_status(self, job_id: str, model: str = "") -> Dict[str, Any]:
        """
        Check status of Fal.ai generation job (Seedream-compatible interface)

        Args:
            job_id: Fal.ai request ID
            model: Model name (auto-detected, kept for compatibility)

        Returns:
            {
                "status": "COMPLETED" | "PROCESSING" | "FAILED",
                "result_url": "https://..." (if completed),
                "error": "Error message" (if failed),
                "progress": 75 (optional)
            }
        """
        # Try to detect model from context, or use Seedream text-to-image as default
        if not model:
            model = "fal-ai/bytedance/seedream/v4/text-to-image"

        return await self.check_status(job_id, model)

    async def wait_for_completion(
        self,
        job_id: str,
        timeout: int = 120,
        poll_interval: int = 5
    ) -> str:
        """
        Poll job status until completion or timeout (Seedream-compatible)

        Args:
            job_id: Fal.ai request ID
            timeout: Maximum wait time in seconds (default: 120)
            poll_interval: Seconds between status checks (default: 5)

        Returns:
            Result URL of generated image/video

        Raises:
            Exception: If job fails or times out
        """
        logger.info(f"⏳ [Fal] Waiting for job {job_id} to complete...")

        start_time = asyncio.get_event_loop().time()

        # Try different models for status checking
        models_to_try = [
            "fal-ai/bytedance/seedream/v4/text-to-image",
            "fal-ai/bytedance/seedream/v4/edit",
        ]

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed > timeout:
                raise Exception(f"Fal.ai job {job_id} timed out after {timeout}s")

            # Try each model until one works
            status_data = None
            for model in models_to_try:
                try:
                    status_data = await self.check_status(job_id, model)
                    break  # If successful, stop trying other models
                except:
                    continue  # Try next model

            if not status_data:
                # If all models failed, wait and retry
                await asyncio.sleep(poll_interval)
                continue

            if status_data["status"] == "COMPLETED":
                if not status_data["result_url"]:
                    raise Exception("Job completed but no result URL found")
                return status_data["result_url"]

            elif status_data["status"] == "FAILED":
                raise Exception(f"Fal.ai job failed: {status_data.get('error', 'Unknown error')}")

            # Still processing, wait and retry
            await asyncio.sleep(poll_interval)
