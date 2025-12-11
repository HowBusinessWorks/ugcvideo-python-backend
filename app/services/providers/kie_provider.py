import httpx
import json
from typing import Dict, Any
from .base import BaseProvider
from app.core.config import settings


class KieProvider(BaseProvider):
    """
    Kie.ai provider implementation

    Cheaper and potentially faster alternative to Fal.ai
    Cost: ~$0.0175 per image
    """

    def __init__(self):
        self.api_key = settings.KIE_API_KEY

        self.base_url = "https://api.kie.ai/api/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def format_prompt(self, user_prompt: str, system_prompt: str) -> str:
        """
        Format prompt for Kie.ai
        Combines system and user prompts
        """
        if system_prompt:
            return f"{system_prompt}\n\n{user_prompt}"
        return user_prompt

    async def submit_job(self, request: Any) -> Dict[str, Any]:
        """
        Submit text-to-video generation job (NOT SUPPORTED by Kie.ai currently)

        Kie.ai supports image-to-video but not text-to-video.
        For text-to-video, the system will use Fal.ai instead.
        """
        raise NotImplementedError(
            "Kie.ai provider does not support text-to-video generation. "
            "Use Fal.ai for text-to-video (wan-t2v model)."
        )

    async def submit_image_job(self, request: Any) -> Dict[str, Any]:
        """
        Submit image editing job to Kie.ai (SeeDream V4)

        API: POST /api/v1/jobs/createTask

        Args:
            request: GenerateImageRequest with image_urls array

        Returns:
            Dict with job_id and estimated_time
        """
        # Format the full prompt
        full_prompt = self.format_prompt(request.prompt, request.system_prompt)

        # Extract parameters with defaults
        image_size = request.parameters.get("image_size", "square_hd")
        image_resolution = request.parameters.get("image_resolution", "1K")
        max_images = request.parameters.get("max_images", 1)
        seed = request.parameters.get("seed")

        # Build Kie.ai API request following their exact structure
        payload = {
            "model": "bytedance/seedream-v4-edit",
            "input": {
                "prompt": full_prompt,
                "image_urls": request.image_urls,
                "image_size": image_size,
                "image_resolution": image_resolution,
                "max_images": max_images
            }
        }

        # Add optional seed if provided
        if seed is not None:
            payload["input"]["seed"] = seed

        # Add callback URL if provided (Kie.ai calls it "callBackUrl")
        if hasattr(request, 'webhook_url') and request.webhook_url:
            payload["callBackUrl"] = request.webhook_url

        print(f"🚀 Submitting to Kie.ai: {json.dumps(payload, indent=2)}")

        # Submit to Kie.ai
        async with httpx.AsyncClient(timeout=30.0) as client:
            endpoint = f"{self.base_url}/jobs/createTask"

            response = await client.post(
                endpoint,
                headers=self.headers,
                json=payload
            )

            print(f"📥 Kie.ai response status: {response.status_code}")
            print(f"📥 Kie.ai response: {response.text}")

            if response.status_code != 200:
                error_detail = response.text
                raise Exception(f"Kie.ai API error ({response.status_code}): {error_detail}")

            result = response.json()

            # Kie.ai response format: { code: 200, message: "success", data: { taskId: "..." } }
            if result.get("code") != 200:
                error_msg = result.get("message", "Unknown error")
                raise Exception(f"Kie.ai error: {error_msg}")

            task_id = result.get("data", {}).get("taskId")

            if not task_id:
                raise Exception(f"No taskId in Kie.ai response: {result}")

            print(f"✅ Kie.ai task created: {task_id}")

            return {
                "job_id": task_id,
                "estimated_time": 30  # Kie.ai image editing is fast (typically 8-15 seconds)
            }

    async def submit_image_to_video_job(self, request: Any) -> Dict[str, Any]:
        """
        Submit image-to-video generation job to Kie.ai (Veo 3.1)

        API: POST /api/v1/veo/generate

        Args:
            request: GenerateImageToVideoRequest with image_url and prompt

        Returns:
            Dict with job_id and estimated_time
        """
        # Format the full prompt
        full_prompt = self.format_prompt(request.prompt, request.system_prompt)

        # Extract parameters with defaults
        use_fast = request.parameters.get("use_fast", True)  # Use veo3_fast by default
        aspect_ratio = request.parameters.get("aspect_ratio", "9:16")  # Default portrait
        enable_translation = request.parameters.get("enable_translation", True)

        # Build Kie.ai API request for Veo 3.1 image-to-video
        # See: https://docs.kie.ai/api-reference/veo3-api/generate-veo-3-video
        payload = {
            "prompt": full_prompt,
            "imageUrls": [request.image_url],  # Veo3 uses imageUrls array (1 or 2 images)
            "model": "veo3_fast" if use_fast else "veo3",  # veo3_fast (2-3 min) or veo3 (4-6 min)
            "aspectRatio": aspect_ratio,  # "16:9" or "9:16"
            "generationType": "FIRST_AND_LAST_FRAMES_2_VIDEO",  # Image-to-video mode
            "enableTranslation": enable_translation  # Auto-translate prompts to English
        }

        # Add callback URL if provided
        if hasattr(request, 'webhook_url') and request.webhook_url:
            payload["callBackUrl"] = request.webhook_url

        print(f"🚀 Submitting to Kie.ai (Veo 3.1): {json.dumps(payload, indent=2)}")

        # Submit to Kie.ai
        async with httpx.AsyncClient(timeout=30.0) as client:
            endpoint = f"{self.base_url}/veo/generate"  # NEW Veo3 endpoint

            response = await client.post(
                endpoint,
                headers=self.headers,
                json=payload
            )

            print(f"📥 Kie.ai response status: {response.status_code}")
            print(f"📥 Kie.ai response: {response.text}")

            if response.status_code != 200:
                error_detail = response.text
                raise Exception(f"Kie.ai API error ({response.status_code}): {error_detail}")

            result = response.json()

            # Kie.ai response format: { code: 200, msg: "success", data: { taskId: "..." } }
            if result.get("code") != 200:
                error_msg = result.get("msg", "Unknown error")
                raise Exception(f"Kie.ai error: {error_msg}")

            task_id = result.get("data", {}).get("taskId")

            if not task_id:
                raise Exception(f"No taskId in Kie.ai response: {result}")

            print(f"✅ Kie.ai Veo3 task created: {task_id}")

            return {
                "job_id": task_id,
                "estimated_time": 120 if use_fast else 240  # veo3_fast: 2 min, veo3: 4 min
            }

    async def check_status(self, job_id: str, model: str = "seedream-v4-edit") -> Dict[str, Any]:
        """
        Check the status of a Kie.ai job

        API:
        - Veo3: GET /api/v1/veo/record-info?taskId=xxx
        - Other: GET /api/v1/jobs/recordInfo?taskId=xxx

        Args:
            job_id: The taskId from Kie.ai
            model: Model identifier (used to determine which endpoint to use)

        Returns:
            Dict with status and result_url if completed
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Veo3 tasks use a different endpoint than other jobs
            # Detect if this is a Veo3 task by checking if model contains "veo"
            is_veo3_task = "veo" in model.lower()

            if is_veo3_task:
                # Veo3 endpoint: GET /api/v1/veo/record-info?taskId=xxx
                endpoint = f"{self.base_url}/veo/record-info"
            else:
                # Legacy endpoint for Seedream and other jobs: GET /api/v1/jobs/recordInfo?taskId=xxx
                endpoint = f"{self.base_url}/jobs/recordInfo"

            response = await client.get(
                endpoint,
                headers=self.headers,
                params={"taskId": job_id}
            )

            print(f"🔍 Kie.ai status check ({endpoint}) response ({response.status_code}): {response.text}")

            if response.status_code != 200:
                return {
                    "status": "FAILED",
                    "result_url": None,
                    "error": f"Kie.ai status check failed: {response.text}"
                }

            result = response.json()

            # Kie.ai response format: { code: 200, message: "success", data: { state: "...", resultJson: "..." } }
            if result.get("code") != 200:
                error_msg = result.get("message", "Unknown error")
                return {
                    "status": "FAILED",
                    "result_url": None,
                    "error": error_msg
                }

            data = result.get("data", {})

            # Handle two different response formats:
            # 1. Old format (Seedream/Hailuo): uses "state" field
            # 2. New format (Veo3): uses "successFlag" field

            # Check if this is the new Veo3 format
            if "successFlag" in data:
                # Veo3 format: { successFlag: 1, resultUrls: [...], ... }
                success_flag = data.get("successFlag")

                if success_flag == 1:
                    # Success! Extract result URL from response.resultUrls
                    response_data = data.get("response", {})
                    result_urls = response_data.get("resultUrls", [])

                    if result_urls and len(result_urls) > 0:
                        result_url = result_urls[0]
                    else:
                        result_url = None

                    print(f"✅ Kie.ai Veo3 generation completed! Result URL: {result_url}")

                    return {
                        "status": "COMPLETED",
                        "result_url": result_url
                    }
                elif success_flag == 0:
                    # Check if it's failed or still processing
                    error_code = data.get("errorCode")
                    if error_code:
                        error_msg = data.get("errorMessage", "Unknown error")
                        print(f"❌ Kie.ai Veo3 task failed: {error_msg}")
                        return {
                            "status": "FAILED",
                            "result_url": None,
                            "error": error_msg
                        }
                    else:
                        # Still processing
                        print(f"⏳ Kie.ai Veo3 task processing...")
                        return {
                            "status": "PROCESSING",
                            "result_url": None
                        }
                else:
                    # Unknown successFlag value
                    print(f"⚠️ Unknown Kie.ai successFlag: {success_flag}")
                    return {
                        "status": "PROCESSING",
                        "result_url": None
                    }

            else:
                # Old format: uses "state" field
                state = data.get("state", "").lower()

                # Map Kie.ai states: waiting, queuing, generating, success, fail
                if state == "success":
                    # Extract result URL from resultJson string
                    result_json_str = data.get("resultJson", "{}")

                    try:
                        result_data = json.loads(result_json_str)
                        result_urls = result_data.get("resultUrls", [])

                        if result_urls and len(result_urls) > 0:
                            result_url = result_urls[0]  # Take first image
                        else:
                            result_url = None

                    except json.JSONDecodeError as e:
                        print(f"❌ Failed to parse resultJson: {e}")
                        result_url = None

                    print(f"✅ Kie.ai generation completed! Result URL: {result_url}")

                    return {
                        "status": "COMPLETED",
                        "result_url": result_url
                    }

                elif state in ["waiting", "queuing", "generating"]:
                    print(f"⏳ Kie.ai task {state}...")
                    return {
                        "status": "PROCESSING",
                        "result_url": None
                    }

                elif state == "fail":
                    fail_code = data.get("failCode", "")
                    fail_msg = data.get("failMsg", "Unknown error")
                    error_msg = f"{fail_code}: {fail_msg}" if fail_code else fail_msg

                    print(f"❌ Kie.ai task failed: {error_msg}")

                    return {
                        "status": "FAILED",
                        "result_url": None,
                        "error": error_msg
                    }

                else:
                    # Unknown state, treat as processing
                    print(f"⚠️ Unknown Kie.ai state: {state}")
                    return {
                        "status": "PROCESSING",
                        "result_url": None
                    }

    async def get_status(self, job_id: str, model: str = "") -> Dict[str, Any]:
        """
        Alias for check_status to maintain compatibility with other providers

        Args:
            job_id: The taskId from Kie.ai
            model: Model identifier (not used, kept for compatibility)

        Returns:
            Dict with status and result_url if completed
        """
        return await self.check_status(job_id, model)
