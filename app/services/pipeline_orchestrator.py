"""
Pipeline Orchestrator for 3-Stage UGC Video Generation

Coordinates the complete workflow:
1. Stage 1: Generate AI person (Fal.ai/Seedream)
2. Stage 2: Composite person + product (Fal.ai/Seedream)
3. Stage 3: Generate video (Veo3 via Kie.ai/Fal.ai)

Handles S3 uploads, progress tracking, and error recovery.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
import httpx

from .providers import FalProvider, Veo3Provider
from .providers.seedream_provider import build_person_prompt_from_fields
from .providers.openrouter_provider import OpenRouterProvider
from .providers.n8n_prompt_builder import N8nPromptBuilder
from app.utils.s3_storage import upload_person_image, upload_composite_image, upload_video
from app.utils.error_classifier import classify_error, ErrorType
from app.core.config import settings

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the complete 3-stage UGC video generation pipeline"""

    def __init__(
        self,
        fal_provider: FalProvider,
        veo3_provider: Veo3Provider,
        openrouter_provider: Optional[OpenRouterProvider] = None,
        webhook_url: Optional[str] = None
    ):
        """
        Initialize pipeline orchestrator

        Args:
            fal_provider: Fal.ai provider for Stage 1 & 2 (Seedream models)
            veo3_provider: Provider for Stage 3 (with fallback)
            openrouter_provider: OpenRouter provider for GPT-4o prompt enhancement (optional)
            webhook_url: Optional webhook URL for progress updates
        """
        self.fal_seedream = fal_provider
        self.veo3 = veo3_provider
        self.openrouter = openrouter_provider
        self.webhook_url = webhook_url or settings.WASP_WEBHOOK_URL

    # ===== INDIVIDUAL STAGE METHODS (for tab-based architecture) =====

    async def generate_person_only(
        self,
        generation_id: str,
        user_id: str,
        stage1_mode: str,  # "EASY" or "ADVANCED"
        person_prompt: Optional[str] = None,
        person_fields: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generate AI person image only (Stage 1 standalone)

        Stage 1 generates ONLY the person - no product included.
        The person will have hand positioned naturally, ready for Stage 2 compositing.

        Args:
            generation_id: UUID of VideoGeneration record
            user_id: UUID of user
            stage1_mode: "EASY" or "ADVANCED"
            person_prompt: Custom prompt (if ADVANCED mode)
            person_fields: Form fields (if EASY mode)

        Returns:
            {
                "success": True,
                "generation_id": "abc-123",
                "person_url": "https://s3.../person-images/...",
                "person_s3_key": "person-images/..."
            }
        """
        logger.info(f"🚀 [Stage 1 Only] Starting person generation for {generation_id}")

        try:
            # Build prompt based on mode using n8n YAML format
            if stage1_mode == "EASY":
                if not person_fields:
                    raise ValueError("person_fields required for EASY mode")

                # Build n8n-style YAML prompt
                final_person_prompt = N8nPromptBuilder.build_person_prompt(
                    age=person_fields.get("age", "20s"),
                    gender=person_fields.get("gender", "female"),
                    ethnicity=person_fields.get("ethnicity", "caucasian"),
                    expression=person_fields.get("expression", "smiling"),
                    clothing=person_fields.get("clothing", "casual"),
                    background=person_fields.get("background", "home")
                )
                logger.info(f"📝 [Stage 1] Using n8n-style YAML prompt (EASY mode)")
            else:
                if not person_prompt:
                    raise ValueError("person_prompt required for ADVANCED mode")
                final_person_prompt = person_prompt
                logger.info(f"📝 [Stage 1] Using custom prompt (ADVANCED mode)")

            logger.info(f"✨ [Stage 1] n8n prompt format - NO GPT-4o enhancement needed")

            # Generate person image
            stage1_result = await self.fal_seedream.submit_text_to_image_job(
                prompt=final_person_prompt,
                width=1080,
                height=1440
            )

            person_image_url = await self.fal_seedream.wait_for_completion(
                job_id=stage1_result["job_id"],
                timeout=120
            )

            logger.info(f"✅ [Stage 1 Only] Person generated: {person_image_url}")

            # Upload to S3
            person_s3 = await upload_person_image(
                url=person_image_url,
                user_id=user_id,
                generation_id=generation_id
            )

            logger.info(f"☁️  [Stage 1 Only] Uploaded to S3: {person_s3['s3_url']}")

            return {
                "success": True,
                "generation_id": generation_id,
                "person_url": person_s3['s3_url'],
                "person_s3_key": person_s3['s3_key']
            }

        except Exception as e:
            logger.error(f"❌ [Stage 1 Only] Failed: {str(e)}")
            # Send failure webhook with error classification
            await self._send_failure_webhook(
                generation_id=generation_id,
                exception=e,
                stage=1,
                stage_error_field="stage1_error"
            )
            raise

    async def generate_composite_only(
        self,
        generation_id: str,
        user_id: str,
        person_image_url: str,
        product_image_url: str,
        composite_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate composite image only (Stage 2 standalone)

        Args:
            generation_id: UUID of VideoGeneration record
            user_id: UUID of user
            person_image_url: S3 URL of person image
            product_image_url: S3 URL of product image
            composite_prompt: Compositing instructions (optional)

        Returns:
            {
                "success": True,
                "generation_id": "abc-123",
                "composite_url": "https://s3.../composites/...",
                "composite_s3_key": "composites/..."
            }
        """
        logger.info(f"🚀 [Stage 2 Only] Starting composite generation for {generation_id}")

        try:
            # Analyze product image if GPT-4o is available (for brand name and description)
            product_info = None
            if self.openrouter:
                try:
                    logger.info(f"🔍 [GPT-4o] Analyzing product image for metadata...")
                    product_info = await self.openrouter.analyze_product_image(product_image_url)
                    logger.info(f"✅ [GPT-4o] Product: {product_info.get('brand_name', 'Unknown')}")
                except Exception as e:
                    logger.warning(f"⚠️  [GPT-4o] Product analysis failed: {str(e)}")

            # Build n8n-style YAML composite prompt (no GPT-4o enhancement needed)
            if not composite_prompt:
                # Use n8n format
                brand_name = product_info.get('brand_name', 'Unknown') if product_info else 'Unknown'
                visual_desc = product_info.get('visual_description', 'product') if product_info else 'product'

                composite_prompt = N8nPromptBuilder.build_composite_prompt(
                    person_description="person from the original image",
                    product_description=visual_desc,
                    brand_name=brand_name
                )
                logger.info(f"📝 [Stage 2] Using n8n-style YAML composite prompt")
            else:
                logger.info(f"📝 [Stage 2] Using custom composite prompt")

            logger.info(f"✨ [Stage 2] n8n prompt format - NO GPT-4o enhancement needed")

            # Composite product with person (preserve lighting and person appearance)
            stage2_result = await self.fal_seedream.submit_image_edit_job(
                base_image_url=person_image_url,
                prompt=composite_prompt,
                overlay_image_url=product_image_url,
                strength=0.45  # Lower strength to preserve original lighting and person appearance
            )

            composite_image_url = await self.fal_seedream.wait_for_completion(
                job_id=stage2_result["job_id"],
                timeout=90
            )

            logger.info(f"✅ [Stage 2 Only] Composite generated: {composite_image_url}")

            # Upload to S3
            composite_s3 = await upload_composite_image(
                url=composite_image_url,
                user_id=user_id,
                generation_id=generation_id
            )

            logger.info(f"☁️  [Stage 2 Only] Uploaded to S3: {composite_s3['s3_url']}")

            return {
                "success": True,
                "generation_id": generation_id,
                "composite_url": composite_s3['s3_url'],
                "composite_s3_key": composite_s3['s3_key']
            }

        except Exception as e:
            logger.error(f"❌ [Stage 2 Only] Failed: {str(e)}")
            # Send failure webhook with error classification
            await self._send_failure_webhook(
                generation_id=generation_id,
                exception=e,
                stage=2,
                stage_error_field="stage2_error"
            )
            raise

    async def generate_video_only(
        self,
        generation_id: str,
        user_id: str,
        composite_image_url: str,
        video_prompt: str,
        veo3_mode: str = "STANDARD",
        duration: int = 8,
        aspect_ratio: str = "9:16",
        product_image_url: Optional[str] = None  # NEW: For GPT-4o enhancement
    ) -> Dict[str, Any]:
        """
        Generate video from composite image only (Stage 3 standalone)

        Args:
            generation_id: UUID of VideoGeneration record
            user_id: UUID of user
            composite_image_url: S3 URL of composite image
            video_prompt: Video animation prompt
            veo3_mode: "FAST" or "STANDARD"
            duration: Video duration in seconds
            aspect_ratio: Video aspect ratio
            product_image_url: Product image URL (optional, for GPT-4o enhancement)

        Returns:
            {
                "success": True,
                "generation_id": "abc-123",
                "video_url": "https://s3.../videos/...",
                "video_s3_key": "videos/...",
                "provider_used": "kie" or "fal",
                "fallback_triggered": True/False
            }
        """
        logger.info(f"🚀 [Stage 3 Only] Starting video generation for {generation_id}")

        try:
            # Analyze product image if GPT-4o is available (for product type and brand name)
            product_info = None
            if self.openrouter and product_image_url:
                try:
                    logger.info(f"🔍 [GPT-4o] Analyzing product image for metadata...")
                    product_info = await self.openrouter.analyze_product_image(product_image_url)
                    logger.info(f"✅ [GPT-4o] Product: {product_info.get('brand_name', 'Unknown')}")
                except Exception as e:
                    logger.warning(f"⚠️  [GPT-4o] Product analysis failed: {str(e)}")

            # Parse video_prompt to extract dialogue, action, emotion
            # Assume format: "dialogue: ...\naction: ...\nemotion: ..."
            # Or if freeform text, treat as dialogue and generate defaults
            dialogue = ""
            action = "character sits holding the product casually while speaking"
            emotion = "casual and happy"

            if "\n" in video_prompt and ":" in video_prompt:
                # Structured prompt - parse it
                for line in video_prompt.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower()
                        value = value.strip()
                        if key == "dialogue":
                            dialogue = value
                        elif key == "action":
                            action = value
                        elif key == "emotion":
                            emotion = value
            else:
                # Freeform prompt - treat as dialogue
                dialogue = video_prompt

            # Generate casual dialogue if not provided
            if not dialogue:
                product_type = product_info.get('type', 'product') if product_info else 'product'
                brand_name = product_info.get('brand_name', '') if product_info else ''
                dialogue = N8nPromptBuilder.generate_casual_dialogue(product_type, brand_name)
                logger.info(f"🗣️  [Stage 3] Generated dialogue: {dialogue}")

            # Build n8n-style YAML video prompt (no GPT-4o enhancement needed)
            final_video_prompt = N8nPromptBuilder.build_video_prompt(
                dialogue=dialogue,
                action=action,
                emotion=emotion,
                character_description="person from the composite image",
                product_type=product_info.get('type', 'product') if product_info else 'product'
            )

            logger.info(f"📝 [Stage 3] Using n8n-style YAML video prompt")
            logger.info(f"✨ [Stage 3] n8n prompt format - NO GPT-4o enhancement needed")

            # Generate video with Veo3
            stage3_result = await self.veo3.submit_veo3_job(
                image_url=composite_image_url,
                prompt=final_video_prompt,
                use_fast=(veo3_mode == "FAST"),
                duration=duration,
                aspect_ratio=aspect_ratio,
                preferred_provider="kie",
                fallback_timeout=60
            )

            provider_used = stage3_result["provider"]
            fallback_triggered = stage3_result.get("fallback_triggered", False)

            logger.info(f"📤 [Stage 3 Only] Job submitted to {provider_used}")

            # Wait for completion
            video_url = await self.veo3.wait_for_completion(
                job_id=stage3_result["job_id"],
                provider=provider_used,
                timeout=600,
                poll_interval=30
            )

            logger.info(f"✅ [Stage 3 Only] Video generated: {video_url}")

            # Upload to S3
            video_s3 = await upload_video(
                url=video_url,
                user_id=user_id,
                generation_id=generation_id
            )

            logger.info(f"☁️  [Stage 3 Only] Uploaded to S3: {video_s3['s3_url']}")

            return {
                "success": True,
                "generation_id": generation_id,
                "video_url": video_s3['s3_url'],
                "video_s3_key": video_s3['s3_key'],
                "provider_used": provider_used,
                "fallback_triggered": fallback_triggered
            }

        except Exception as e:
            logger.error(f"❌ [Stage 3 Only] Failed: {str(e)}")
            # Send failure webhook with error classification
            await self._send_failure_webhook(
                generation_id=generation_id,
                exception=e,
                stage=3,
                stage_error_field="stage3_error"
            )
            raise

    # ===== FULL PIPELINE METHOD (legacy support) =====

    async def generate_ugc_video(
        self,
        generation_id: str,
        user_id: str,
        # Stage 1: Person Generation
        stage1_mode: str,  # "EASY" or "ADVANCED"
        # Stage 2: Compositing
        product_image_url: str,
        # Stage 3: Video Generation
        video_prompt: str,
        # Optional parameters
        person_prompt: Optional[str] = None,  # For ADVANCED mode
        person_fields: Optional[Dict[str, str]] = None,  # For EASY mode
        composite_prompt: Optional[str] = None,
        veo3_mode: str = "STANDARD",  # "FAST" or "STANDARD"
        duration: int = 8,
        aspect_ratio: str = "9:16"
    ) -> Dict[str, Any]:
        """
        Execute complete 3-stage video generation pipeline

        Args:
            generation_id: UUID of VideoGeneration record
            user_id: UUID of user
            stage1_mode: "EASY" or "ADVANCED"
            person_prompt: Custom prompt (if ADVANCED mode)
            person_fields: Form fields (if EASY mode): {gender, age, ethnicity, clothing, expression, background}
            product_image_url: S3 URL of product image
            composite_prompt: Instructions for compositing (optional)
            video_prompt: Prompt for video animation
            veo3_mode: "FAST" (2-3 min) or "STANDARD" (4-6 min)
            duration: Video duration in seconds
            aspect_ratio: Video aspect ratio

        Returns:
            {
                "success": True,
                "generation_id": "abc-123",
                "person_url": "https://s3.../person-images/...",
                "composite_url": "https://s3.../composites/...",
                "video_url": "https://s3.../videos/...",
                "provider_used": "kie" or "fal",
                "fallback_triggered": True/False,
                "total_time": 245.3
            }
        """
        import time
        start_time = time.time()

        logger.info(f"🚀 [Pipeline] Starting UGC video generation for {generation_id}")
        logger.info(f"👤 User: {user_id}")
        logger.info(f"📋 Mode: {stage1_mode}, Veo3: {veo3_mode}")

        try:
            # ===== STAGE 1: Person Generation =====
            logger.info(f"\n{'='*60}")
            logger.info(f"🎨 STAGE 1: Person Generation")
            logger.info(f"{'='*60}\n")

            await self._send_webhook({
                "generation_id": generation_id,
                "stage": 1,
                "status": "processing",
                "message": "Generating AI person..."
            })

            # Build prompt based on mode
            if stage1_mode == "EASY":
                if not person_fields:
                    raise ValueError("person_fields required for EASY mode")

                final_person_prompt = build_person_prompt_from_fields(
                    gender=person_fields.get("gender", "female"),
                    age=person_fields.get("age", "26-35"),
                    ethnicity=person_fields.get("ethnicity", "Caucasian"),
                    clothing=person_fields.get("clothing", "casual"),
                    expression=person_fields.get("expression", "smiling"),
                    background=person_fields.get("background", "white")
                )
            else:
                # ADVANCED mode
                if not person_prompt:
                    raise ValueError("person_prompt required for ADVANCED mode")
                final_person_prompt = person_prompt

            # [GPT-4o Enhancement] Always enhance prompts for better UGC style
            product_info = None
            if self.openrouter:
                try:
                    # Analyze product image if provided (optional)
                    if product_image_url:
                        logger.info(f"🔍 [GPT-4o] Analyzing product image...")
                        product_info = await self.openrouter.analyze_product_image(product_image_url)
                        logger.info(f"✅ [GPT-4o] Product: {product_info.get('brand_name', 'Unknown')}")
                    else:
                        logger.info(f"📝 [GPT-4o] No product image provided, using generic UGC enhancement")
                        product_info = {
                            "brand_name": "Unknown",
                            "color_scheme": [],
                            "visual_description": "Generic product"
                        }

                    # Always enhance prompt (with or without product info)
                    logger.info(f"✨ [GPT-4o] Enhancing person prompt with UGC style...")
                    final_person_prompt = await self.openrouter.enhance_person_prompt(
                        user_input=final_person_prompt,
                        product_info=product_info
                    )
                    logger.info(f"✅ [GPT-4o] Person prompt enhanced")
                except Exception as e:
                    logger.warning(f"⚠️  [GPT-4o] Enhancement failed: {str(e)}, using base UGC prompt")
            else:
                logger.info(f"⚠️  [GPT-4o] OpenRouter not configured, using base UGC prompt")

            # Generate person image using Fal.ai Seedream model
            stage1_result = await self.fal_seedream.submit_text_to_image_job(
                prompt=final_person_prompt,
                width=1080,
                height=1440
            )

            # Wait for completion
            person_image_url = await self.fal_seedream.wait_for_completion(
                job_id=stage1_result["job_id"],
                timeout=120
            )

            logger.info(f"✅ [Stage 1] Person image generated: {person_image_url}")

            # Upload to S3
            person_s3 = await upload_person_image(
                url=person_image_url,
                user_id=user_id,
                generation_id=generation_id
            )

            logger.info(f"☁️  [Stage 1] Uploaded to S3: {person_s3['s3_url']}")

            await self._send_webhook({
                "generation_id": generation_id,
                "stage": 1,
                "status": "completed",
                "person_url": person_s3['s3_url']
            })

            # ===== STAGE 2: Compositing =====
            logger.info(f"\n{'='*60}")
            logger.info(f"🎨 STAGE 2: Compositing")
            logger.info(f"{'='*60}\n")

            await self._send_webhook({
                "generation_id": generation_id,
                "stage": 2,
                "status": "processing",
                "message": "Compositing product with person..."
            })

            # Build composite prompt if not provided (natural product presentation)
            if not composite_prompt:
                composite_prompt = (
                    "The person is casually presenting the product to the camera in a natural, relaxed way. "
                    "The product is held at a comfortable distance from the camera - not too close or too far. "
                    "Natural hand grip with fingers relaxed, as if showing it to a friend. "
                    "The product should appear normal-sized, not oversized or dominant in the frame. "
                    "Keep the person's face, expression, and overall pose similar to the original. "
                    "Casual, authentic UGC presentation style - not staged or overly posed. "
                    "The person looks natural and comfortable holding the product."
                )

            # [GPT-4o Enhancement] Enhance composite prompt
            if self.openrouter and product_info:
                try:
                    logger.info(f"✨ [GPT-4o] Enhancing composite prompt...")
                    composite_prompt = await self.openrouter.enhance_composite_prompt(
                        person_info={'url': person_s3['s3_url']},
                        product_info=product_info,
                        user_instructions=composite_prompt
                    )
                    logger.info(f"✅ [GPT-4o] Composite prompt enhanced")
                except Exception as e:
                    logger.warning(f"⚠️  [GPT-4o] Enhancement failed: {str(e)}, using original prompt")

            # Composite product with person using Fal.ai Seedream edit
            stage2_result = await self.fal_seedream.submit_image_edit_job(
                base_image_url=person_s3['s3_url'],
                prompt=composite_prompt,
                overlay_image_url=product_image_url,
                strength=0.45  # Lower strength to preserve original lighting and person appearance
            )

            # Wait for completion
            composite_image_url = await self.fal_seedream.wait_for_completion(
                job_id=stage2_result["job_id"],
                timeout=90
            )

            logger.info(f"✅ [Stage 2] Composite generated: {composite_image_url}")

            # Upload to S3
            composite_s3 = await upload_composite_image(
                url=composite_image_url,
                user_id=user_id,
                generation_id=generation_id
            )

            logger.info(f"☁️  [Stage 2] Uploaded to S3: {composite_s3['s3_url']}")

            await self._send_webhook({
                "generation_id": generation_id,
                "stage": 2,
                "status": "completed",
                "composite_url": composite_s3['s3_url']
            })

            # ===== STAGE 3: Video Generation =====
            logger.info(f"\n{'='*60}")
            logger.info(f"🎥 STAGE 3: Video Generation (Veo3)")
            logger.info(f"{'='*60}\n")

            await self._send_webhook({
                "generation_id": generation_id,
                "stage": 3,
                "status": "processing",
                "message": f"Generating video ({'fast' if veo3_mode == 'FAST' else 'standard'} mode)..."
            })

            # [GPT-4o Enhancement] Enhance video prompt
            final_video_prompt = video_prompt
            if self.openrouter and product_info:
                try:
                    logger.info(f"✨ [GPT-4o] Enhancing video prompt...")
                    final_video_prompt = await self.openrouter.enhance_video_prompt(
                        composite_info={'url': composite_s3['s3_url']},
                        user_input=video_prompt,
                        product_type=product_info.get('brand_name', 'product')
                    )
                    logger.info(f"✅ [GPT-4o] Video prompt enhanced")
                except Exception as e:
                    logger.warning(f"⚠️  [GPT-4o] Enhancement failed: {str(e)}, using original prompt")

            # Generate video with Veo3 (automatic fallback)
            stage3_result = await self.veo3.submit_veo3_job(
                image_url=composite_s3['s3_url'],
                prompt=final_video_prompt,
                use_fast=(veo3_mode == "FAST"),
                duration=duration,
                aspect_ratio=aspect_ratio,
                preferred_provider="kie",
                fallback_timeout=60
            )

            provider_used = stage3_result["provider"]
            fallback_triggered = stage3_result.get("fallback_triggered", False)

            logger.info(f"📤 [Stage 3] Job submitted to {provider_used}")
            if fallback_triggered:
                logger.warning(f"⚠️  [Stage 3] Fallback was triggered!")

            # Wait for completion (may take 2-6 minutes)
            video_url = await self.veo3.wait_for_completion(
                job_id=stage3_result["job_id"],
                provider=provider_used,
                timeout=600,  # 10 minutes max
                poll_interval=30
            )

            logger.info(f"✅ [Stage 3] Video generated: {video_url}")

            # Upload to S3
            video_s3 = await upload_video(
                url=video_url,
                user_id=user_id,
                generation_id=generation_id
            )

            logger.info(f"☁️  [Stage 3] Uploaded to S3: {video_s3['s3_url']}")

            # Final webhook with completion
            total_time = time.time() - start_time

            await self._send_webhook({
                "generation_id": generation_id,
                "stage": 3,
                "status": "completed",
                "video_url": video_s3['s3_url'],
                "provider_used": provider_used,
                "fallback_triggered": fallback_triggered,
                "total_time": total_time
            })

            logger.info(f"\n{'='*60}")
            logger.info(f"🎉 PIPELINE COMPLETE!")
            logger.info(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
            logger.info(f"{'='*60}\n")

            return {
                "success": True,
                "generation_id": generation_id,
                "person_url": person_s3['s3_url'],
                "person_s3_key": person_s3['s3_key'],
                "composite_url": composite_s3['s3_url'],
                "composite_s3_key": composite_s3['s3_key'],
                "video_url": video_s3['s3_url'],
                "video_s3_key": video_s3['s3_key'],
                "provider_used": provider_used,
                "fallback_triggered": fallback_triggered,
                "total_time": total_time
            }

        except Exception as e:
            logger.error(f"\n❌ [Pipeline] FAILED: {str(e)}")

            # Send failure webhook with error classification
            await self._send_failure_webhook(
                generation_id=generation_id,
                exception=e,
                stage=None  # Full pipeline - no specific stage
            )

            raise

    async def generate_person_and_composite(
        self,
        generation_id: str,
        user_id: str,
        stage1_mode: str,
        person_prompt: Optional[str],
        person_fields: Optional[Dict[str, str]],
        product_image_url: str,
        composite_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """
        TEST METHOD: Generate only Stages 1-2 (Person + Composite)

        Skips Stage 3 (video generation) for cost-effective testing.

        Args:
            generation_id: Unique ID for this generation
            user_id: User ID
            stage1_mode: "EASY" or "ADVANCED"
            person_prompt: Custom prompt (ADVANCED mode)
            person_fields: Person attributes (EASY mode)
            product_image_url: URL of product image
            composite_prompt: Optional compositing instructions

        Returns:
            Dict with person_url, composite_url, and timing info
        """
        start_time = time.time()

        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 TEST: Starting Stages 1-2 only")
            logger.info(f"{'='*60}")
            logger.info(f"Generation ID: {generation_id}")
            logger.info(f"Mode: {stage1_mode}")

            # STAGE 1: Generate Person
            logger.info(f"\n🎨 [Stage 1/2] Generating AI person...")

            if stage1_mode == "EASY":
                person_prompt = build_person_prompt_from_fields(**person_fields)
                logger.info(f"📝 Easy Mode - Built prompt from fields")
            else:
                logger.info(f"📝 Advanced Mode - Using custom prompt")

            # [GPT-4o Enhancement] Analyze product and enhance person prompt if OpenRouter is available
            product_info = None
            if self.openrouter and product_image_url:
                try:
                    logger.info(f"🔍 Analyzing product image with GPT-4o...")
                    product_info = await self.openrouter.analyze_product_image(product_image_url)

                    logger.info(f"✨ Enhancing person prompt with GPT-4o...")
                    person_prompt = await self.openrouter.enhance_person_prompt(
                        user_input=person_prompt,
                        product_info=product_info
                    )
                    logger.info(f"✅ Person prompt enhanced")
                except Exception as e:
                    logger.warning(f"⚠️ GPT-4o enhancement failed, using original prompt: {str(e)}")

            # Submit person generation job
            person_job = await self.fal_seedream.submit_text_to_image_job(
                prompt=person_prompt,
                negative_prompt="blurry, low quality, distorted",
                width=1080,
                height=1440,
                num_inference_steps=50,
                guidance_scale=7.5
            )
            logger.info(f"✅ Person job submitted: {person_job['job_id']}")

            # Poll for completion
            person_image_url = await self.fal_seedream.wait_for_completion(
                job_id=person_job['job_id'],
                timeout=120
            )
            logger.info(f"✅ Person generated: {person_image_url}")

            # Upload to S3
            person_s3 = await upload_person_image(
                url=person_image_url,
                user_id=user_id,
                generation_id=generation_id
            )
            logger.info(f"📤 Person uploaded to S3: {person_s3['s3_key']}")

            # Send webhook for Stage 1
            await self._send_webhook({
                "generation_id": generation_id,
                "status": "processing",
                "current_stage": 1,
                "progress": 50,
                "generated_person_url": person_s3['s3_url'],
                "s3_key_person": person_s3['s3_key']
            })

            # STAGE 2: Composite Product
            logger.info(f"\n🎨 [Stage 2/2] Compositing product with person...")

            composite_prompt_text = composite_prompt or "product held naturally by person"

            # [GPT-4o Enhancement] Enhance composite prompt if OpenRouter is available
            if self.openrouter and product_info:
                try:
                    logger.info(f"✨ Enhancing composite prompt with GPT-4o...")
                    composite_prompt_text = await self.openrouter.enhance_composite_prompt(
                        person_info={'url': person_s3['s3_url']},
                        product_info=product_info,
                        user_instructions=composite_prompt or ""
                    )
                    logger.info(f"✅ Composite prompt enhanced")
                except Exception as e:
                    logger.warning(f"⚠️ GPT-4o enhancement failed, using original prompt: {str(e)}")

            composite_job = await self.fal_seedream.submit_image_edit_job(
                base_image_url=person_s3['s3_url'],
                prompt=composite_prompt_text,
                overlay_image_url=product_image_url,
                strength=0.45,  # Lower strength to preserve original lighting and person appearance
                edit_mode="inpaint_and_blend"
            )
            logger.info(f"✅ Composite job submitted: {composite_job['job_id']}")

            # Poll for completion
            composite_image_url = await self.fal_seedream.wait_for_completion(
                job_id=composite_job['job_id'],
                timeout=120
            )
            logger.info(f"✅ Composite generated: {composite_image_url}")

            # Upload to S3
            composite_s3 = await upload_composite_image(
                url=composite_image_url,
                user_id=user_id,
                generation_id=generation_id
            )
            logger.info(f"📤 Composite uploaded to S3: {composite_s3['s3_key']}")

            # Send webhook for Stage 2
            await self._send_webhook({
                "generation_id": generation_id,
                "status": "completed",  # Mark as complete since we're skipping stage 3
                "current_stage": 2,
                "progress": 100,
                "composite_image_url": composite_s3['s3_url'],
                "s3_key_composite": composite_s3['s3_key']
            })

            total_time = time.time() - start_time
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ TEST COMPLETE: Stages 1-2 successful!")
            logger.info(f"{'='*60}")
            logger.info(f"Total Time: {total_time:.1f}s")
            logger.info(f"{'='*60}\n")

            return {
                "success": True,
                "generation_id": generation_id,
                "person_url": person_s3['s3_url'],
                "person_s3_key": person_s3['s3_key'],
                "composite_url": composite_s3['s3_url'],
                "composite_s3_key": composite_s3['s3_key'],
                "total_time": total_time
            }

        except Exception as e:
            logger.error(f"\n❌ TEST FAILED: Stages 1-2 - {str(e)}")

            # Send failure webhook with error classification
            await self._send_failure_webhook(
                generation_id=generation_id,
                exception=e,
                stage=None  # Test method - no specific stage
            )

            raise

    async def _send_webhook(self, data: Dict[str, Any]):
        """
        Send webhook notification to Wasp backend

        Args:
            data: Webhook payload (includes generation_id, stage, status, etc.)
        """
        if not self.webhook_url:
            logger.warning("⚠️  No webhook URL configured, skipping notification")
            return

        try:
            # Add authorization header
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.PYTHON_API_KEY}"
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.webhook_url,
                    json=data,
                    headers=headers
                )

                if response.status_code == 200:
                    logger.info(f"📡 Webhook sent: stage={data.get('stage')}, status={data.get('status')}")
                else:
                    logger.warning(f"⚠️  Webhook failed: {response.status_code}")

        except Exception as e:
            logger.error(f"❌ Webhook error: {str(e)}")
            # Don't raise - webhook failures shouldn't stop the pipeline

    async def _send_failure_webhook(
        self,
        generation_id: str,
        exception: Exception,
        stage: Optional[int] = None,
        stage_error_field: Optional[str] = None
    ):
        """
        Send failure webhook with error classification

        Args:
            generation_id: UUID of generation
            exception: The exception that occurred
            stage: Optional stage number (1, 2, or 3)
            stage_error_field: Optional stage error field name (e.g., "stage1_error")
        """
        # Classify the error
        error_info = classify_error(exception, context=f"stage{stage}" if stage else None)

        # Build webhook payload
        payload = {
            "generation_id": generation_id,
            "status": "FAILED",
            "error_type": error_info["error_type"],
            "error_message": error_info["error_message"],
            "is_refundable": error_info["is_refundable"],
            "can_retry": error_info["can_retry"],
        }

        # Add stage information if provided
        if stage:
            payload["current_stage"] = stage
            payload["progress"] = 0

        # Add stage-specific error details
        if stage_error_field:
            payload[stage_error_field] = error_info["technical_details"]

        # Send webhook
        await self._send_webhook(payload)
        logger.info(f"📡 Failure webhook sent: {error_info['error_type']} (refundable={error_info['is_refundable']})")


# Singleton instance (initialized in main.py)
_orchestrator: Optional[PipelineOrchestrator] = None


def init_pipeline_orchestrator(
    fal_provider: FalProvider,
    veo3_provider: Veo3Provider,
    openrouter_provider: Optional[OpenRouterProvider] = None
) -> PipelineOrchestrator:
    """
    Initialize pipeline orchestrator singleton

    Args:
        fal_provider: Fal.ai provider instance (for Seedream models)
        veo3_provider: Veo3 provider instance
        openrouter_provider: OpenRouter provider instance (optional, for GPT-4o prompt enhancement)

    Returns:
        PipelineOrchestrator instance
    """
    global _orchestrator
    _orchestrator = PipelineOrchestrator(
        fal_provider=fal_provider,
        veo3_provider=veo3_provider,
        openrouter_provider=openrouter_provider
    )
    return _orchestrator


def get_pipeline_orchestrator() -> PipelineOrchestrator:
    """
    Get pipeline orchestrator singleton

    Returns:
        PipelineOrchestrator instance

    Raises:
        RuntimeError: If orchestrator not initialized
    """
    if _orchestrator is None:
        raise RuntimeError(
            "Pipeline orchestrator not initialized. "
            "Call init_pipeline_orchestrator() first."
        )
    return _orchestrator
