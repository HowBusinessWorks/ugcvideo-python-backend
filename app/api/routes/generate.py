from fastapi import APIRouter, HTTPException
from app.models.request import (
    GenerateVideoRequest,
    GenerateImageToVideoRequest,
    GenerateImageRequest,
    GenerateTextToImageRequest,
    GenerateUGCVideoRequest,
    GeneratePersonRequest,
    GenerateCompositeRequest,
    GenerateVideoOnlyRequest
)
from app.models.response import GenerateResponse, ErrorResponse, GenerateUGCVideoResponse
from app.services.providers.fal_provider import FalProvider
from app.services.providers.kie_provider import KieProvider
from app.services.providers.orchestrator import ProviderOrchestrator
from app.services.pipeline_orchestrator import get_pipeline_orchestrator

router = APIRouter()

# Initialize providers
fal_provider = FalProvider()
kie_provider = KieProvider()  # For testing Kie.ai directly
orchestrator = ProviderOrchestrator()


@router.post("/generate/video", response_model=GenerateResponse)
async def generate_video(request: GenerateVideoRequest):
    """
    Generate video using AI model

    For MVP, we only support Fal.ai's Wan T2V model
    """
    try:
        # Validate model (for MVP, only wan-t2v is supported)
        if request.model != "wan-t2v":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {request.model}. Only 'wan-t2v' is supported."
            )

        # Submit job to Fal.ai
        result = await fal_provider.submit_job(request)

        # Return response
        return GenerateResponse(
            success=True,
            job_id=result["job_id"],
            provider="fal",
            estimated_time_seconds=result["estimated_time"],
            generation_id=request.generation_id
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Log and return error
        print(f"Error generating video: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate video: {str(e)}"
        )


@router.post("/generate/image-to-video", response_model=GenerateResponse)
async def generate_image_to_video(request: GenerateImageToVideoRequest):
    """
    Generate video from image using Minimax Hailuo I2V model with smart provider fallback

    Tries Kie.ai first (cheaper), falls back to Fal.ai if timeout or error
    Requires a reference image URL (from S3) and a motion description prompt
    """
    try:
        # Validate model (only minimax-hailuo-i2v is supported for I2V)
        if request.model != "minimax-hailuo-i2v":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {request.model}. Only 'minimax-hailuo-i2v' is supported."
            )

        # Check if fallback is disabled (same provider for both)
        if request.preferred_provider == request.fallback_provider:
            # Testing mode - call provider directly without fallback
            print(f"🎬 Testing mode: calling {request.preferred_provider} directly (no fallback)")
            provider = orchestrator._get_provider(request.preferred_provider)
            result = await provider.submit_image_to_video_job(request)

            return GenerateResponse(
                success=True,
                job_id=result["job_id"],
                provider=request.preferred_provider,
                estimated_time_seconds=result["estimated_time"],
                generation_id=request.generation_id,
                fallback_triggered=False
            )

        # Use orchestrator for smart fallback
        print(f"🎬 Starting image-to-video generation with primary: {request.preferred_provider}")
        result = await orchestrator.submit_image_to_video_job_with_fallback(
            request=request,
            primary_provider=request.preferred_provider,
            fallback_provider=request.fallback_provider,
            timeout_seconds=request.fallback_timeout
        )

        # Return response with provider info
        return GenerateResponse(
            success=True,
            job_id=result["job_id"],
            provider=result["provider_used"],  # Which provider actually succeeded
            estimated_time_seconds=result["estimated_time"],
            generation_id=request.generation_id,
            fallback_triggered=result.get("fallback_triggered", False),
            primary_provider=result.get("primary_provider")
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Log and return error
        print(f"Error generating image-to-video: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate image-to-video: {str(e)}"
        )


@router.post("/generate/image", response_model=GenerateResponse)
async def generate_image(request: GenerateImageRequest):
    """
    Generate/edit image using SeeDream V4 (Fal.ai only, no fallback)

    Phase J: Photo Enhancer uses Fal.ai directly (no provider fallback)
    """
    try:
        # Validate model (only seedream-v4 is supported for image editing)
        if request.model != "seedream-v4":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {request.model}. Only 'seedream-v4' is supported."
            )

        # Photo Enhancer uses Fal.ai directly (no fallback needed)
        print("🎨 Photo Enhancer: Using Fal.ai directly (no fallback)")
        result = await fal_provider.submit_image_job(request)

        # Return response
        return GenerateResponse(
            success=True,
            job_id=result["job_id"],
            provider="fal",
            estimated_time_seconds=result["estimated_time"],
            generation_id=request.generation_id,
            fallback_triggered=False
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Log and return error
        print(f"Error generating image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate image: {str(e)}"
        )


@router.post("/generate/text-to-image", response_model=GenerateResponse)
async def generate_text_to_image(request: GenerateTextToImageRequest):
    """
    Generate image from text prompt only (SeeDream V4 Text-to-Image)
    No reference images required - pure text-to-image generation

    Uses Fal.ai directly (no provider fallback)
    """
    try:
        # Validate model
        if request.model not in ['seedream-v4-text', 'seedream-v4-t2i']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {request.model}. Only 'seedream-v4-text' is supported."
            )

        # Use Fal.ai directly for text-to-image
        print(f"🎨 Generating text-to-image with Fal.ai: {request.generation_id}")
        result = await fal_provider.submit_text_to_image_job(request)

        # Return response
        return GenerateResponse(
            success=True,
            job_id=result["job_id"],
            provider="fal",
            estimated_time_seconds=result["estimated_time"],
            generation_id=request.generation_id,
            fallback_triggered=False
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Log and return error
        print(f"Error generating text-to-image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate text-to-image: {str(e)}"
        )


@router.get("/generate/status/{job_id}")
async def check_generation_status(
    job_id: str,
    model: str = "fal-ai/wan-t2v",
    provider: str = "fal"
):
    """
    Check the status of a generation job on any provider

    This is used as a fallback when webhooks don't work (e.g., localhost)

    Args:
        job_id: The job ID from the provider
        model: The model endpoint (default: fal-ai/wan-t2v)
        provider: Which provider to check ('kie' or 'fal', default: 'fal')
    """
    try:
        # Use orchestrator to check status on correct provider
        result = await orchestrator.check_status_any_provider(
            job_id=job_id,
            provider=provider,
            model=model
        )
        return result

    except Exception as e:
        print(f"Error checking generation status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check generation status: {str(e)}"
        )


# ===== NEW ENDPOINTS FOR TAB-BASED ARCHITECTURE =====

@router.post("/generate/person")
async def generate_person_only(request: GeneratePersonRequest):
    """
    Generate AI person image only (Stage 1 standalone)

    This endpoint is for the Person Generation tab.
    Users can generate a person, preview it, download it, and later use it in Stage 2.

    Cost: ~$0.125 (Stage 1 only)
    Time: ~30-60 seconds
    """
    try:
        # Validation
        if request.stage1_mode not in ["EASY", "ADVANCED"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stage1_mode: {request.stage1_mode}"
            )

        if request.stage1_mode == "EASY" and not request.person_fields:
            raise HTTPException(status_code=400, detail="person_fields required for EASY mode")

        if request.stage1_mode == "ADVANCED" and not request.person_prompt:
            raise HTTPException(status_code=400, detail="person_prompt required for ADVANCED mode")

        print(f"\n{'='*60}")
        print(f"🎨 PERSON GENERATION (Stage 1 Only)")
        print(f"{'='*60}")
        print(f"Generation ID: {request.generation_id}")
        print(f"User ID: {request.user_id}")
        print(f"Mode: {request.stage1_mode}")
        print(f"{'='*60}\n")

        # Get pipeline orchestrator
        pipeline = get_pipeline_orchestrator()

        # Convert person_fields
        person_fields_dict = None
        if request.person_fields:
            person_fields_dict = {
                "gender": request.person_fields.gender,
                "age": request.person_fields.age,
                "ethnicity": request.person_fields.ethnicity,
                "clothing": request.person_fields.clothing,
                "expression": request.person_fields.expression,
                "background": request.person_fields.background
            }

        # Generate person only (NO product image - Stage 1 is person only)
        result = await pipeline.generate_person_only(
            generation_id=request.generation_id,
            user_id=request.user_id,
            stage1_mode=request.stage1_mode,
            person_prompt=request.person_prompt,
            person_fields=person_fields_dict
        )

        print(f"\n{'='*60}")
        print(f"✅ PERSON GENERATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Person URL: {result['person_url']}")
        print(f"{'='*60}\n")

        return {
            "success": True,
            "generation_id": result["generation_id"],
            "person_url": result["person_url"],
            "person_s3_key": result["person_s3_key"]
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ ERROR: Person generation failed")
        print(f"Generation ID: {request.generation_id}")
        print(f"Error: {str(e)}\n")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate person: {str(e)}"
        )


@router.post("/generate/composite")
async def generate_composite_only(request: GenerateCompositeRequest):
    """
    Generate composite image only (Stage 2 standalone)

    This endpoint is for the Composite tab.
    Users upload/select a person image and product image, then generate the composite.

    Cost: ~$0.125 (Stage 2 only)
    Time: ~30-60 seconds
    """
    try:
        print(f"\n{'='*60}")
        print(f"🖼️  COMPOSITE GENERATION (Stage 2 Only)")
        print(f"{'='*60}")
        print(f"Generation ID: {request.generation_id}")
        print(f"User ID: {request.user_id}")
        print(f"{'='*60}\n")

        pipeline = get_pipeline_orchestrator()

        result = await pipeline.generate_composite_only(
            generation_id=request.generation_id,
            user_id=request.user_id,
            person_image_url=request.person_image_url,
            product_image_url=request.product_image_url,
            composite_prompt=request.composite_prompt
        )

        print(f"\n{'='*60}")
        print(f"✅ COMPOSITE GENERATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Composite URL: {result['composite_url']}")
        print(f"{'='*60}\n")

        return {
            "success": True,
            "generation_id": result["generation_id"],
            "composite_url": result["composite_url"],
            "composite_s3_key": result["composite_s3_key"]
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ ERROR: Composite generation failed")
        print(f"Error: {str(e)}\n")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate composite: {str(e)}"
        )


@router.post("/generate/video-only")
async def generate_video_from_composite(request: GenerateVideoOnlyRequest):
    """
    Generate video from composite image only (Stage 3 standalone)

    This endpoint is for the Video Generation tab.
    Users upload/select a composite image and generate the video.

    Cost: ~$2.25-$2.75 (Stage 3 only)
    Time: ~4-8 minutes depending on mode
    """
    try:
        # Validation
        if request.veo3_mode not in ["FAST", "STANDARD"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid veo3_mode: {request.veo3_mode}"
            )

        print(f"\n{'='*60}")
        print(f"🎥 VIDEO GENERATION (Stage 3 Only)")
        print(f"{'='*60}")
        print(f"Generation ID: {request.generation_id}")
        print(f"User ID: {request.user_id}")
        print(f"Veo3 Mode: {request.veo3_mode}")
        print(f"{'='*60}\n")

        pipeline = get_pipeline_orchestrator()

        result = await pipeline.generate_video_only(
            generation_id=request.generation_id,
            user_id=request.user_id,
            composite_image_url=request.composite_image_url,
            video_prompt=request.video_prompt,
            veo3_mode=request.veo3_mode,
            duration=request.duration,
            aspect_ratio=request.aspect_ratio,
            product_image_url=request.product_image_url  # NEW: For GPT-4o enhancement
        )

        print(f"\n{'='*60}")
        print(f"✅ VIDEO GENERATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Video URL: {result['video_url']}")
        print(f"Provider: {result['provider_used']}")
        print(f"{'='*60}\n")

        return {
            "success": True,
            "generation_id": result["generation_id"],
            "video_url": result["video_url"],
            "video_s3_key": result["video_s3_key"],
            "provider_used": result["provider_used"],
            "fallback_triggered": result["fallback_triggered"]
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ ERROR: Video generation failed")
        print(f"Error: {str(e)}\n")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate video: {str(e)}"
        )


# ===== LEGACY FULL PIPELINE ENDPOINT =====

@router.post("/generate/ugc-video", response_model=GenerateUGCVideoResponse)
async def generate_ugc_video(request: GenerateUGCVideoRequest):
    """
    Generate complete UGC video using 3-stage pipeline

    This endpoint orchestrates the complete workflow:
    1. Stage 1: Generate AI person (Seedream) - Easy or Advanced mode
    2. Stage 2: Composite person with product (Seedream)
    3. Stage 3: Generate video animation (Veo3 via Kie.ai with Fal.ai fallback)

    The process is fully automated and sends webhook updates after each stage.
    Total time: ~4-8 minutes depending on Veo3 mode (FAST vs STANDARD).

    Args:
        request: UGC video generation request with all stage parameters

    Returns:
        GenerateUGCVideoResponse with URLs for all generated assets
    """
    try:
        # Validate stage1_mode
        if request.stage1_mode not in ["EASY", "ADVANCED"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stage1_mode: {request.stage1_mode}. Must be 'EASY' or 'ADVANCED'."
            )

        # Validate EASY mode has person_fields
        if request.stage1_mode == "EASY" and not request.person_fields:
            raise HTTPException(
                status_code=400,
                detail="person_fields required for EASY mode"
            )

        # Validate ADVANCED mode has person_prompt
        if request.stage1_mode == "ADVANCED" and not request.person_prompt:
            raise HTTPException(
                status_code=400,
                detail="person_prompt required for ADVANCED mode"
            )

        # Validate veo3_mode
        if request.veo3_mode not in ["FAST", "STANDARD"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid veo3_mode: {request.veo3_mode}. Must be 'FAST' or 'STANDARD'."
            )

        print(f"\n{'='*60}")
        print(f"🚀 NEW UGC VIDEO GENERATION REQUEST")
        print(f"{'='*60}")
        print(f"Generation ID: {request.generation_id}")
        print(f"User ID: {request.user_id}")
        print(f"Stage 1 Mode: {request.stage1_mode}")
        print(f"Veo3 Mode: {request.veo3_mode}")
        print(f"{'='*60}\n")

        # Get pipeline orchestrator
        pipeline = get_pipeline_orchestrator()

        # Convert person_fields to dict if present
        person_fields_dict = None
        if request.person_fields:
            person_fields_dict = {
                "gender": request.person_fields.gender,
                "age": request.person_fields.age,
                "ethnicity": request.person_fields.ethnicity,
                "clothing": request.person_fields.clothing,
                "expression": request.person_fields.expression,
                "background": request.person_fields.background
            }

        # Execute complete 3-stage pipeline
        result = await pipeline.generate_ugc_video(
            generation_id=request.generation_id,
            user_id=request.user_id,
            # Stage 1
            stage1_mode=request.stage1_mode,
            person_prompt=request.person_prompt,
            person_fields=person_fields_dict,
            # Stage 2
            product_image_url=request.product_image_url,
            composite_prompt=request.composite_prompt,
            # Stage 3
            video_prompt=request.video_prompt,
            veo3_mode=request.veo3_mode,
            # Options
            duration=request.duration,
            aspect_ratio=request.aspect_ratio
        )

        print(f"\n{'='*60}")
        print(f"✅ UGC VIDEO GENERATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Generation ID: {result['generation_id']}")
        print(f"Person URL: {result['person_url']}")
        print(f"Composite URL: {result['composite_url']}")
        print(f"Video URL: {result['video_url']}")
        print(f"Provider: {result['provider_used']}")
        print(f"Fallback Triggered: {result['fallback_triggered']}")
        print(f"Total Time: {result['total_time']:.1f}s ({result['total_time']/60:.1f} min)")
        print(f"{'='*60}\n")

        # Return response
        return GenerateUGCVideoResponse(
            success=True,
            generation_id=result["generation_id"],
            person_url=result["person_url"],
            person_s3_key=result["person_s3_key"],
            composite_url=result["composite_url"],
            composite_s3_key=result["composite_s3_key"],
            video_url=result["video_url"],
            video_s3_key=result["video_s3_key"],
            provider_used=result["provider_used"],
            fallback_triggered=result["fallback_triggered"],
            total_time=result["total_time"]
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Log and return error
        print(f"\n❌ ERROR: UGC video generation failed")
        print(f"Generation ID: {request.generation_id}")
        print(f"Error: {str(e)}")
        print(f"{'='*60}\n")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate UGC video: {str(e)}"
        )


@router.post("/generate/ugc-video-test-stages12")
async def generate_ugc_video_test_stages12(request: GenerateUGCVideoRequest):
    """
    TEST ENDPOINT: Generate only Stages 1-2 (Person + Composite)

    Skips Stage 3 (video generation) to save costs during testing.
    Use this to verify the first two stages work correctly before doing full tests.

    Cost: ~$0.25 (vs ~$3.25 for full pipeline)
    Time: ~1 minute (vs ~8 minutes for full pipeline)

    Returns person image and composite image URLs.
    """
    try:
        # Validation (same as full endpoint)
        if request.stage1_mode not in ["EASY", "ADVANCED"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stage1_mode: {request.stage1_mode}"
            )

        if request.stage1_mode == "EASY" and not request.person_fields:
            raise HTTPException(status_code=400, detail="person_fields required for EASY mode")

        if request.stage1_mode == "ADVANCED" and not request.person_prompt:
            raise HTTPException(status_code=400, detail="person_prompt required for ADVANCED mode")

        print(f"\n{'='*60}")
        print(f"🧪 TEST: STAGES 1-2 ONLY (Skipping video generation)")
        print(f"{'='*60}")
        print(f"Generation ID: {request.generation_id}")
        print(f"Mode: {request.stage1_mode}")
        print(f"{'='*60}\n")

        # Get pipeline orchestrator
        pipeline = get_pipeline_orchestrator()

        # Convert person_fields
        person_fields_dict = None
        if request.person_fields:
            person_fields_dict = {
                "gender": request.person_fields.gender,
                "age": request.person_fields.age,
                "ethnicity": request.person_fields.ethnicity,
                "clothing": request.person_fields.clothing,
                "expression": request.person_fields.expression,
                "background": request.person_fields.background
            }

        # Execute ONLY stages 1-2
        result = await pipeline.generate_person_and_composite(
            generation_id=request.generation_id,
            user_id=request.user_id,
            stage1_mode=request.stage1_mode,
            person_prompt=request.person_prompt,
            person_fields=person_fields_dict,
            product_image_url=request.product_image_url,
            composite_prompt=request.composite_prompt
        )

        print(f"\n{'='*60}")
        print(f"✅ TEST COMPLETE: Stages 1-2 successful!")
        print(f"{'='*60}")
        print(f"Person URL: {result['person_url']}")
        print(f"Composite URL: {result['composite_url']}")
        print(f"Total Time: {result['total_time']:.1f}s")
        print(f"{'='*60}\n")

        # Return partial response (no video)
        return {
            "success": True,
            "generation_id": result["generation_id"],
            "person_url": result["person_url"],
            "person_s3_key": result["person_s3_key"],
            "composite_url": result["composite_url"],
            "composite_s3_key": result["composite_s3_key"],
            "video_url": None,  # Skipped
            "video_s3_key": None,  # Skipped
            "provider_used": "fal",
            "fallback_triggered": False,
            "total_time": result["total_time"],
            "note": "TEST MODE: Stage 3 (video) was skipped"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ TEST FAILED: Stages 1-2")
        print(f"Error: {str(e)}\n")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate person+composite: {str(e)}"
        )
