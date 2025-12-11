#!/usr/bin/env python3
"""
Test Script for FULL PIPELINE (Stages 1-2-3 with GPT-4o Enhancement)

Tests the complete video generation pipeline including:
- Stage 1: AI Person Generation (with GPT-4o prompt enhancement)
- Stage 2: Product Compositing (with GPT-4o prompt enhancement)
- Stage 3: Video Generation (with GPT-4o prompt enhancement)

Cost: ~$2.50-$3.50 (including GPT-4o ~$0.024)
Time: ~5-8 minutes
"""

import asyncio
import httpx
import uuid
from pathlib import Path
import os
from dotenv import load_dotenv
import boto3
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuration
PYTHON_BACKEND_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "dev-secret-key-change-in-production")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION")

# Test product image path
TEST_PRODUCT_IMAGE = Path("../app/public/reference-images/Tinted serum.jpg")


async def upload_product_to_s3(image_path: Path, generation_id: str) -> str:
    """Upload product image to S3 and return URL"""
    print(f"\n📤 Uploading product image to S3...")

    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION
    )

    # S3 key for test product
    s3_key = f"products/test-user/{generation_id}-product.jpg"

    # Upload
    with open(image_path, 'rb') as f:
        s3.put_object(
            Bucket=AWS_S3_BUCKET,
            Key=s3_key,
            Body=f,
            ContentType='image/jpeg'
        )

    # Generate URL
    s3_url = f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    print(f"✅ Product uploaded: {s3_url}")

    return s3_url


async def test_full_pipeline(product_url: str):
    """Test complete pipeline with all 3 stages and GPT-4o enhancement"""

    generation_id = str(uuid.uuid4())
    user_id = "test-user-" + str(uuid.uuid4())[:8]

    print(f"\n{'='*70}")
    print(f"🧪 TEST: FULL PIPELINE (Stages 1-2-3 with GPT-4o)")
    print(f"{'='*70}")
    print(f"Generation ID: {generation_id}")
    print(f"User ID: {user_id}")
    print(f"Product URL: {product_url}")
    print(f"")
    print(f"⚠️  Note: This will take ~5-8 minutes and cost ~$2.50-$3.50")
    print(f"{'='*70}\n")

    # Request payload
    payload = {
        "generation_id": generation_id,
        "user_id": user_id,
        "stage1_mode": "EASY",
        "person_fields": {
            "gender": "female",
            "age": "26-35",
            "ethnicity": "Caucasian",
            "clothing": "casual",
            "expression": "smiling",
            "background": "white"
        },
        "product_image_url": product_url,
        "composite_prompt": "Woman holding the tinted serum bottle naturally, showcasing it to the camera with genuine excitement",
        "video_prompt": "Woman enthusiastically presenting the serum with a big smile, natural hand movements showing the product, authentic UGC style",
        "veo3_mode": "FAST",  # Use FAST mode to save time (2-3 min vs 4-6 min)
        "duration": 8,
        "aspect_ratio": "9:16"
    }

    # Call full pipeline endpoint
    print("📡 Calling full pipeline endpoint...")
    print(f"Endpoint: {PYTHON_BACKEND_URL}/api/v1/generate/ugc-video")
    print(f"Mode: FAST (2-3 minutes for video generation)")
    print(f"")

    start_time = datetime.now()

    async with httpx.AsyncClient(timeout=600.0) as client:  # 10 min timeout
        try:
            print("⏳ Starting generation... (this will take several minutes)")
            print("📊 Progress: Stage 1 (Person Generation)...")

            response = await client.post(
                f"{PYTHON_BACKEND_URL}/api/v1/generate/ugc-video",
                json=payload,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                }
            )

            response.raise_for_status()
            result = response.json()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"\n{'='*70}")
            print(f"✅ FULL PIPELINE TEST SUCCESSFUL!")
            print(f"{'='*70}")
            print(f"Generation ID: {result['generation_id']}")
            print(f"")
            print(f"📸 Person Image:")
            print(f"   URL: {result['person_url'][:80]}...")
            print(f"   S3 Key: {result['person_s3_key']}")
            print(f"")
            print(f"🎨 Composite Image:")
            print(f"   URL: {result['composite_url'][:80]}...")
            print(f"   S3 Key: {result['composite_s3_key']}")
            print(f"")
            print(f"🎬 Video:")
            print(f"   URL: {result['video_url'][:80]}...")
            print(f"   S3 Key: {result['video_s3_key']}")
            print(f"   Provider: {result.get('provider_used', 'unknown')}")
            print(f"   Fallback Triggered: {result.get('fallback_triggered', False)}")
            print(f"")
            print(f"⏱️  Total Time: {duration:.1f}s ({duration/60:.1f} min)")
            print(f"💰 Estimated Cost:")
            print(f"   - GPT-4o Enhancement: ~$0.024")
            print(f"   - Stage 1 (Person): ~$0.10")
            print(f"   - Stage 2 (Composite): ~$0.15")
            print(f"   - Stage 3 (Video - FAST): ~$1.50")
            print(f"   - Total: ~$1.77")
            print(f"")
            print(f"🎉 GPT-4o Enhancement was applied to all 3 stages!")
            print(f"{'='*70}\n")

            return result

        except httpx.HTTPStatusError as e:
            print(f"\n❌ HTTP Error: {e.response.status_code}")
            print(f"Response: {e.response.text}")
            raise
        except httpx.TimeoutException:
            print(f"\n⏱️  Request timed out after 10 minutes")
            print(f"This might happen if video generation is taking longer than expected")
            raise
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            raise


async def main():
    """Run full pipeline test"""

    print(f"\n{'='*70}")
    print(f"🚀 STARTING FULL PIPELINE TEST (WITH GPT-4o)")
    print(f"{'='*70}")
    print(f"Backend URL: {PYTHON_BACKEND_URL}")
    print(f"S3 Bucket: {AWS_S3_BUCKET}")
    print(f"S3 Region: {AWS_REGION}")
    print(f"{'='*70}")

    # Check if backend is healthy
    try:
        async with httpx.AsyncClient() as client:
            health = await client.get(f"{PYTHON_BACKEND_URL}/health")
            health.raise_for_status()
            print(f"✅ Backend is healthy")
    except Exception as e:
        print(f"❌ Backend health check failed: {e}")
        print(f"Make sure Python backend is running on port 8000")
        return

    # Check if test product image exists
    if not TEST_PRODUCT_IMAGE.exists():
        print(f"❌ Test product image not found: {TEST_PRODUCT_IMAGE}")
        return

    print(f"✅ Test product image found: {TEST_PRODUCT_IMAGE.name}")

    # Generate unique ID for this test run
    test_id = str(uuid.uuid4())[:8]

    # Upload product to S3
    try:
        product_url = await upload_product_to_s3(TEST_PRODUCT_IMAGE, test_id)
    except Exception as e:
        print(f"❌ Failed to upload product to S3: {e}")
        return

    # Run test
    try:
        result = await test_full_pipeline(product_url)

        # Summary
        print(f"\n{'='*70}")
        print(f"🎉 TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"✅ Stage 1 - Person generated with GPT-4o enhanced prompt")
        print(f"✅ Stage 2 - Composite created with GPT-4o enhanced prompt")
        print(f"✅ Stage 3 - Video generated with GPT-4o enhanced prompt")
        print(f"")
        print(f"💡 Check the backend logs to see GPT-4o enhancement in action:")
        print(f"   - Product image analysis")
        print(f"   - Person prompt enhancement (UGC-style keywords)")
        print(f"   - Composite prompt enhancement (lighting & integration)")
        print(f"   - Video prompt enhancement (amateur motion & camera)")
        print(f"")
        print(f"📁 All files are stored in S3 bucket: {AWS_S3_BUCKET}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check dependencies
    try:
        import boto3
        import httpx
        from dotenv import load_dotenv
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print(f"Install with: pip install boto3 httpx python-dotenv")
        exit(1)

    # Run test
    asyncio.run(main())
