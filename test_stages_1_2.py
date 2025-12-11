#!/usr/bin/env python3
"""
Test Script for Stages 1-2 (Person Generation + Compositing)

This script tests the pipeline without running the expensive Stage 3 (video generation).
Cost: ~$0.25 (vs ~$3.25 for full pipeline)
Time: ~1-2 minutes
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


async def test_stages_1_2_easy_mode(product_url: str):
    """Test with Easy Mode (dropdown fields)"""

    generation_id = str(uuid.uuid4())
    user_id = "test-user-" + str(uuid.uuid4())[:8]

    print(f"\n{'='*70}")
    print(f"🧪 TEST: STAGES 1-2 ONLY (Easy Mode)")
    print(f"{'='*70}")
    print(f"Generation ID: {generation_id}")
    print(f"User ID: {user_id}")
    print(f"Product URL: {product_url}")
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
        "composite_prompt": "Woman holding the tinted serum bottle naturally, showcasing it to the camera",
        "video_prompt": "Woman enthusiastically presenting the serum with a big smile",
        "veo3_mode": "FAST",  # Not used in test, but required field
        "duration": 8,
        "aspect_ratio": "9:16"
    }

    # Call test endpoint
    print("📡 Calling test endpoint...")
    print(f"Endpoint: {PYTHON_BACKEND_URL}/api/v1/generate/ugc-video-test-stages12")

    start_time = datetime.now()

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{PYTHON_BACKEND_URL}/api/v1/generate/ugc-video-test-stages12",
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
            print(f"✅ TEST SUCCESSFUL!")
            print(f"{'='*70}")
            print(f"Generation ID: {result['generation_id']}")
            print(f"")
            print(f"📸 Person Image:")
            print(f"   URL: {result['person_url']}")
            print(f"   S3 Key: {result['person_s3_key']}")
            print(f"")
            print(f"🎨 Composite Image:")
            print(f"   URL: {result['composite_url']}")
            print(f"   S3 Key: {result['composite_s3_key']}")
            print(f"")
            print(f"⏱️  Total Time: {duration:.1f}s ({duration/60:.1f} min)")
            print(f"💰 Estimated Cost: ~$0.25")
            print(f"")
            print(f"Note: {result.get('note', '')}")
            print(f"{'='*70}\n")

            return result

        except httpx.HTTPStatusError as e:
            print(f"\n❌ HTTP Error: {e.response.status_code}")
            print(f"Response: {e.response.text}")
            raise
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            raise


async def test_stages_1_2_advanced_mode(product_url: str):
    """Test with Advanced Mode (custom prompt)"""

    generation_id = str(uuid.uuid4())
    user_id = "test-user-" + str(uuid.uuid4())[:8]

    print(f"\n{'='*70}")
    print(f"🧪 TEST: STAGES 1-2 ONLY (Advanced Mode)")
    print(f"{'='*70}")
    print(f"Generation ID: {generation_id}")
    print(f"User ID: {user_id}")
    print(f"Product URL: {product_url}")
    print(f"{'='*70}\n")

    # Request payload
    payload = {
        "generation_id": generation_id,
        "user_id": user_id,
        "stage1_mode": "ADVANCED",
        "person_prompt": "Professional Asian woman in her early 30s, wearing a white blazer, natural makeup, confident smile, studio lighting with soft shadows, white background, photorealistic, high quality",
        "product_image_url": product_url,
        "composite_prompt": "Woman elegantly holding the tinted serum bottle in her right hand at chest level, presenting it towards camera",
        "video_prompt": "Woman presenting the serum confidently",
        "veo3_mode": "FAST",
        "duration": 8,
        "aspect_ratio": "9:16"
    }

    # Call test endpoint
    print("📡 Calling test endpoint...")
    print(f"Endpoint: {PYTHON_BACKEND_URL}/api/v1/generate/ugc-video-test-stages12")

    start_time = datetime.now()

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{PYTHON_BACKEND_URL}/api/v1/generate/ugc-video-test-stages12",
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
            print(f"✅ TEST SUCCESSFUL!")
            print(f"{'='*70}")
            print(f"Generation ID: {result['generation_id']}")
            print(f"")
            print(f"📸 Person Image:")
            print(f"   URL: {result['person_url']}")
            print(f"   S3 Key: {result['person_s3_key']}")
            print(f"")
            print(f"🎨 Composite Image:")
            print(f"   URL: {result['composite_url']}")
            print(f"   S3 Key: {result['composite_s3_key']}")
            print(f"")
            print(f"⏱️  Total Time: {duration:.1f}s ({duration/60:.1f} min)")
            print(f"💰 Estimated Cost: ~$0.25")
            print(f"")
            print(f"Note: {result.get('note', '')}")
            print(f"{'='*70}\n")

            return result

        except httpx.HTTPStatusError as e:
            print(f"\n❌ HTTP Error: {e.response.status_code}")
            print(f"Response: {e.response.text}")
            raise
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            raise


async def main():
    """Run all tests"""

    print(f"\n{'='*70}")
    print(f"🚀 STARTING STAGES 1-2 TEST")
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

    # Run tests
    print(f"\n{'='*70}")
    print(f"Running 2 tests: Easy Mode + Advanced Mode")
    print(f"{'='*70}\n")

    try:
        # Test 1: Easy Mode
        print("\n🧪 TEST 1/2: Easy Mode (dropdown fields)")
        result1 = await test_stages_1_2_easy_mode(product_url)

        # Wait a bit between tests
        print("\n⏳ Waiting 5 seconds before next test...\n")
        await asyncio.sleep(5)

        # Test 2: Advanced Mode
        print("\n🧪 TEST 2/2: Advanced Mode (custom prompt)")
        result2 = await test_stages_1_2_advanced_mode(product_url)

        # Summary
        print(f"\n{'='*70}")
        print(f"🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"✅ Easy Mode Test - Person & Composite generated")
        print(f"✅ Advanced Mode Test - Person & Composite generated")
        print(f"")
        print(f"💰 Total Estimated Cost: ~$0.50 (2 tests)")
        print(f"⏱️  Stage 3 (video) was skipped to save costs")
        print(f"")
        print(f"Next Steps:")
        print(f"1. Check the S3 bucket for generated images")
        print(f"2. Review the quality of person generation and compositing")
        print(f"3. If satisfied, run full pipeline test with Stage 3")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
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

    # Run tests
    asyncio.run(main())
