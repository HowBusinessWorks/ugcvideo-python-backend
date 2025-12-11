"""
Simple validation script to test imports and endpoint structure
This validates the code without making actual API calls
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")

    try:
        from app.main import app
        print("✅ Main app imports successful")
    except Exception as e:
        print(f"❌ Main app import failed: {e}")
        return False

    try:
        from app.models.request import GenerateUGCVideoRequest, PersonFieldsRequest
        print("✅ Request models import successful")
    except Exception as e:
        print(f"❌ Request models import failed: {e}")
        return False

    try:
        from app.models.response import GenerateUGCVideoResponse
        print("✅ Response models import successful")
    except Exception as e:
        print(f"❌ Response models import failed: {e}")
        return False

    try:
        from app.services.providers import SeedreamProvider, Veo3Provider
        print("✅ Providers import successful")
    except Exception as e:
        print(f"❌ Providers import failed: {e}")
        return False

    try:
        from app.services.pipeline_orchestrator import (
            init_pipeline_orchestrator,
            get_pipeline_orchestrator
        )
        print("✅ Pipeline orchestrator import successful")
    except Exception as e:
        print(f"❌ Pipeline orchestrator import failed: {e}")
        return False

    return True


def test_request_model():
    """Test request model validation"""
    print("\nTesting request model validation...")

    from app.models.request import GenerateUGCVideoRequest, PersonFieldsRequest

    try:
        # Test EASY mode request
        person_fields = PersonFieldsRequest(
            gender="female",
            age="26-35",
            ethnicity="Asian",
            clothing="casual",
            expression="smiling",
            background="white"
        )

        request = GenerateUGCVideoRequest(
            generation_id="test-gen-123",
            user_id="test-user-456",
            stage1_mode="EASY",
            person_fields=person_fields,
            product_image_url="https://s3.amazonaws.com/test/product.jpg",
            video_prompt="Woman presents product enthusiastically",
            veo3_mode="STANDARD",
            duration=8,
            aspect_ratio="9:16"
        )

        print(f"✅ EASY mode request model valid")
        print(f"   - Generation ID: {request.generation_id}")
        print(f"   - Stage 1 Mode: {request.stage1_mode}")
        print(f"   - Person Gender: {request.person_fields.gender}")

    except Exception as e:
        print(f"❌ Request model validation failed: {e}")
        return False

    try:
        # Test ADVANCED mode request
        request = GenerateUGCVideoRequest(
            generation_id="test-gen-789",
            user_id="test-user-456",
            stage1_mode="ADVANCED",
            person_prompt="A young Asian woman in casual clothes, smiling naturally against white background",
            product_image_url="https://s3.amazonaws.com/test/product.jpg",
            video_prompt="Woman presents product enthusiastically",
            veo3_mode="FAST"
        )

        print(f"✅ ADVANCED mode request model valid")
        print(f"   - Generation ID: {request.generation_id}")
        print(f"   - Stage 1 Mode: {request.stage1_mode}")
        print(f"   - Veo3 Mode: {request.veo3_mode}")

    except Exception as e:
        print(f"❌ ADVANCED mode validation failed: {e}")
        return False

    return True


def test_response_model():
    """Test response model structure"""
    print("\nTesting response model structure...")

    from app.models.response import GenerateUGCVideoResponse

    try:
        response = GenerateUGCVideoResponse(
            success=True,
            generation_id="test-gen-123",
            person_url="https://s3.amazonaws.com/bucket/person.png",
            person_s3_key="person-images/user-123/gen-123.png",
            composite_url="https://s3.amazonaws.com/bucket/composite.png",
            composite_s3_key="composites/user-123/gen-123.png",
            video_url="https://s3.amazonaws.com/bucket/video.mp4",
            video_s3_key="videos/user-123/gen-123.mp4",
            provider_used="kie",
            fallback_triggered=False,
            total_time=245.3
        )

        print(f"✅ Response model valid")
        print(f"   - Person URL: {response.person_url}")
        print(f"   - Composite URL: {response.composite_url}")
        print(f"   - Video URL: {response.video_url}")
        print(f"   - Provider: {response.provider_used}")
        print(f"   - Total Time: {response.total_time}s")

    except Exception as e:
        print(f"❌ Response model validation failed: {e}")
        return False

    return True


def test_endpoint_registration():
    """Test that endpoint is registered in FastAPI app"""
    print("\nTesting endpoint registration...")

    from app.main import app

    # Get all routes
    routes = [route.path for route in app.routes]

    if "/api/v1/generate/ugc-video" in routes:
        print(f"✅ UGC video endpoint registered: /api/v1/generate/ugc-video")
    else:
        print(f"❌ UGC video endpoint not found in routes")
        print(f"   Available routes: {routes}")
        return False

    # Check other important endpoints
    if "/health" in routes:
        print(f"✅ Health endpoint registered: /health")

    if "/" in routes:
        print(f"✅ Root endpoint registered: /")

    return True


def main():
    """Run all validation tests"""
    print("\n" + "="*60)
    print("🧪 UGC VIDEO ENDPOINT VALIDATION")
    print("="*60 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Request Model", test_request_model),
        ("Response Model", test_response_model),
        ("Endpoint Registration", test_endpoint_registration),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\n🎯 Result: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All validations passed! Endpoint is ready for testing.")
        print("\n📝 Next steps:")
        print("   1. Start the FastAPI server: uvicorn app.main:app --reload")
        print("   2. View API docs: http://localhost:8000/docs")
        print("   3. Test with Postman or curl")
        return 0
    else:
        print("\n❌ Some validations failed. Fix errors before testing.")
        return 1


if __name__ == "__main__":
    exit(main())
