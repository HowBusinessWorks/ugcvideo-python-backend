"""
Error Classification Utility for UGC Video Generation

Categorizes errors into refundable vs non-refundable types to ensure
fair credit handling for users.
"""

from typing import Dict, Any, Optional
from enum import Enum
import httpx


class ErrorType(str, Enum):
    """Error type categories matching Node.js backend"""
    USER_ERROR = "USER_ERROR"           # User's fault - no refund
    VALIDATION_ERROR = "VALIDATION_ERROR"  # Invalid input - no refund
    SERVICE_ERROR = "SERVICE_ERROR"     # AI service error - refund
    SYSTEM_ERROR = "SYSTEM_ERROR"       # Our backend error - refund
    TIMEOUT = "TIMEOUT"                 # Processing timeout - refund


# User-friendly error messages
ERROR_MESSAGES = {
    ErrorType.USER_ERROR: "Invalid input provided. Please check your data and try again.",
    ErrorType.VALIDATION_ERROR: "Please fix the validation errors and try again.",
    ErrorType.SERVICE_ERROR: "The AI service is temporarily unavailable. Your credits will be refunded.",
    ErrorType.SYSTEM_ERROR: "A system error occurred. Your credits will be refunded automatically.",
    ErrorType.TIMEOUT: "Generation took too long and timed out. Your credits will be refunded.",
}


def classify_error(exception: Exception, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Classify an exception and return error details for webhook.

    Args:
        exception: The exception that occurred
        context: Optional context (e.g., "stage1", "stage2", "stage3")

    Returns:
        dict: {
            "error_type": str,
            "error_message": str,
            "is_refundable": bool,
            "can_retry": bool,
            "technical_details": str
        }
    """
    error_str = str(exception).lower()
    exception_type = type(exception).__name__

    # 1. VALIDATION_ERROR - Invalid input from user (NO REFUND)
    if _is_validation_error(exception, error_str):
        return {
            "error_type": ErrorType.VALIDATION_ERROR,
            "error_message": _extract_validation_message(exception),
            "is_refundable": False,
            "can_retry": True,
            "technical_details": str(exception)
        }

    # 2. TIMEOUT - Processing timeout (REFUND)
    if _is_timeout_error(exception, error_str):
        return {
            "error_type": ErrorType.TIMEOUT,
            "error_message": ERROR_MESSAGES[ErrorType.TIMEOUT],
            "is_refundable": True,
            "can_retry": True,
            "technical_details": str(exception)
        }

    # 3. SERVICE_ERROR - External AI service error (REFUND)
    if _is_service_error(exception, error_str):
        return {
            "error_type": ErrorType.SERVICE_ERROR,
            "error_message": ERROR_MESSAGES[ErrorType.SERVICE_ERROR],
            "is_refundable": True,
            "can_retry": True,
            "technical_details": str(exception)
        }

    # 4. USER_ERROR - Content policy violations (NO REFUND)
    if _is_user_error(exception, error_str):
        return {
            "error_type": ErrorType.USER_ERROR,
            "error_message": "Content violates usage policies. Please try with appropriate content.",
            "is_refundable": False,
            "can_retry": False,
            "technical_details": str(exception)
        }

    # 5. SYSTEM_ERROR - Default for unexpected errors (REFUND)
    return {
        "error_type": ErrorType.SYSTEM_ERROR,
        "error_message": ERROR_MESSAGES[ErrorType.SYSTEM_ERROR],
        "is_refundable": True,
        "can_retry": True,
        "technical_details": f"{exception_type}: {str(exception)}"
    }


def _is_validation_error(exception: Exception, error_str: str) -> bool:
    """Check if error is a validation error (user input problem)"""
    validation_indicators = [
        "validation",
        "invalid",
        "missing required",
        "must be",
        "expected",
        "malformed",
        "unsupported format",
        "invalid format",
        "invalid image",
        "image too large",
        "image too small",
        "invalid dimensions",
        "corrupt",
        "not a valid",
    ]

    # Check exception type
    if isinstance(exception, ValueError):
        return True

    # Check error message
    return any(indicator in error_str for indicator in validation_indicators)


def _is_timeout_error(exception: Exception, error_str: str) -> bool:
    """Check if error is a timeout"""
    timeout_indicators = [
        "timeout",
        "timed out",
        "deadline exceeded",
        "request timeout",
        "connection timeout",
        "read timeout",
    ]

    # Check exception type
    if isinstance(exception, (asyncio.TimeoutError, TimeoutError)):
        return True

    # Check for httpx timeout
    if isinstance(exception, httpx.TimeoutException):
        return True

    # Check error message
    return any(indicator in error_str for indicator in timeout_indicators)


def _is_service_error(exception: Exception, error_str: str) -> bool:
    """Check if error is from external AI service"""
    service_indicators = [
        "api error",
        "service unavailable",
        "503",
        "502",
        "504",
        "rate limit",
        "quota exceeded",
        "seedream",
        "fal.ai",
        "kie.ai",
        "veo3",
        "provider error",
        "external service",
        "downstream",
        "failed to generate",
        "generation failed",
        "500 internal server error",  # From AI provider, not our server
    ]

    # Check for HTTP errors from external services
    if isinstance(exception, httpx.HTTPStatusError):
        # 5xx errors from AI services are service errors
        if exception.response.status_code >= 500:
            return True
        # 429 rate limiting
        if exception.response.status_code == 429:
            return True

    # Check error message
    return any(indicator in error_str for indicator in service_indicators)


def _is_user_error(exception: Exception, error_str: str) -> bool:
    """Check if error is due to user content violations"""
    user_error_indicators = [
        "nsfw",
        "inappropriate",
        "prohibited content",
        "policy violation",
        "content moderation",
        "restricted content",
        "harmful content",
        "unsafe content",
    ]

    return any(indicator in error_str for indicator in user_error_indicators)


def _extract_validation_message(exception: Exception) -> str:
    """Extract user-friendly validation error message"""
    error_str = str(exception)

    # Try to extract specific validation message
    if "invalid" in error_str.lower():
        return f"Invalid input: {error_str}"
    elif "required" in error_str.lower():
        return f"Missing required field: {error_str}"
    elif "format" in error_str.lower():
        return f"Invalid format: {error_str}"
    else:
        return ERROR_MESSAGES[ErrorType.VALIDATION_ERROR]


# For easier imports
import asyncio

__all__ = ['ErrorType', 'classify_error', 'ERROR_MESSAGES']
