from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseProvider(ABC):
    """Base class for AI provider implementations"""

    @abstractmethod
    async def submit_job(self, request: Any) -> Dict[str, Any]:
        """
        Submit a generation job to the AI provider

        Args:
            request: Generation request data

        Returns:
            Dict with job_id and estimated_time
        """
        pass

    @abstractmethod
    def format_prompt(self, user_prompt: str, system_prompt: str) -> str:
        """
        Format the prompt for this provider

        Args:
            user_prompt: User's prompt
            system_prompt: System/template prompt

        Returns:
            Formatted prompt string
        """
        pass
