"""
Gemini LLM integration.
Generates answers using Google Generative AI.
"""
import os
from typing import Optional
import google.generativeai as genai


class GeminiTimeoutError(Exception):
    """Raised when Gemini does not respond within the configured time."""


class GeminiRateLimitError(Exception):
    """Raised when the Gemini project has exhausted its request quota."""


class GeminiLLM:
    """Interface to Gemini LLM."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.5-flash"):
        """
        Initialize Gemini LLM.
        
        Args:
            api_key: Google API key (from .env if not provided)
            model_name: Gemini model to use
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Please set it in .env file or pass it as argument."
            )
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        print(f"✓ Initialized Gemini LLM: {model_name}")
    
    def generate_answer(self, prompt: str) -> str:
        """
        Generate answer using Gemini.
        
        Args:
            prompt: Full prompt including context and question
            
        Returns:
            Generated answer text
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 512},
            )
            return response.text
        except Exception as e:
            message = str(e)
            lowered_message = message.lower()
            if "429" in message or "quota" in lowered_message or "rate limit" in lowered_message:
                raise GeminiRateLimitError(
                    "Gemini request quota has been reached. Please wait and try again."
                ) from e
            if (
                "timeout" in lowered_message
                or "timed out" in lowered_message
                or "deadline exceeded" in lowered_message
                or "504" in message
            ):
                raise GeminiTimeoutError(
                    "Gemini took too long to generate an answer. Please try again."
                ) from e
            raise Exception(f"Error generating answer: {message}") from e
