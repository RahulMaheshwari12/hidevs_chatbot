import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


class GeminiChatService:
    """Initialize and provide access to Google's Gemini chat model."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")

        self.model_name = os.getenv(
            "GEMINI_CHAT_MODEL",
            "gemini-3.5-flash-lite",
        )

        self.temperature = float(
            os.getenv("GEMINI_TEMPERATURE", "0.0")
        )

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Check your .env file."
            )

        if not self.model_name.strip():
            raise ValueError(
                "GEMINI_CHAT_MODEL cannot be empty."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "GEMINI_TEMPERATURE must be between 0.0 and 2.0."
            )

        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the Gemini chat model."""

        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
        )

    def get_model(self):
        """Return the initialized LangChain chat model."""

        return self.model