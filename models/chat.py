
from langchain_core.messages import BaseMessage, HumanMessage
from models.llm_provider import GeminiChatService


class ChatService:
    """Handle Gemini responses with strings or LangChain messages."""

    def __init__(self):
        self.llm = GeminiChatService().get_model()

    def generate_response(
        self,
        message: str | list[BaseMessage],
        chat_history: list[BaseMessage] | None = None,
    ) -> str:
        """Generate a response from a string or formatted message list."""

        if isinstance(message, str):
            if not message.strip():
                raise ValueError("Message cannot be empty.")

            if chat_history is None:
                chat_history = []

            if not isinstance(chat_history, list):
                raise TypeError("Chat history must be a list.")

            if not all(
                isinstance(item, BaseMessage) for item in chat_history
            ):
                raise TypeError(
                    "Every chat history item must be a LangChain message."
                )

            messages = [
                *chat_history,
                HumanMessage(content=message.strip()),
            ]

        elif isinstance(message, list):
            if not message:
                raise ValueError("Message list cannot be empty.")

            if not all(
                isinstance(item, BaseMessage) for item in message
            ):
                raise TypeError(
                    "Every item must be a LangChain message."
                )

            if chat_history:
                raise ValueError(
                    "Pass chat history separately only when message is a string."
                )

            messages = message

        else:
            raise TypeError(
                "Message must be a string or a list of LangChain messages."
            )

        try:
            response = self.llm.invoke(messages)
            content = response.content

            if isinstance(content, list):
                text_parts = [
                    block["text"]
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                    and block.get("text")
                ]
                content = "\n".join(text_parts)

            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("The model returned an empty response.")

            return content.strip()

        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate chat response: {exc}"
            ) from exc
