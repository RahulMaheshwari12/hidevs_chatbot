
import logging

from retrieval.pipeline import RAGPipeline


logger = logging.getLogger(__name__)


class HiDevsSupportAgent:
    """Orchestrate user interactions with the HiDevs support system."""

    def __init__(self):
        self.rag_pipeline = RAGPipeline()

    def _handle_greeting(self) -> dict:
        """Return a consistent greeting without invoking retrieval."""

        return {
            "answer": (
                "Hello! Welcome to HiDevs Support. "
                "I'm here to help you with HiDevs features, programs, "
                "projects, and platform-related questions. "
                "How can I help you today?"
            ),
            "sources": [],
            "intent": "greeting",
        }

    def _classify_message(self, message: str) -> str:
        """Identify simple greetings without classifying other questions."""

        normalized_message = message.lower().strip()

        greetings = {
            "hi",
            "hello",
            "hey",
            "hii",
            "hiii",
            "good morning",
            "good afternoon",
            "good evening",
            "good night",
        }

        if normalized_message in greetings:
            return "greeting"

        return "knowledge_query"

    def process_message(self, message: str) -> dict:
        """Process a user message and return a support response."""

        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message cannot be empty.")

        message = message.strip()

        try:
            # 1. Handle greetings directly.
            intent = self._classify_message(message)

            if intent == "greeting":
                return self._handle_greeting()

            # 2. Route other messages through the RAG pipeline.
            result = self.rag_pipeline.ask(message)

            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "intent": "knowledge_query",
            }

        except Exception:
            logger.exception(
                "Support agent failed while processing the message."
            )
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    agent = HiDevsSupportAgent()

    print("\n" + "=" * 60)
    print("          HiDevs Support Agent")
    print("=" * 60)
    print("Ask me anything about HiDevs.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            message = input("\nYou: ").strip()

            if message.lower() in {"exit", "quit"}:
                print("\nHiDevs Support Agent: Goodbye!")
                break

            if not message:
                continue

            response = agent.process_message(message)

            print("\nHiDevs Support Agent:")
            print(response["answer"])

            print(f"\nIntent: {response['intent']}")
            print(f"Sources retrieved: {len(response['sources'])}")

        except KeyboardInterrupt:
            print("\n\nExiting HiDevs Support Agent.")
            break

        except Exception as e:
            print(f"\nError: {e}")