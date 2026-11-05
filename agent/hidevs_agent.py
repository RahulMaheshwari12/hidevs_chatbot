
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from models.chat import ChatService
from prompts.router_prompt import ROUTER_PROMPT
from retrieval.pipeline import RAGPipeline


logger = logging.getLogger(__name__)

VALID_INTENTS = {"knowledge_query", "small_talk", "unclear", "off_topic"}


class HiDevsSupportAgent:
    """Orchestrate user interactions with the HiDevs support system.

    Every message is first routed by the model. Only messages that need
    HiDevs information go to the knowledge base. Social messages get a
    short direct reply, and unclear messages get a clarifying question.
    """

    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        self.chat_service = ChatService()

    def _route(self, message: str) -> dict:
        """Ask the model how the message should be handled."""

        messages = [
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=message),
        ]

        raw = self.chat_service.generate_response(messages)

        return self._parse_route(raw)

    def _parse_route(self, raw: str) -> dict:
        """Parse the router's JSON. Fall back to knowledge_query if invalid."""

        fallback = {"intent": "knowledge_query", "reply": ""}

        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if not match:
            logger.warning("Router returned no JSON: %r", raw)
            return fallback

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Router returned invalid JSON: %r", raw)
            return fallback

        intent = data.get("intent")

        if intent not in VALID_INTENTS:
            logger.warning("Router returned unknown intent: %r", intent)
            return fallback

        reply = str(data.get("reply") or "").strip()

        return {"intent": intent, "reply": reply}

    def process_message(self, message: str) -> dict:
        """Process a user message and return a support response."""

        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message cannot be empty.")

        message = message.strip()

        try:
            route = self._route(message)
            intent = route["intent"]

            # Knowledge questions go through retrieval.
            # If a social or unclear message has no reply, retrieval is the fallback.
            if intent == "knowledge_query" or not route["reply"]:
                result = self.rag_pipeline.ask(message)

                return {
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "intent": "knowledge_query",
                }

            # Social and unclear messages are answered directly, without retrieval.
            return {
                "answer": route["reply"],
                "sources": [],
                "intent": intent,
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
