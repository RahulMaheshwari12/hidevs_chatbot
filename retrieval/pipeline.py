
import logging

from retrieval.vector_search import VectorSearch
from models.chat import ChatService
from prompts.templates import get_rag_prompt


logger = logging.getLogger(__name__)


class RAGPipeline:
    """Connect document retrieval with grounded LLM answer generation."""

    def __init__(self):
        self.vector_search = VectorSearch()
        self.chat_service = ChatService()
        self.prompt = get_rag_prompt()

    def _build_context(self, documents) -> str:
        """Combine retrieved document chunks into context."""

        context_parts = []

        for index, document in enumerate(documents, start=1):
            context_parts.append(
                f"[Source {index}]\n{document.page_content}"
            )

        return "\n\n".join(context_parts)

    def ask(self, question: str) -> dict:
        """Retrieve relevant context and generate a grounded answer."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        question = question.strip()

        try:
            # 1. Retrieve relevant documents
            documents = self.vector_search.search(question)

            if not documents:
                return {
                    "answer": (
                        "I couldn't find relevant information in the "
                        "available HiDevs knowledge base to answer "
                        "your question."
                    ),
                    "sources": [],
                }

            # 2. Build context
            context = self._build_context(documents)

            # 3. Format the reusable prompt
            messages = self.prompt.format_messages(
                context=context,
                question=question,
            )

            # 4. Generate the answer
            answer = self.chat_service.generate_response(
                messages
            )

            # 5. Prepare source information
            sources = [
                {
                    "content": document.page_content,
                    "metadata": document.metadata,
                }
                for document in documents
            ]

            return {
                "answer": answer,
                "sources": sources,
            }

        except Exception:
            logger.exception(
                "RAG pipeline failed while answering: %s",
                question,
            )
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pipeline = RAGPipeline()

    test_questions = [
        "What is CodeQuest and what types of coding problems does it provide?",
        "What is LeetPrompt?",
        "What features are available in Dr. Agent?",
        "What is the purpose of the AI Agents Hub?",
        "What is the official customer support phone number for HiDevs?",
    ]

    for question in test_questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        result = pipeline.ask(question)

        print("\nANSWER:")
        print(result["answer"])

        print("\nRETRIEVED SOURCE COUNT:")
        print(len(result["sources"]))

        for index, source in enumerate(result["sources"], start=1):
            print(f"\nSource {index}")
            print(f"Metadata: {source['metadata']}")
