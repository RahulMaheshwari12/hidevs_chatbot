
import os

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from dotenv import load_dotenv


load_dotenv()


class KnowledgeBaseLoader:
    """Load the local HiDevs knowledge base from a text file."""

    def __init__(self):
        self.file_path = os.getenv(
            "KNOWLEDGE_BASE_PATH",
            "data/knowledge_base.txt"
        )

    def load(self) -> list[Document]:
        """Load knowledge-base content into LangChain documents."""

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Knowledge base file not found: {self.file_path}"
            )

        if not os.path.isfile(self.file_path):
            raise ValueError(
                f"Knowledge base path is not a file: {self.file_path}"
            )

        loader = TextLoader(
            self.file_path,
            encoding="utf-8"
        )

        documents = loader.load()

        if not documents:
            raise ValueError("Knowledge base file is empty.")

        if not any(document.page_content.strip() for document in documents):
            raise ValueError("Knowledge base contains no readable text.")

        return documents
