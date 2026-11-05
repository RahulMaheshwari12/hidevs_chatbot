from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A single user message sent from the frontend."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's message.",
    )


class SourceItem(BaseModel):
    """A knowledge-base passage used to produce an answer."""

    source_title: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    content: str


class ChatResponse(BaseModel):
    """The agent's reply to a user message."""

    answer: str
    intent: Literal["knowledge_query", "small_talk", "unclear", "off_topic"]
    sources: list[SourceItem] = []


class HealthResponse(BaseModel):
    status: str
