import logging

from fastapi import APIRouter, HTTPException, Request

from app.schemas.chat import ChatRequest, ChatResponse, HealthResponse, SourceItem


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Confirm the API is running."""
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Send one user message to the HiDevs agent and return its reply.

    Declared as a regular (sync) function so FastAPI runs the agent call
    in a worker thread instead of blocking the event loop.
    """
    agent = request.app.state.agent

    message = payload.message.strip()

    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    try:
        result = agent.process_message(message)

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Chat request failed.")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while answering. Please try again.",
        ) from exc

    sources = [
        SourceItem(
            source_title=source["metadata"].get("source_title"),
            section_number=source["metadata"].get("section_number"),
            section_title=source["metadata"].get("section_title"),
            content=source["content"],
        )
        for source in result["sources"]
    ]

    return ChatResponse(
        answer=result["answer"],
        intent=result["intent"],
        sources=sources,
    )
