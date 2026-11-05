"""
HiDevs Support API.

Run locally from the project root:
    uvicorn app.main:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.hidevs_agent import HiDevsSupportAgent
from app.api.routes import router


load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://localhost:5173"


def _cors_origins() -> list[str]:
    """Read allowed frontend origins from CORS_ORIGINS (comma-separated)."""
    raw = os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the agent once when the server starts."""
    logger.info("Starting HiDevs Support API...")
    app.state.agent = HiDevsSupportAgent()
    logger.info("HiDevs Support agent ready.")
    yield


app = FastAPI(
    title="HiDevs Support API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(router)
