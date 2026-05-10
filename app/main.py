import logging

# pyrefly: ignore [missing-import]
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.config.settings import settings
from src.routers.api import api_router
from app.dependencies import get_prediction_service

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models into memory
    logger.info("Starting up FastAPI application...")
    service = get_prediction_service()
    service.load_artifacts()
    yield
    # Shutdown: Clean up resources
    logger.info("Shutting down FastAPI application...")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="A scalable API for YouTube Viewer Sentiment Analysis.",
        version="1.0.0",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # Enable CORS for the Chrome Extension
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
