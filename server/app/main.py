import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.schema import create_tables
from app.db.session import engine

logging.basicConfig(
    level=getattr(logging, settings.aeris_log_level),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("AERIS starting up (env=%s)", settings.aeris_env)
    if settings.aeris_env == "development":
        try:
            await create_tables(engine)
        except Exception:
            logger.warning("Could not auto-create tables (DB may not support TimescaleDB)")
    yield
    await engine.dispose()
    logger.info("AERIS shut down")


app = FastAPI(
    title="AERIS",
    description="Autonomous Environmental RAG & Inference System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
from app.api.routes.system import router as system_router
from app.api.routes.data import router as data_router

app.include_router(system_router, prefix="/api")
app.include_router(data_router, prefix="/api")
