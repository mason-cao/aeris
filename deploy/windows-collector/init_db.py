"""One-time database setup for the AERIS Windows collector box.

Creates every table in the database named by DATABASE_URL (server/.env).
Safe to re-run. Run from anywhere in the repo:

    python deploy\\windows-collector\\init_db.py
"""
import asyncio
import os
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[2] / "server"
os.chdir(SERVER_DIR)  # pydantic-settings reads .env relative to the working dir
sys.path.insert(0, str(SERVER_DIR))

from app.db.schema import create_tables  # noqa: E402
from app.db.session import engine  # noqa: E402


async def main() -> None:
    await create_tables(engine)
    await engine.dispose()
    print(f"Tables ready in {engine.url}")


if __name__ == "__main__":
    asyncio.run(main())
