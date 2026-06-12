import argparse
import asyncio
import logging
from collections.abc import Iterable

from app.collectors.base import BaseCollector, CollectionResult
from app.collectors.logsetup import configure_logging
from app.collectors.registry import create_collectors, source_choices
from app.config import settings
from app.db.session import async_session, engine

logger = logging.getLogger(__name__)

# Credentials each source needs to collect fully. sentinel5p degrades to
# catalog-only without its pair; openaq/openweather fail outright (401).
CREDENTIAL_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "openaq": ("openaq_api_key",),
    "openweather": ("openweather_api_key",),
    "sentinel5p": ("cdse_username", "cdse_password"),
}


def missing_credentials(sources: Iterable[str]) -> dict[str, list[str]]:
    """Map source name -> unset settings fields it needs."""
    missing: dict[str, list[str]] = {}
    for source in sources:
        unset = [
            field
            for field in CREDENTIAL_REQUIREMENTS.get(source, ())
            if not getattr(settings, field)
        ]
        if unset:
            missing[source] = unset
    return missing


def warn_missing_credentials(collectors: Iterable[BaseCollector]) -> None:
    for source, fields in missing_credentials(
        collector.source_name for collector in collectors
    ).items():
        message = (
            f"{source}: credentials not set ({', '.join(fields)}) - "
            "collection will fail or run degraded; set them in server/.env"
        )
        print(f"WARNING: {message}")
        logger.warning(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AERIS data collectors.")
    parser.add_argument(
        "--source",
        choices=source_choices(),
        help="Run one collector source instead of all registered collectors.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum attempts per collector before reporting failure.",
    )
    return parser


async def run_collectors(
    collectors: Iterable[BaseCollector],
    *,
    max_retries: int = 3,
) -> list[CollectionResult]:
    results: list[CollectionResult] = []

    for collector in collectors:
        try:
            async with async_session() as session:
                result = await collector.collect(session, max_retries=max_retries)
        except Exception as exc:
            result = CollectionResult(
                source=collector.source_name,
                success=False,
                errors=[f"{type(exc).__name__}: {exc}"],
            )
        finally:
            await collector.close()

        results.append(result)

    return results


def format_result(result: CollectionResult) -> str:
    status = "ok" if result.success else "failed"
    parts = [
        result.source,
        status,
        f"records={result.record_count}",
        f"duration_ms={result.duration_ms}",
    ]
    if result.errors:
        parts.append(f"errors={'; '.join(result.errors)}")
    return " | ".join(parts)


def exit_code(results: Iterable[CollectionResult]) -> int:
    return 0 if all(result.success for result in results) else 1


async def async_main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    collectors = create_collectors(args.source)
    warn_missing_credentials(collectors)

    results = await run_collectors(collectors, max_retries=args.max_retries)

    for result in results:
        print(format_result(result))

    await engine.dispose()
    return exit_code(results)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
