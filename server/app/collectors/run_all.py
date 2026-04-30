import argparse
import asyncio
from collections.abc import Iterable

from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.base import BaseCollector, CollectionResult
from app.collectors.registry import create_collectors, source_choices
from app.db.session import async_session, engine


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
    session: AsyncSession,
    collectors: Iterable[BaseCollector],
    *,
    max_retries: int = 3,
) -> list[CollectionResult]:
    results: list[CollectionResult] = []

    for collector in collectors:
        try:
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
    args = build_parser().parse_args(argv)
    collectors = create_collectors(args.source)

    async with async_session() as session:
        results = await run_collectors(
            session,
            collectors,
            max_retries=args.max_retries,
        )

    for result in results:
        print(format_result(result))

    await engine.dispose()
    return exit_code(results)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
