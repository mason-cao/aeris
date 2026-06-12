"""Process-level logging for the collector entry points.

The scheduled runs redirect stdout/stderr into collector.log; without an
explicit config only WARNING+ and the SQLAlchemy echo ever land there, which
is how Sentinel-5P ran catalog-only for three weeks without leaving a trace.
Entry points call ``configure_logging()`` before doing anything else.
"""

import logging
import sys

from app.config import settings


def configure_logging() -> None:
    level = getattr(logging, settings.aeris_log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
