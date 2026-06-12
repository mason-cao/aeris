"""Process-level logging for the collector entry points.

The scheduled runs redirect stdout/stderr into collector.log; without an
explicit config only WARNING+ and the SQLAlchemy echo ever land there, which
is how Sentinel-5P ran catalog-only for three weeks without leaving a trace.
Entry points call ``configure_logging()`` before doing anything else.
"""

import logging
import sys

from app.config import settings

# Attributes every LogRecord carries; anything beyond these came in via
# ``extra={...}`` and is the part worth reading (error text, product ids).
_STANDARD_ATTRS = frozenset(
    vars(logging.makeLogRecord({})).keys()
) | {"message", "asctime", "taskName"}


class _ExtraFormatter(logging.Formatter):
    """Append ``extra={...}`` fields as key=value pairs.

    The collectors put their context (source, error, product_id) in
    ``extra``, which plain format strings silently drop — a warning like
    "S5P granule extraction failed" is useless without the error beside it.
    """

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_ATTRS
        }
        if extras:
            joined = " ".join(f"{k}={v}" for k, v in sorted(extras.items()))
            return f"{base} | {joined}"
        return base


def configure_logging() -> None:
    level = getattr(logging, settings.aeris_log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        _ExtraFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logging.basicConfig(level=level, handlers=[handler])
