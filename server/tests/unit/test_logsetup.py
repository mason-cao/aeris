import logging

from app.collectors.logsetup import _ExtraFormatter


def _record(msg: str, **extra: str) -> logging.LogRecord:
    record = logging.LogRecord(
        name="app.collectors.sentinel5p",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


class TestExtraFormatter:
    def test_appends_extra_fields_as_key_value_pairs(self) -> None:
        formatter = _ExtraFormatter("%(levelname)s %(name)s %(message)s")
        out = formatter.format(
            _record(
                "S5P granule extraction failed",
                product_id="abc-123",
                error="401 Unauthorized",
            )
        )
        assert out.endswith("| error=401 Unauthorized product_id=abc-123")

    def test_plain_record_has_no_trailing_separator(self) -> None:
        formatter = _ExtraFormatter("%(levelname)s %(message)s")
        out = formatter.format(_record("Collection complete"))
        assert out == "WARNING Collection complete"
