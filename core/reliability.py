"""Timeouts, transient-error retries, and job-scoped logging."""

import asyncio
import logging
from contextvars import ContextVar

job_id_var: ContextVar[str] = ContextVar("job_id", default="-")

SCORE_TIMEOUT_SECONDS = 180
EMAIL_TIMEOUT_SECONDS = 240
DISCOVERY_TIMEOUT_SECONDS = 300
BULK_EMAIL_TIMEOUT_SECONDS = 240

MAX_AUTOPILOT = 10
MAX_AGENT = 25
MAX_BULK = 100

BULK_CONCURRENCY = 4
AGENT_CONCURRENCY = 2

# First try plus two retries.
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0

_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_TRANSIENT_TEXT = (
    "connection reset",
    "connection aborted",
    "rate limit",
    "too many requests",
    "status code 429",
    "status code 500",
    "status code 502",
    "status code 503",
    "status code 504",
    "internal server error",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
)


class JobIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = job_id_var.get()
        return True


def configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    for handler in root.handlers:
        if not any(isinstance(existing, JobIdFilter) for existing in handler.filters):
            handler.addFilter(JobIdFilter())


def is_transient(exc: BaseException) -> bool:
    """True for rate limits, server errors, and dropped connections.

    Timeouts and parse failures are not retried: a timeout should fail the
    prospect, and a bad model reply will not improve on a second try.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return False

    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        if int(status) in _TRANSIENT_STATUS:
            return True
    except (TypeError, ValueError):
        pass

    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        return True

    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_TEXT)


async def with_retries(operation, *, attempts: int | None = None, base_delay: float | None = None):
    """Run an async operation, retrying only transient failures."""
    if attempts is None:
        attempts = RETRY_ATTEMPTS
    if base_delay is None:
        base_delay = RETRY_BASE_DELAY_SECONDS
    logger = logging.getLogger(__name__)
    delay = base_delay
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            if attempt >= attempts or not is_transient(exc):
                raise
            logger.warning(
                "job=%s transient error (attempt %s/%s): %s",
                job_id_var.get(),
                attempt,
                attempts,
                exc,
            )
            await asyncio.sleep(delay)
            delay *= 2
