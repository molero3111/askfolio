"""Logging filter that keeps secrets (bot token, API keys) out of log output."""
import logging
import re
import traceback

_TOKEN_RE = re.compile(r"bot\d+:[A-Za-z0-9_-]{30,}")


class RedactionFilter(logging.Filter):
    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self._secrets = sorted({s for s in secrets if s and s.strip()}, key=len, reverse=True)

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return _TOKEN_RE.sub("bot***", text)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.redact(record.getMessage())
        record.args = ()
        if record.exc_info and record.exc_info[0] is not None:
            record.exc_text = self.redact("".join(traceback.format_exception(*record.exc_info)).rstrip("\n"))
        elif record.exc_text:
            record.exc_text = self.redact(record.exc_text)
        if record.stack_info:
            record.stack_info = self.redact(record.stack_info)
        return True


def install_redaction(secrets: list[str]) -> None:
    """Attach the filter to the root logger and every root handler."""
    flt = RedactionFilter(secrets)
    root = logging.getLogger()
    root.addFilter(flt)
    for handler in root.handlers:
        handler.addFilter(flt)
