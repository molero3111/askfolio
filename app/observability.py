import traceback
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

try:
    from langsmith import traceable as _traceable
    from langsmith import tracing_context as _tracing_context
except Exception:  # pragma: no cover - optional dependency/runtime
    _traceable = None
    _tracing_context = None


def traceable(*args: Any, **kwargs: Any) -> Callable:
    """Return LangSmith traceable decorator when available; otherwise no-op."""
    if _traceable is None:
        def _decorator(fn: Callable) -> Callable:
            return fn
        return _decorator
    return _traceable(*args, **kwargs)


@contextmanager
def tracing_context(**kwargs: Any):
    """LangSmith tracing context when available; otherwise no-op context."""
    if _tracing_context is None:
        yield
        return
    with _tracing_context(**kwargs):
        yield


def is_langsmith_exception(exc: BaseException) -> bool:
    """Best-effort detection for tracing backend/runtime errors."""
    stack = traceback.format_exception(type(exc), exc, exc.__traceback__)
    text = "".join(stack).lower()
    if "langsmith" in text or "smith.langchain.com" in text:
        return True
    cause = exc.__cause__ or exc.__context__
    if cause is None:
        return False
    return is_langsmith_exception(cause)
