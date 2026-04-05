from contextlib import contextmanager
from typing import Any, Callable

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
