import importlib

import pytest


@pytest.mark.parametrize("mod", [
    "app.config", "app.conversation", "app.calendar_client", "app.rag",
    "app.telegram_client", "app.observability", "app.graph",
    "ingest.json_to_documents",
])
def test_module_imports(mod):
    importlib.import_module(mod)
