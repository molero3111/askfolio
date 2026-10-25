import os


def test_real_dotenv_secrets_did_not_leak_into_the_test_process(monkeypatch):
    # Runs last (name sorts last). The shell never exports this, only .env could set it.
    assert os.environ.get("TELEGRAM_BOT_TOKEN") in (None, "")
