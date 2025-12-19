import os

# Ensure local `.env` (loaded by python-dotenv in src.core.config) cannot override
# safe defaults during test collection/import.
#
# python-dotenv's load_dotenv() does not override existing env vars by default, so
# setting these here (conftest imports before test modules) prevents accidental use
# of real credentials in tests.
os.environ.setdefault("ALPACA_API_KEY", "test")
os.environ.setdefault("ALPACA_SECRET_KEY", "test")
os.environ.setdefault("ALPACA_PAPER", "True")
os.environ.setdefault("DATA_INGESTION_URL", "http://central.test")
