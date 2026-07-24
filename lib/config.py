"""Environment-based application settings."""

from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

BASE_URL_ENV = "WEQUANT_DELIVER_BASE_URL"
USERNAME_ENV = "WEQUANT_DELIVER_USERNAME"
PASSWORD_ENV = "WEQUANT_DELIVER_PASSWORD"
REQUIRED_ENV_VARS = (BASE_URL_ENV, USERNAME_ENV, PASSWORD_ENV)


class ConfigurationError(RuntimeError):
    """Raised when required configuration is unavailable."""


@dataclass(frozen=True)
class DeliverSettings:
    base_url: str
    username: str
    password: str


def load_deliver_settings(env_path: Path = DEFAULT_ENV_PATH) -> DeliverSettings:
    """Load Deliver settings without overriding process environment variables."""
    if env_path.is_file():
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path, override=False)

    values = {name: os.getenv(name, "").strip() for name in REQUIRED_ENV_VARS}
    missing = [name for name, value in values.items() if not value]
    if missing:
        names = ", ".join(missing)
        raise ConfigurationError(f"Required environment variables are not set: {names}")

    return DeliverSettings(
        base_url=values[BASE_URL_ENV].rstrip("/"),
        username=values[USERNAME_ENV],
        password=values[PASSWORD_ENV],
    )
