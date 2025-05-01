from dynaconf import Dynaconf
import os
from pathlib import Path

# Define base directory
BASE_DIR = Path(__file__).parent

# Initialize configuration
settings = Dynaconf(
    envvar_prefix="MES",
    settings_files=[
        f"{BASE_DIR}/settings.toml",
        f"{BASE_DIR}/settings.local.toml",
        f"{BASE_DIR}/.secrets.toml",
    ],
    environments=True,
    load_dotenv=True,
    env_switcher="MES_ENV",
    dotenv_path=f"{BASE_DIR}/.env",
)

# Ensure configuration is validated
def validate_settings():
    """Validate required settings are present."""
    required_settings = [
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "AWS_REGION",
        "TRADOVATE_API_URL",
    ]
    
    missing = [s for s in required_settings if not settings.get(s)]
    if missing:
        raise ValueError(f"Missing required configuration settings: {', '.join(missing)}")

# Call validation on import
validate_settings()

# Export settings as global
config = settings 