"""Application configuration via Pydantic BaseSettings."""

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Global configuration for the Deltx pipeline."""

    model_config = {"env_prefix": "DELTX_"}
