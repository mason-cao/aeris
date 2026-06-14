from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://aeris:aeris@localhost:5432/aeris"
    database_url_sync: str = "postgresql://aeris:aeris@localhost:5432/aeris"

    # Air Quality APIs
    openaq_api_key: str = ""
    # OpenAQ /latest returns a station's most recent reading regardless of age,
    # so an offline station re-emits a stale value every hourly run. Drop any
    # reading older than this (seconds). Default 6h: well past any real station
    # cadence + publication lag, but far short of the weeks/months-old values
    # dead stations carry.
    openaq_max_reading_age_s: float = 6 * 3600.0

    # Weather
    openweather_api_key: str = ""

    # NASA / Satellite
    nasa_earthdata_token: str = ""
    cdse_username: str = ""
    cdse_password: str = ""

    # Mapping
    mapbox_token: str = ""

    # LLM (cloud comparison only)
    openai_api_key: str = ""
    google_api_key: str = ""

    # App
    aeris_env: str = "development"
    aeris_log_level: str = "INFO"
    aeris_target_lat: float = 29.7604
    aeris_target_lon: float = -95.3698
    aeris_target_radius_km: float = 50.0

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_async_database_url(cls, value: object) -> object:
        if isinstance(value, str) and value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+asyncpg://", 1)
        return value


settings = Settings()
