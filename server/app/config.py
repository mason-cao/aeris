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
    airnow_api_key: str = ""
    purpleair_api_key: str = ""

    # Weather
    openweather_api_key: str = ""

    # Traffic
    tomtom_api_key: str = ""

    # Energy
    eia_api_key: str = ""

    # NASA / Satellite
    nasa_earthdata_token: str = ""

    # Mapping
    mapbox_token: str = ""

    # LLM (cloud comparison only)
    openai_api_key: str = ""
    google_api_key: str = ""

    # App
    aeris_env: str = "development"
    aeris_log_level: str = "INFO"
    aeris_target_lat: float = 34.0515
    aeris_target_lon: float = -84.0713
    aeris_target_radius_km: float = 50.0


settings = Settings()
