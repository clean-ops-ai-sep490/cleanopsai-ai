from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "CleanOpsAI"
    app_version: str = "0.1.0"
    debug: bool = False
    model_path: str = "app/models/trained"
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "case_sensitive": False}


settings = Settings()
