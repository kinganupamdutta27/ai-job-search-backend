"""Application configuration loaded from environment variables."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration for the Job Search Automation platform."""

    # --- OpenAI ---
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")

    # --- LangSmith ---
    langsmith_tracing: bool = Field(default=True, description="Enable LangSmith tracing")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", description="LangSmith API endpoint"
    )
    langsmith_api_key: str = Field(default="", description="LangSmith API key")
    langsmith_project: str = Field(
        default="JOBSEARCH", description="LangSmith project name"
    )

    # --- Search ---
    serp_api_key: str = Field(default="", description="SerpAPI key for web search")
    tavily_api_key: str = Field(default="", description="Tavily API key for AI-optimized search")

    # --- SMTP ---
    smtp_host: str = Field(default="smtp.gmail.com", description="SMTP server host")
    smtp_port: int = Field(default=587, description="SMTP server port")
    smtp_email: str = Field(default="", description="Sender email address")
    smtp_password: str = Field(
        default="", description="SMTP password (Gmail App Password for 2FA)"
    )

    # --- Storage ---
    upload_dir: str = Field(default="./uploads", description="CV upload directory")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def setup_langsmith(self) -> None:
        """Set LangSmith environment variables for tracing."""
        if self.langsmith_api_key:
            os.environ["LANGSMITH_TRACING"] = str(self.langsmith_tracing).lower()
            os.environ["LANGSMITH_ENDPOINT"] = self.langsmith_endpoint
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
