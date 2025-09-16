"""
IRAM Configuration Schema

Centralized configuration using Pydantic Settings with environment variable support,
validation, and defaults for all application components.
"""

from typing import Optional, List, Dict, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings
import os


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: Optional[str] = Field(None, env="DATABASE_URL")
    pool_size: int = Field(10, env="DB_POOL_SIZE") 
    max_overflow: int = Field(20, env="DB_MAX_OVERFLOW")
    echo: bool = Field(False, env="DB_ECHO")
    
    class Config:
        env_prefix = "DB_"


class InstagramSettings(BaseSettings):
    """Instagram scraping configuration."""
    
    username: Optional[str] = Field(None, env="INSTAGRAM_USERNAME")
    password: Optional[str] = Field(None, env="INSTAGRAM_PASSWORD")
    session_file: str = Field("instagram_session.json", env="INSTAGRAM_SESSION_FILE")
    public_fallback: bool = Field(True, env="INSTAGRAM_PUBLIC_FALLBACK")
    
    # Rate limiting and evasion
    requests_per_minute: int = Field(30, env="INSTAGRAM_RPM")
    delay_min: float = Field(1.0, env="INSTAGRAM_DELAY_MIN") 
    delay_max: float = Field(3.0, env="INSTAGRAM_DELAY_MAX")
    user_agents: List[str] = Field(
        default_factory=lambda: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36", 
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        ],
        env="INSTAGRAM_USER_AGENTS"
    )
    
    class Config:
        env_prefix = "INSTAGRAM_"


class LLMSettings(BaseSettings):
    """LLM provider configuration."""
    
    # API keys
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    openrouter_api_key: Optional[str] = Field(None, env="OPENROUTER_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    
    # Model settings
    default_model: str = Field("openrouter/anthropic/claude-3.5-sonnet", env="LLM_MODEL")
    anthropic_model: str = Field("claude-3-5-sonnet-20241022", env="ANTHROPIC_MODEL")
    openai_model: str = Field("gpt-4-turbo-preview", env="OPENAI_MODEL")
    temperature: float = Field(0.7, env="LLM_TEMPERATURE")
    max_tokens: int = Field(4000, env="LLM_MAX_TOKENS")
    
    # Cost controls
    max_cost_per_request: float = Field(1.0, env="LLM_MAX_COST_PER_REQUEST")
    daily_budget: float = Field(50.0, env="LLM_DAILY_BUDGET")
    
    # Provider preferences (Anthropic > OpenAI > OpenRouter)
    prefer_anthropic: bool = Field(True, env="LLM_PREFER_ANTHROPIC")
    prefer_openrouter: bool = Field(False, env="LLM_PREFER_OPENROUTER")
    
    @validator("anthropic_api_key", "openrouter_api_key", "openai_api_key", pre=True)
    def validate_api_keys(cls, v):
        """Validate API key format if provided."""
        if v and len(v) < 10:
            raise ValueError("API key appears to be too short")
        return v
    
    class Config:
        env_prefix = "LLM_"


class ServerSettings(BaseSettings):
    """FastAPI server configuration."""
    
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    debug: bool = Field(False, env="DEBUG")
    reload: bool = Field(False, env="RELOAD")
    
    # CORS
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")
    cors_credentials: bool = Field(True, env="CORS_CREDENTIALS") 
    
    # Security
    allowed_hosts: List[str] = Field(["*"], env="ALLOWED_HOSTS")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(10, env="RATE_LIMIT_BURST")
    
    # Request limits
    max_request_size: int = Field(10 * 1024 * 1024, env="MAX_REQUEST_SIZE")  # 10MB
    request_timeout: int = Field(300, env="REQUEST_TIMEOUT")  # 5 minutes
    
    class Config:
        env_prefix = "SERVER_"


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    level: str = Field("INFO", env="LOG_LEVEL")
    format: str = Field("json", env="LOG_FORMAT")  # json or text
    file: Optional[str] = Field(None, env="LOG_FILE")
    
    # Structured logging fields
    include_trace_id: bool = Field(True, env="LOG_INCLUDE_TRACE_ID")
    include_timestamp: bool = Field(True, env="LOG_INCLUDE_TIMESTAMP")
    
    # Secret redaction
    redact_secrets: bool = Field(True, env="LOG_REDACT_SECRETS")
    secret_patterns: List[str] = Field(
        default_factory=lambda: [
            r"api[_-]?key",
            r"password", 
            r"secret",
            r"token",
            r"auth",
            r"Bearer\s+\S+"
        ],
        env="LOG_SECRET_PATTERNS"
    )
    
    @validator("level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    class Config:
        env_prefix = "LOG_"


class RedisSettings(BaseSettings):
    """Redis caching configuration."""
    
    url: Optional[str] = Field(None, env="REDIS_URL")
    host: str = Field("localhost", env="REDIS_HOST")
    port: int = Field(6379, env="REDIS_PORT")
    db: int = Field(0, env="REDIS_DB")
    password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    
    # Connection pooling
    max_connections: int = Field(10, env="REDIS_MAX_CONNECTIONS")
    
    # Default TTLs (in seconds)
    default_ttl: int = Field(3600, env="REDIS_DEFAULT_TTL")  # 1 hour
    session_ttl: int = Field(86400, env="REDIS_SESSION_TTL")  # 24 hours
    rate_limit_ttl: int = Field(60, env="REDIS_RATE_LIMIT_TTL")  # 1 minute
    
    class Config:
        env_prefix = "REDIS_"


class StorageSettings(BaseSettings):
    """File and artifact storage configuration."""
    
    type: str = Field("local", env="STORAGE_TYPE")  # local, s3, gcs
    base_path: str = Field("./storage", env="STORAGE_BASE_PATH")
    
    # S3 settings (if type=s3)
    s3_bucket: Optional[str] = Field(None, env="S3_BUCKET")
    s3_region: Optional[str] = Field(None, env="S3_REGION")
    s3_access_key: Optional[str] = Field(None, env="S3_ACCESS_KEY")
    s3_secret_key: Optional[str] = Field(None, env="S3_SECRET_KEY")
    
    # File size limits
    max_file_size: int = Field(100 * 1024 * 1024, env="STORAGE_MAX_FILE_SIZE")  # 100MB
    
    # Retention
    retention_days: int = Field(90, env="STORAGE_RETENTION_DAYS")
    
    class Config:
        env_prefix = "STORAGE_"


class FirecrawlSettings(BaseSettings):
    """Firecrawl web scraping configuration."""
    
    api_key: Optional[str] = Field(None, env="FIRECRAWL_API_KEY")
    base_url: str = Field("https://api.firecrawl.dev", env="FIRECRAWL_BASE_URL")
    
    # Default scraping options
    default_formats: List[str] = Field(["markdown", "html"], env="FIRECRAWL_DEFAULT_FORMATS")
    default_timeout: int = Field(30, env="FIRECRAWL_TIMEOUT")  # seconds
    
    # Crawling limits
    max_pages_per_crawl: int = Field(100, env="FIRECRAWL_MAX_PAGES")  # up to 100k in v2.2.0
    max_concurrent_requests: int = Field(5, env="FIRECRAWL_CONCURRENT_REQUESTS")
    
    # PDF parsing settings
    pdf_max_pages: int = Field(50, env="FIRECRAWL_PDF_MAX_PAGES")
    
    class Config:
        env_prefix = "FIRECRAWL_"


class FeatureFlags(BaseSettings):
    """Feature toggle configuration."""
    
    # Analysis features
    enable_topic_modeling: bool = Field(False, env="FEATURE_TOPIC_MODELING")
    enable_computer_vision: bool = Field(False, env="FEATURE_COMPUTER_VISION") 
    enable_sentiment_analysis: bool = Field(True, env="FEATURE_SENTIMENT_ANALYSIS")
    
    # Scraping methods
    enable_playwright: bool = Field(True, env="FEATURE_PLAYWRIGHT")
    enable_instagrapi: bool = Field(True, env="FEATURE_INSTAGRAPI")
    enable_firecrawl: bool = Field(True, env="FEATURE_FIRECRAWL")
    
    # Background processing
    enable_background_jobs: bool = Field(True, env="FEATURE_BACKGROUND_JOBS")
    enable_scheduling: bool = Field(True, env="FEATURE_SCHEDULING")
    
    # API features  
    enable_webhooks: bool = Field(False, env="FEATURE_WEBHOOKS")
    enable_rate_limiting: bool = Field(True, env="FEATURE_RATE_LIMITING")
    
    class Config:
        env_prefix = "FEATURE_"


class IRamConfig(BaseSettings):
    """Main IRAM configuration combining all subsystem settings."""
    
    # Environment info
    environment: str = Field("development", env="ENVIRONMENT")
    version: str = Field("1.0.0", env="VERSION")
    
    # Subsystem configurations
    database: DatabaseSettings = DatabaseSettings()
    instagram: InstagramSettings = InstagramSettings()
    llm: LLMSettings = LLMSettings()
    server: ServerSettings = ServerSettings()
    logging: LoggingSettings = LoggingSettings()
    redis: RedisSettings = RedisSettings()
    storage: StorageSettings = StorageSettings()
    firecrawl: FirecrawlSettings = FirecrawlSettings()
    features: FeatureFlags = FeatureFlags()
    
    @validator("environment")
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def has_database(self) -> bool:
        """Check if database is configured."""
        return bool(self.database.url)
    
    def has_redis(self) -> bool:
        """Check if Redis is configured."""
        return bool(self.redis.url) or (self.redis.host != "localhost")
    
    def has_llm_provider(self) -> bool:
        """Check if at least one LLM provider is configured."""
        return bool(
            self.llm.anthropic_api_key or 
            self.llm.openrouter_api_key or 
            self.llm.openai_api_key
        )
    
    def has_instagram_auth(self) -> bool:
        """Check if Instagram credentials are configured."""
        return bool(self.instagram.username and self.instagram.password)
    
    def has_firecrawl(self) -> bool:
        """Check if Firecrawl is configured."""
        return bool(self.firecrawl.api_key) and self.features.enable_firecrawl
    
    def get_primary_llm_provider(self) -> str:
        """Get the primary LLM provider to use."""
        # Priority: Anthropic > OpenAI > OpenRouter (unless preferences override)
        if self.llm.prefer_anthropic and self.llm.anthropic_api_key:
            return "anthropic"
        elif self.llm.prefer_openrouter and self.llm.openrouter_api_key:
            return "openrouter"
        elif self.llm.anthropic_api_key:
            return "anthropic"
        elif self.llm.openai_api_key:
            return "openai"
        elif self.llm.openrouter_api_key:
            return "openrouter"
        else:
            raise ValueError("No LLM provider configured")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"


# Global configuration instance
_config: Optional[IRamConfig] = None


def get_config() -> IRamConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = IRamConfig()
    return _config


def reload_config() -> IRamConfig:
    """Reload configuration from environment."""
    global _config
    _config = None
    return get_config()


# Export commonly used settings for convenience
def get_database_settings() -> DatabaseSettings:
    return get_config().database


def get_instagram_settings() -> InstagramSettings:
    return get_config().instagram


def get_llm_settings() -> LLMSettings:
    return get_config().llm


def get_server_settings() -> ServerSettings:
    return get_config().server


def get_logging_settings() -> LoggingSettings:
    return get_config().logging