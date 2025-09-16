"""
IRAM Structured Logging Configuration

Provides JSON-structured logging with secret redaction, trace IDs, 
and correlation IDs for improved observability.
"""

import json
import logging
import logging.config
import re
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, List, Optional
import traceback

from .config import get_logging_settings


# Context variables for request tracing
trace_id: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class SecretRedactingFilter(logging.Filter):
    """Filter to redact sensitive information from log records."""
    
    def __init__(self, patterns: List[str]):
        super().__init__()
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Redact secrets from the log record."""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self._redact_secrets(record.msg)
        
        if hasattr(record, 'args') and record.args:
            record.args = tuple(
                self._redact_secrets(str(arg)) if isinstance(arg, str) else arg 
                for arg in record.args
            )
        
        return True
    
    def _redact_secrets(self, text: str) -> str:
        """Redact secrets from text using configured patterns."""
        for pattern in self.compiled_patterns:
            text = pattern.sub('[REDACTED]', text)
        return text


class JSONFormatter(logging.Formatter):
    """JSON log formatter with structured output."""
    
    def __init__(self, include_trace_id: bool = True, include_timestamp: bool = True):
        super().__init__()
        self.include_trace_id = include_trace_id
        self.include_timestamp = include_timestamp
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        if self.include_timestamp:
            log_data['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        if self.include_trace_id:
            if trace_id.get():
                log_data['trace_id'] = trace_id.get()
            if correlation_id.get():
                log_data['correlation_id'] = correlation_id.get()
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName', 
                          'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info'):
                log_data[key] = value
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter."""
    
    def __init__(self, include_trace_id: bool = True):
        super().__init__()
        self.include_trace_id = include_trace_id
        
        fmt_parts = [
            '%(asctime)s',
            '[%(levelname)s]',
            '%(name)s',
        ]
        
        if include_trace_id:
            fmt_parts.append('[%(trace_id)s]')
        
        fmt_parts.append('%(message)s')
        
        self.fmt = ' '.join(fmt_parts)
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as human-readable text."""
        if self.include_trace_id:
            record.trace_id = trace_id.get() or 'no-trace'
        
        return super().format(record)


def setup_logging() -> None:
    """Configure logging based on application settings."""
    settings = get_logging_settings()
    
    # Create formatters
    if settings.format == 'json':
        formatter = JSONFormatter(
            include_trace_id=settings.include_trace_id,
            include_timestamp=settings.include_timestamp
        )
    else:
        formatter = TextFormatter(include_trace_id=settings.include_trace_id)
    
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    if settings.redact_secrets:
        secret_filter = SecretRedactingFilter(settings.secret_patterns)
        console_handler.addFilter(secret_filter)
    
    handlers.append(console_handler)
    
    # File handler (if configured)
    if settings.file:
        file_handler = logging.FileHandler(settings.file)
        file_handler.setFormatter(formatter)
        
        if settings.redact_secrets:
            file_handler.addFilter(secret_filter)
        
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.level),
        handlers=handlers,
        force=True
    )
    
    # Set level for specific loggers
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


def set_trace_id(new_trace_id: Optional[str] = None) -> str:
    """Set the trace ID for the current context."""
    if new_trace_id is None:
        new_trace_id = str(uuid.uuid4())[:8]
    
    trace_id.set(new_trace_id)
    return new_trace_id


def set_correlation_id(new_correlation_id: Optional[str] = None) -> str:
    """Set the correlation ID for the current context."""
    if new_correlation_id is None:
        new_correlation_id = str(uuid.uuid4())[:8]
    
    correlation_id.set(new_correlation_id)
    return new_correlation_id


def get_trace_id() -> Optional[str]:
    """Get the current trace ID."""
    return trace_id.get()


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return correlation_id.get()


class LoggerMixin:
    """Mixin class to add logger functionality to other classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get a logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__module__ + '.' + self.__class__.__name__)
        return self._logger


class StructuredLogger:
    """Wrapper for structured logging with additional context."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.extra_context: Dict[str, Any] = {}
    
    def with_context(self, **kwargs) -> 'StructuredLogger':
        """Create a new logger with additional context."""
        new_logger = StructuredLogger(self.logger.name)
        new_logger.extra_context = {**self.extra_context, **kwargs}
        return new_logger
    
    def _log(self, level: int, message: str, *args, **kwargs):
        """Log with extra context."""
        extra = {**self.extra_context, **kwargs}
        self.logger.log(level, message, *args, extra=extra)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log exception with traceback."""
        kwargs['exc_info'] = True
        self._log(logging.ERROR, message, *args, **kwargs)


# Convenience functions
def create_structured_logger(name: str) -> StructuredLogger:
    """Create a structured logger instance."""
    return StructuredLogger(name)


def log_function_entry(logger: logging.Logger, function_name: str, **kwargs):
    """Log function entry with parameters."""
    logger.debug(f"Entering {function_name}", extra={'function': function_name, 'parameters': kwargs})


def log_function_exit(logger: logging.Logger, function_name: str, result=None):
    """Log function exit with result."""
    logger.debug(f"Exiting {function_name}", extra={'function': function_name, 'result': str(result)[:200]})


def log_performance(logger: logging.Logger, operation: str, duration: float, **kwargs):
    """Log performance metrics."""
    logger.info(
        f"Performance: {operation} took {duration:.3f}s",
        extra={
            'operation': operation,
            'duration_ms': duration * 1000,
            'performance': True,
            **kwargs
        }
    )