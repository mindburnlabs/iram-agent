"""
Instagram Research Agent MCP (IRAM) - Utilities

This module contains utility functions and helpers used throughout the application.
"""

import os
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Configure logger
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logger.level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
    
    return logger


def validate_instagram_username(username: str) -> bool:
    """Validate Instagram username format."""
    if not username or not isinstance(username, str):
        return False
    
    # Remove @ if present
    username = username.lstrip('@')
    
    # Instagram username rules:
    # - 1-30 characters
    # - Only letters, numbers, underscores, periods
    # - Cannot end with period
    # - Cannot have consecutive periods
    pattern = r'^[a-zA-Z0-9._]{1,30}$'
    
    if not re.match(pattern, username):
        return False
    
    if username.endswith('.'):
        return False
    
    if '..' in username:
        return False
    
    return True


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace invalid characters
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    return sanitized


def format_number(number: int) -> str:
    """Format numbers in human-readable format (K, M, B)."""
    if number < 1000:
        return str(number)
    elif number < 1000000:
        return f"{number/1000:.1f}K"
    elif number < 1000000000:
        return f"{number/1000000:.1f}M"
    else:
        return f"{number/1000000000:.1f}B"


def calculate_engagement_rate(likes: int, comments: int, followers: int) -> float:
    """Calculate engagement rate as percentage."""
    if followers == 0:
        return 0.0
    
    total_engagement = likes + comments
    return (total_engagement / followers) * 100


def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text."""
    if not text:
        return []
    
    hashtag_pattern = r'#\w+'
    hashtags = re.findall(hashtag_pattern, text, re.IGNORECASE)
    return [tag.lower() for tag in hashtags]


def extract_mentions(text: str) -> List[str]:
    """Extract @mentions from text."""
    if not text:
        return []
    
    mention_pattern = r'@\w+'
    mentions = re.findall(mention_pattern, text, re.IGNORECASE)
    return [mention.lower() for mention in mentions]


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    
    return text.strip()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def parse_instagram_url(url: str) -> Dict[str, Any]:
    """Parse Instagram URL to extract relevant information."""
    result = {"valid": False, "type": None, "identifier": None}
    
    if not url or not isinstance(url, str):
        return result
    
    # Instagram URL patterns
    patterns = {
        "profile": r"instagram\.com/([a-zA-Z0-9_.]+)/?$",
        "post": r"instagram\.com/p/([a-zA-Z0-9_-]+)/?",
        "reel": r"instagram\.com/reel/([a-zA-Z0-9_-]+)/?",
        "story": r"instagram\.com/stories/([a-zA-Z0-9_.]+)/?",
    }
    
    for url_type, pattern in patterns.items():
        match = re.search(pattern, url)
        if match:
            result["valid"] = True
            result["type"] = url_type
            result["identifier"] = match.group(1)
            break
    
    return result


def timestamp_to_datetime(timestamp: Any) -> Optional[datetime]:
    """Convert various timestamp formats to datetime."""
    if not timestamp:
        return None
    
    try:
        # If it's already a datetime
        if isinstance(timestamp, datetime):
            return timestamp
        
        # If it's a string
        if isinstance(timestamp, str):
            # Try ISO format first
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                pass
        
        # If it's a number (Unix timestamp)
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp)
        
        return None
    except Exception:
        return None


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def deep_get(dictionary: Dict[str, Any], keys: str, default: Any = None) -> Any:
    """Get nested dictionary value using dot notation."""
    try:
        keys_list = keys.split('.')
        value = dictionary
        
        for key in keys_list:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    except Exception:
        return default


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries, with later ones taking precedence."""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


def batch_process(items: List[Any], batch_size: int = 100):
    """Yield batches of items for processing."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def is_business_hours(hour: int = None) -> bool:
    """Check if current time (or specified hour) is within business hours."""
    if hour is None:
        hour = datetime.now().hour
    
    # Consider 9 AM to 5 PM as business hours
    return 9 <= hour <= 17


def rate_limit_key(identifier: str, action: str) -> str:
    """Generate a rate limiting key for an identifier and action."""
    return f"rate_limit:{identifier}:{action}"


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff delay."""
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)


def create_error_response(error_message: str, error_code: str = "UNKNOWN_ERROR") -> Dict[str, Any]:
    """Create a standardized error response."""
    return {
        "success": False,
        "error": {
            "code": error_code,
            "message": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
    }


def create_success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """Create a standardized success response."""
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }


def load_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Safely load JSON from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        get_logger(__name__).error(f"Failed to load JSON file {file_path}: {e}")
        return None


def save_json_file(data: Any, file_path: str) -> bool:
    """Safely save data to JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        get_logger(__name__).error(f"Failed to save JSON file {file_path}: {e}")
        return False


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0


def ensure_directory(directory: str) -> bool:
    """Ensure directory exists, create if it doesn't."""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        get_logger(__name__).error(f"Failed to create directory {directory}: {e}")
        return False