"""
Basic tests for IRAM components.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils import validate_instagram_username, format_number, extract_hashtags


class TestUtils:
    """Test utility functions."""
    
    def test_validate_instagram_username(self):
        """Test Instagram username validation."""
        # Valid usernames
        assert validate_instagram_username("test_user") == True
        assert validate_instagram_username("user123") == True
        assert validate_instagram_username("user.name") == True
        assert validate_instagram_username("a") == True
        
        # Invalid usernames
        assert validate_instagram_username("") == False
        assert validate_instagram_username("user..name") == False
        assert validate_instagram_username("user.") == False
        assert validate_instagram_username("user@name") == False
        assert validate_instagram_username("a" * 31) == False  # Too long
    
    def test_format_number(self):
        """Test number formatting."""
        assert format_number(500) == "500"
        assert format_number(1500) == "1.5K"
        assert format_number(1500000) == "1.5M"
        assert format_number(1500000000) == "1.5B"
    
    def test_extract_hashtags(self):
        """Test hashtag extraction."""
        text = "Check out this #amazing #photo #instagram"
        hashtags = extract_hashtags(text)
        assert len(hashtags) == 3
        assert "#amazing" in hashtags
        assert "#photo" in hashtags
        assert "#instagram" in hashtags


class TestImports:
    """Test that all modules can be imported."""
    
    def test_import_agent_orchestrator(self):
        """Test importing agent orchestrator."""
        from src.agent_orchestrator import create_instagram_agent
        assert create_instagram_agent is not None
    
    def test_import_scraping_module(self):
        """Test importing scraping module."""
        from src.scraping_module import InstagramScraper
        assert InstagramScraper is not None
    
    def test_import_analysis_module(self):
        """Test importing analysis module."""
        from src.analysis_module import ContentAnalyzer
        assert ContentAnalyzer is not None
    
    def test_import_evasion_manager(self):
        """Test importing evasion manager."""
        from src.evasion_manager import EvasionManager
        assert EvasionManager is not None


class TestConfiguration:
    """Test configuration and environment."""
    
    def test_environment_variables_exist(self):
        """Test that example environment file exists."""
        env_example = Path(__file__).parent.parent / ".env.example"
        assert env_example.exists(), ".env.example file should exist"
    
    def test_requirements_file_exists(self):
        """Test that requirements file exists."""
        requirements = Path(__file__).parent.parent / "requirements.txt"
        assert requirements.exists(), "requirements.txt file should exist"


if __name__ == "__main__":
    pytest.main([__file__])