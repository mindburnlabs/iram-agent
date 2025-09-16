"""
Unit tests for IRAM modules.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.budget_monitor import BudgetMonitor
from src.security_manager import SecurityManager
from src.update_manager import UpdateManager
from src.vector_store import VectorStore


class TestBudgetMonitor:
    def test_initialization(self):
        monitor = BudgetMonitor()
        assert monitor.daily_budget > 0

    def test_record_request(self):
        monitor = BudgetMonitor()
        initial_cost = monitor.get_status()["today_cost"]
        monitor.record_request()
        assert monitor.get_status()["today_cost"] > initial_cost


class TestSecurityManager:
    def test_initialization(self):
        manager = SecurityManager()
        assert manager is not None

    def test_consent_check(self):
        manager = SecurityManager()
        assert manager.check_consent("test_action") == True


class TestUpdateManager:
    def test_initialization(self):
        manager = UpdateManager()
        assert manager is not None

    def test_get_installed_version(self):
        manager = UpdateManager()
        # Test with a known package
        version = manager._get_installed_version("pytest")
        assert version is not None


class TestVectorStore:
    def test_initialization(self):
        store = VectorStore()
        assert store is not None

# Add more unit tests here
