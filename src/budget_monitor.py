"""
Budget Monitoring System for IRAM

This module tracks LLM and API spend against a configured daily budget and emits
alerts via logs and an MCP endpoint.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from .utils import get_logger
from .config import get_config

logger = get_logger(__name__)


class BudgetMonitor:
    """Tracks API spend against a configured budget."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the budget monitor."""
        self.config = get_config()
        self.custom_config = config or {}
        
        # Budget settings
        self.daily_budget = self.config.llm.daily_budget
        self.cost_per_request = self.config.llm.max_cost_per_request
        
        # State tracking
        self.usage_file = "data/budget_usage.json"
        self.usage_data = self._load_usage()
        
        logger.info("Budget monitor initialized")

    def _load_usage(self) -> Dict[str, Any]:
        """Load usage data from file."""
        try:
            if Path(self.usage_file).exists():
                with open(self.usage_file, 'r') as f:
                    return json.load(f)
            else:
                return self._default_usage_data()
        except Exception as e:
            logger.warning(f"Failed to load usage data: {e}")
            return self._default_usage_data()

    def _save_usage(self):
        """Save usage data to file."""
        try:
            os.makedirs(Path(self.usage_file).parent, exist_ok=True)
            with open(self.usage_file, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save usage data: {e}")

    def _default_usage_data(self) -> Dict[str, Any]:
        """Return default usage data structure."""
        return {
            "today_cost": 0.0,
            "today_requests": 0,
            "last_update_date": datetime.utcnow().date().isoformat()
        }

    def _reset_if_new_day(self):
        """Reset daily usage if a new day has started."""
        today_str = datetime.utcnow().date().isoformat()
        if self.usage_data["last_update_date"] != today_str:
            self.usage_data = self._default_usage_data()
            logger.info("New day started, resetting budget usage.")

    def record_request(self, cost: Optional[float] = None):
        """Record a request and update usage."""
        self._reset_if_new_day()
        
        request_cost = cost if cost is not None else self.cost_per_request
        self.usage_data["today_cost"] += request_cost
        self.usage_data["today_requests"] += 1
        self.usage_data["last_update_date"] = datetime.utcnow().date().isoformat()
        
        self._save_usage()
        self.check_budget_alerts()

    def check_budget_alerts(self):
        """Check if budget is nearing or has exceeded the limit."""
        if self.daily_budget <= 0:
            return

        usage_percentage = (self.usage_data["today_cost"] / self.daily_budget) * 100

        if usage_percentage >= 100:
            logger.critical(f"Daily budget EXCEEDED! Usage: {usage_percentage:.2f}%")
        elif usage_percentage >= 90:
            logger.warning(f"Daily budget nearing limit. Usage: {usage_percentage:.2f}%")

    def get_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        self._reset_if_new_day()
        return {
            "daily_budget": self.daily_budget,
            "today_cost": self.usage_data["today_cost"],
            "today_requests": self.usage_data["today_requests"],
            "usage_percentage": (self.usage_data["today_cost"] / self.daily_budget) * 100 if self.daily_budget > 0 else 0,
            "last_update_date": self.usage_data["last_update_date"]
        }

    def is_within_budget(self) -> bool:
        """Check if current usage is within budget."""
        if self.daily_budget <= 0:
            return True
        return self.usage_data["today_cost"] < self.daily_budget

# Global budget monitor instance
_budget_monitor: Optional[BudgetMonitor] = None

def get_budget_monitor() -> BudgetMonitor:
    """Get global budget monitor instance."""
    global _budget_monitor
    if _budget_monitor is None:
        _budget_monitor = BudgetMonitor()
    return _budget_monitor
