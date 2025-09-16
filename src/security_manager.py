"""
Security and Compliance Module for IRAM

This module implements security and compliance features, including consent checks,
GDPR/CCPA notices, and runtime guardrails.
"""

from .utils import get_logger
from .config import get_config

logger = get_logger(__name__)


class SecurityManager:
    """Manages security and compliance features."""
    
    def __init__(self, config=None):
        self.config = get_config()
        logger.info("Security manager initialized")

    def check_consent(self, action: str) -> bool:
        """Placeholder for consent check logic."""
        # In a real application, this would check a database or other storage
        # for user consent for specific actions.
        logger.info(f"Performing consent check for action: {action}")
        return True

    def get_gdpr_ccpa_notice(self) -> str:
        """Returns a generic GDPR/CCPA notice."""
        return ("As part of our data processing, we may collect and store "
                "publicly available information. You have the right to access, "
                "rectify, or erase your personal data. Please contact us for more information.")

    def enforce_public_only(self, is_public: bool) -> bool:
        """Enforce public-only mode if configured."""
        if self.config.instagram.public_fallback and not is_public:
            logger.warning("Public-only mode is enforced. Action may be restricted.")
            return False
        return True

    def check_rate_limit(self) -> bool:
        """Placeholder for checking runtime guardrails like rate limits."""
        # This would be integrated with a more sophisticated rate limiting system.
        return True


# Global security manager instance
_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager
