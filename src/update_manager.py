"""
Automatic Update Manager for IRAM

This module handles automatic updates for dependencies, particularly Instagrapi,
with version checking, scheduled updates, and notification logging.
"""

import os
import sys
import json
import asyncio
import subprocess
import pkg_resources
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import requests
import schedule
import time
from threading import Thread

from .utils import get_logger
from .config import get_config

logger = get_logger(__name__)


class UpdateManager:
    """Manages automatic updates for IRAM dependencies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize update manager."""
        self.config = get_config()
        self.custom_config = config or {}
        
        # Update settings
        self.auto_update_enabled = self.custom_config.get("auto_update_enabled", True)
        self.update_check_interval_hours = self.custom_config.get("update_check_interval_hours", 24)
        self.critical_packages = ["instagrapi", "langchain", "fastapi", "anthropic", "openai"]
        
        # State tracking
        self.last_check_time = None
        self.update_status_file = "update_status.json"
        self.update_history: List[Dict[str, Any]] = []
        
        # Load previous state
        self._load_update_status()
        
        # Background scheduler
        self.scheduler_thread = None
        self.running = False
        
        logger.info("Update manager initialized")
    
    def _load_update_status(self):
        """Load update status from file."""
        try:
            if Path(self.update_status_file).exists():
                with open(self.update_status_file, 'r') as f:
                    data = json.load(f)
                    self.last_check_time = datetime.fromisoformat(data.get("last_check_time", ""))
                    self.update_history = data.get("update_history", [])
        except Exception as e:
            logger.warning(f"Failed to load update status: {e}")
    
    def _save_update_status(self):
        """Save update status to file."""
        try:
            data = {
                "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
                "update_history": self.update_history[-50:]  # Keep last 50 entries
            }
            with open(self.update_status_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save update status: {e}")
    
    async def check_for_updates(self) -> Dict[str, Any]:
        """Check for available updates for critical packages."""
        try:
            logger.info("Checking for package updates...")
            self.last_check_time = datetime.utcnow()
            
            updates_available = {}
            
            for package_name in self.critical_packages:
                try:
                    # Get current version
                    current_version = self._get_installed_version(package_name)
                    if not current_version:
                        logger.warning(f"Package {package_name} not found in environment")
                        continue
                    
                    # Get latest version from PyPI
                    latest_version = await self._get_latest_version_from_pypi(package_name)
                    if not latest_version:
                        logger.warning(f"Could not fetch latest version for {package_name}")
                        continue
                    
                    # Compare versions
                    if self._is_version_newer(latest_version, current_version):
                        updates_available[package_name] = {
                            "current_version": current_version,
                            "latest_version": latest_version,
                            "update_available": True
                        }
                        logger.info(f"Update available for {package_name}: {current_version} -> {latest_version}")
                    else:
                        updates_available[package_name] = {
                            "current_version": current_version,
                            "latest_version": latest_version,
                            "update_available": False
                        }
                
                except Exception as e:
                    logger.error(f"Failed to check updates for {package_name}: {e}")
            
            # Save status
            self._save_update_status()
            
            result = {
                "check_time": self.last_check_time.isoformat(),
                "packages": updates_available,
                "total_updates_available": sum(1 for pkg in updates_available.values() if pkg.get("update_available"))
            }
            
            # Log summary
            total_updates = result["total_updates_available"]
            if total_updates > 0:
                logger.info(f"Found {total_updates} package updates available")
                
                # Add to history
                self.update_history.append({
                    "timestamp": self.last_check_time.isoformat(),
                    "type": "check",
                    "updates_found": total_updates,
                    "packages": [pkg for pkg, info in updates_available.items() if info.get("update_available")]
                })
            else:
                logger.info("All packages are up to date")
            
            return result
            
        except Exception as e:
            logger.error(f"Update check failed: {e}")
            return {"error": str(e), "check_time": datetime.utcnow().isoformat()}
    
    def _get_installed_version(self, package_name: str) -> Optional[str]:
        """Get the currently installed version of a package."""
        try:
            return pkg_resources.get_distribution(package_name).version
        except pkg_resources.DistributionNotFound:
            return None
        except Exception as e:
            logger.warning(f"Failed to get version for {package_name}: {e}")
            return None
    
    async def _get_latest_version_from_pypi(self, package_name: str) -> Optional[str]:
        """Get the latest version of a package from PyPI."""
        try:
            async with asyncio.create_task(self._fetch_pypi_info(package_name)) as response:
                if response.get("info"):
                    return response["info"]["version"]
                return None
        except Exception as e:
            logger.error(f"Failed to fetch PyPI info for {package_name}: {e}")
            return None
    
    async def _fetch_pypi_info(self, package_name: str) -> Dict[str, Any]:
        """Fetch package information from PyPI API."""
        try:
            # Use requests in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"PyPI API returned status {response.status_code} for {package_name}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to fetch PyPI info for {package_name}: {e}")
            return {}
    
    def _is_version_newer(self, version1: str, version2: str) -> bool:
        """Compare two version strings."""
        try:
            from packaging import version
            return version.parse(version1) > version.parse(version2)
        except Exception as e:
            logger.warning(f"Failed to compare versions {version1} vs {version2}: {e}")
            # Fallback to string comparison
            return version1 != version2
    
    async def update_package(self, package_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Update a specific package."""
        try:
            logger.info(f"Updating package {package_name}" + (f" to version {version}" if version else " to latest"))
            
            # Prepare update command
            if version:
                cmd = [sys.executable, "-m", "pip", "install", f"{package_name}=={version}"]
            else:
                cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package_name]
            
            # Execute update
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Get new version
                new_version = self._get_installed_version(package_name)
                
                logger.info(f"Successfully updated {package_name} to version {new_version}")
                
                # Record in history
                update_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "update",
                    "package": package_name,
                    "new_version": new_version,
                    "success": True
                }
                self.update_history.append(update_record)
                self._save_update_status()
                
                return {
                    "success": True,
                    "package": package_name,
                    "new_version": new_version,
                    "message": f"Successfully updated {package_name}"
                }
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                logger.error(f"Failed to update {package_name}: {error_msg}")
                
                # Record failure
                update_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "update",
                    "package": package_name,
                    "success": False,
                    "error": error_msg
                }
                self.update_history.append(update_record)
                self._save_update_status()
                
                return {
                    "success": False,
                    "package": package_name,
                    "error": error_msg
                }
                
        except Exception as e:
            logger.error(f"Update failed for {package_name}: {e}")
            return {
                "success": False,
                "package": package_name,
                "error": str(e)
            }
    
    async def update_instagrapi(self) -> Dict[str, Any]:
        """Specifically update Instagrapi with special handling."""
        try:
            logger.info("Checking for Instagrapi updates...")
            
            # Check current version
            current_version = self._get_installed_version("instagrapi")
            if not current_version:
                return {"success": False, "error": "Instagrapi not installed"}
            
            # Check for updates
            latest_version = await self._get_latest_version_from_pypi("instagrapi")
            if not latest_version:
                return {"success": False, "error": "Could not fetch latest version"}
            
            if not self._is_version_newer(latest_version, current_version):
                return {
                    "success": True,
                    "message": f"Instagrapi is already up to date (version {current_version})",
                    "current_version": current_version,
                    "latest_version": latest_version
                }
            
            # Backup current session files before update
            await self._backup_session_files()
            
            # Perform update
            update_result = await self.update_package("instagrapi")
            
            if update_result.get("success"):
                logger.info(f"Instagrapi updated successfully from {current_version} to {latest_version}")
                
                # Check if Instagram tools need reinitialization
                await self._notify_instagrapi_update()
                
                return {
                    "success": True,
                    "message": f"Instagrapi updated from {current_version} to {latest_version}",
                    "previous_version": current_version,
                    "new_version": latest_version,
                    "requires_restart": True
                }
            else:
                return update_result
                
        except Exception as e:
            logger.error(f"Instagrapi update failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _backup_session_files(self):
        """Backup Instagram session files before update."""
        try:
            session_files = list(Path(".").glob("*session*.json"))
            if session_files:
                backup_dir = Path("session_backups")
                backup_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                for session_file in session_files:
                    backup_path = backup_dir / f"{session_file.stem}_{timestamp}.json"
                    session_file.rename(backup_path)
                    logger.info(f"Backed up session file: {session_file} -> {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to backup session files: {e}")
    
    async def _notify_instagrapi_update(self):
        """Notify other components about Instagrapi update."""
        try:
            # This could trigger reinitialization of Instagram tools
            logger.info("Instagrapi updated - Instagram tools may need reinitialization")
            
            # Could send a webhook or internal notification here
            # For now, just log the event
            
        except Exception as e:
            logger.warning(f"Failed to notify about Instagrapi update: {e}")
    
    def start_background_scheduler(self):
        """Start the background update checker."""
        if self.running or not self.auto_update_enabled:
            return
        
        self.running = True
        
        # Schedule periodic checks
        schedule.every(self.update_check_interval_hours).hours.do(self._scheduled_check)
        
        # Start scheduler thread
        self.scheduler_thread = Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(f"Background update checker started (interval: {self.update_check_interval_hours} hours)")
    
    def stop_background_scheduler(self):
        """Stop the background update checker."""
        self.running = False
        schedule.clear()
        logger.info("Background update checker stopped")
    
    def _run_scheduler(self):
        """Run the scheduler in background thread."""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
    
    def _scheduled_check(self):
        """Perform scheduled update check."""
        try:
            # Run async check in the background
            asyncio.create_task(self.check_for_updates())
        except Exception as e:
            logger.error(f"Scheduled check failed: {e}")
    
    def get_update_status(self) -> Dict[str, Any]:
        """Get current update status and history."""
        return {
            "auto_update_enabled": self.auto_update_enabled,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "update_check_interval_hours": self.update_check_interval_hours,
            "critical_packages": self.critical_packages,
            "update_history": self.update_history[-10:],  # Last 10 entries
            "scheduler_running": self.running
        }
    
    def force_check_now(self) -> None:
        """Force an immediate update check."""
        if self.running:
            asyncio.create_task(self.check_for_updates())
        else:
            logger.warning("Background scheduler not running, cannot force check")


# Global update manager instance
_update_manager: Optional[UpdateManager] = None


def get_update_manager() -> UpdateManager:
    """Get global update manager instance."""
    global _update_manager
    if _update_manager is None:
        _update_manager = UpdateManager()
    return _update_manager


def start_update_manager():
    """Start the global update manager."""
    manager = get_update_manager()
    manager.start_background_scheduler()
    return manager


def stop_update_manager():
    """Stop the global update manager."""
    global _update_manager
    if _update_manager:
        _update_manager.stop_background_scheduler()