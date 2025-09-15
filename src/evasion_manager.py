"""
Instagram Research Agent MCP (IRAM) - Evasion Manager

This module implements ML-based evasion strategies to avoid detection and bans.
"""

import random
import time
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

from .utils import get_logger

logger = get_logger(__name__)


class EvasionManager:
    """Manages evasion strategies to avoid Instagram detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the evasion manager."""
        self.config = config or {}
        
        # Request tracking
        self.request_history = []
        self.last_request_time = 0
        self.consecutive_errors = 0
        
        # ML model for risk prediction
        self.risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.model_trained = False
        
        # Evasion parameters
        self.base_delay = 2.0  # Base delay in seconds
        self.max_delay = 30.0  # Maximum delay
        self.error_cooldown = 60  # Cooldown after errors
        
        logger.info("Evasion manager initialized")
    
    async def apply_delay(self):
        """Apply intelligent delay based on current risk level."""
        try:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            # Calculate risk-based delay
            risk_level = self.calculate_risk()
            delay = self.calculate_delay(risk_level)
            
            # Ensure minimum time between requests
            if time_since_last < delay:
                sleep_time = delay - time_since_last
                logger.debug(f"Applying evasion delay: {sleep_time:.2f}s (risk: {risk_level:.2f})")
                await asyncio.sleep(sleep_time)
            
            self.last_request_time = time.time()
            
        except Exception as e:
            logger.error(f"Delay application failed: {e}")
            await asyncio.sleep(self.base_delay)
    
    def calculate_risk(self) -> float:
        """Calculate current risk level based on request patterns."""
        try:
            if not self.request_history:
                return 0.1  # Low risk for first request
            
            # Simple risk calculation based on recent activity
            recent_requests = len([r for r in self.request_history if time.time() - r['timestamp'] < 300])  # 5 minutes
            error_rate = self.consecutive_errors / max(1, len(self.request_history))
            
            # Risk factors
            frequency_risk = min(recent_requests / 50, 1.0)  # High if > 50 requests in 5min
            error_risk = min(error_rate * 2, 1.0)
            
            total_risk = (frequency_risk + error_risk) / 2
            return min(total_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Risk calculation failed: {e}")
            return 0.5  # Medium risk as default
    
    def calculate_delay(self, risk_level: float) -> float:
        """Calculate delay based on risk level."""
        try:
            # Base delay with risk multiplier
            base = self.base_delay
            risk_multiplier = 1 + (risk_level * 10)  # 1x to 11x multiplier
            
            # Add randomization to avoid patterns
            randomization = random.uniform(0.8, 1.2)
            
            delay = base * risk_multiplier * randomization
            return min(delay, self.max_delay)
            
        except Exception as e:
            logger.error(f"Delay calculation failed: {e}")
            return self.base_delay
    
    def record_request(self, success: bool, response_time: float = 0, status_code: int = 200):
        """Record a request for pattern analysis."""
        try:
            request_data = {
                'timestamp': time.time(),
                'success': success,
                'response_time': response_time,
                'status_code': status_code
            }
            
            self.request_history.append(request_data)
            
            # Keep only recent history (last 1000 requests)
            if len(self.request_history) > 1000:
                self.request_history = self.request_history[-1000:]
            
            # Update error tracking
            if success:
                self.consecutive_errors = 0
            else:
                self.consecutive_errors += 1
            
            # Train model periodically
            if len(self.request_history) % 100 == 0:
                self._update_risk_model()
                
        except Exception as e:
            logger.error(f"Request recording failed: {e}")
    
    def _update_risk_model(self):
        """Update the ML risk prediction model."""
        try:
            if len(self.request_history) < 50:
                return
            
            # Prepare training data
            features = []
            targets = []
            
            for i in range(1, len(self.request_history)):
                prev_req = self.request_history[i-1]
                curr_req = self.request_history[i]
                
                # Extract features
                time_diff = curr_req['timestamp'] - prev_req['timestamp']
                feature_vector = [
                    time_diff,
                    prev_req['response_time'],
                    1 if prev_req['success'] else 0,
                    prev_req['status_code']
                ]
                
                # Target: 1 if current request failed, 0 if succeeded
                target = 0 if curr_req['success'] else 1
                
                features.append(feature_vector)
                targets.append(target)
            
            if len(features) < 20:
                return
            
            # Train model
            X = np.array(features)
            y = np.array(targets)
            
            X_scaled = self.scaler.fit_transform(X)
            self.risk_model.fit(X_scaled, y)
            self.model_trained = True
            
            logger.debug("Risk prediction model updated")
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
    
    def predict_risk(self, time_since_last: float, last_response_time: float) -> float:
        """Predict risk for next request using ML model."""
        try:
            if not self.model_trained or not self.request_history:
                return self.calculate_risk()
            
            last_req = self.request_history[-1]
            features = np.array([[
                time_since_last,
                last_response_time,
                1 if last_req['success'] else 0,
                last_req['status_code']
            ]])
            
            features_scaled = self.scaler.transform(features)
            risk_prob = self.risk_model.predict_proba(features_scaled)[0][1]  # Probability of failure
            
            return min(risk_prob, 1.0)
            
        except Exception as e:
            logger.error(f"Risk prediction failed: {e}")
            return self.calculate_risk()
    
    def should_pause(self) -> bool:
        """Determine if we should pause operations due to high risk."""
        try:
            risk = self.calculate_risk()
            
            # Pause if risk is very high or too many consecutive errors
            if risk > 0.8 or self.consecutive_errors > 5:
                logger.warning(f"Pausing operations due to high risk: {risk:.2f}, errors: {self.consecutive_errors}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Pause decision failed: {e}")
            return True  # Err on the side of caution
    
    async def handle_error(self, error_type: str = "unknown"):
        """Handle errors with appropriate cooldown."""
        try:
            self.consecutive_errors += 1
            
            # Record the error
            self.record_request(success=False)
            
            # Apply error-specific cooldown
            cooldown_time = self.error_cooldown * min(self.consecutive_errors, 5)
            
            logger.warning(f"Error detected ({error_type}), applying cooldown: {cooldown_time}s")
            await asyncio.sleep(cooldown_time)
            
        except Exception as e:
            logger.error(f"Error handling failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evasion manager statistics."""
        try:
            if not self.request_history:
                return {"total_requests": 0}
            
            total_requests = len(self.request_history)
            successful_requests = sum(1 for r in self.request_history if r['success'])
            success_rate = successful_requests / total_requests if total_requests > 0 else 0
            
            recent_requests = [r for r in self.request_history if time.time() - r['timestamp'] < 3600]  # Last hour
            recent_success_rate = sum(1 for r in recent_requests if r['success']) / max(1, len(recent_requests))
            
            avg_response_time = np.mean([r['response_time'] for r in self.request_history if r['response_time'] > 0])
            
            return {
                "total_requests": total_requests,
                "success_rate": success_rate,
                "recent_success_rate": recent_success_rate,
                "consecutive_errors": self.consecutive_errors,
                "current_risk": self.calculate_risk(),
                "avg_response_time": avg_response_time,
                "model_trained": self.model_trained
            }
            
        except Exception as e:
            logger.error(f"Stats generation failed: {e}")
            return {"error": str(e)}
    
    def reset_counters(self):
        """Reset error counters and request history."""
        try:
            self.request_history = []
            self.consecutive_errors = 0
            self.last_request_time = 0
            logger.info("Evasion counters reset")
            
        except Exception as e:
            logger.error(f"Counter reset failed: {e}")