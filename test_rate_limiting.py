#!/usr/bin/env python3
"""
Rate Limiting Integration Test

Test script to verify the complete rate limiting system works correctly
including the Redis backend, middleware, and management API routes.
"""

import asyncio
import httpx
import time
import json
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_USER_ID = "test_user_123"
TEST_IP = "192.168.1.100"
TEST_API_KEY = "test_api_key_456"


async def test_rate_limit_middleware():
    """Test rate limiting middleware functionality."""
    print("\n🧪 Testing Rate Limiting Middleware")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        # Test API global rate limit
        print("Testing API global rate limit...")
        responses = []
        
        for i in range(12):  # Should exceed limit of 10/minute
            try:
                response = await client.get(f"{BASE_URL}/health")
                responses.append({
                    "request": i + 1,
                    "status": response.status_code,
                    "headers": {
                        "X-RateLimit-Limit": response.headers.get("X-RateLimit-Limit"),
                        "X-RateLimit-Remaining": response.headers.get("X-RateLimit-Remaining"),
                        "X-RateLimit-Reset": response.headers.get("X-RateLimit-Reset"),
                        "Retry-After": response.headers.get("Retry-After")
                    }
                })
                
                if response.status_code == 429:
                    print(f"  ✅ Request {i + 1}: Rate limited (429) - {response.headers.get('X-RateLimit-Remaining', '0')} remaining")
                    break
                else:
                    print(f"  ✅ Request {i + 1}: Allowed ({response.status_code}) - {response.headers.get('X-RateLimit-Remaining', '?')} remaining")
                
            except Exception as e:
                print(f"  ❌ Request {i + 1}: Error - {e}")
            
            await asyncio.sleep(0.1)  # Small delay
    
    return responses


async def test_rate_limit_api():
    """Test rate limiting management API endpoints."""
    print("\n🛠️ Testing Rate Limiting Management API")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        # Note: These tests assume you have proper authentication setup
        # For basic testing, you might need to modify auth requirements
        
        try:
            # Test getting rate limit configs
            print("Testing rate limit configurations...")
            response = await client.get(f"{BASE_URL}/rate-limit/configs")
            if response.status_code == 200:
                configs = response.json()
                print(f"  ✅ Retrieved {configs.get('total_configs', 0)} configurations")
                for name, config in configs.get('configs', {}).items():
                    print(f"    - {name}: {config['limit']}/{config['window']}s ({config['algorithm']})")
            else:
                print(f"  ❌ Failed to get configs: {response.status_code}")
            
            # Test rate limit status check
            print("\nTesting rate limit status...")
            response = await client.get(
                f"{BASE_URL}/rate-limit/status/api_per_ip",
                params={"identifier": TEST_IP}
            )
            if response.status_code == 200:
                status = response.json()
                print(f"  ✅ Status: {status['limit']} limit, {status['remaining']} remaining")
            else:
                print(f"  ❌ Failed to get status: {response.status_code}")
            
            # Test rate limit health check
            print("\nTesting rate limiter health...")
            response = await client.get(f"{BASE_URL}/rate-limit/health")
            if response.status_code == 200:
                health = response.json()
                print(f"  ✅ Health: {health['status']}")
                print(f"    - Redis available: {health.get('redis_available', False)}")
                print(f"    - Configs loaded: {health.get('configs_loaded', 0)}")
            else:
                print(f"  ❌ Health check failed: {response.status_code}")
            
        except httpx.RequestError as e:
            print(f"  ❌ Request error: {e}")
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")


async def test_custom_rate_limit():
    """Test creating and using custom rate limit configurations."""
    print("\n⚙️ Testing Custom Rate Limit Configuration")
    print("=" * 50)
    
    # This would require proper authentication and admin permissions
    print("Custom rate limit tests require authentication - skipping for now")
    print("To test manually:")
    print("1. Authenticate as admin user")
    print("2. POST /rate-limit/configs with custom configuration")
    print("3. Test the new rate limit")
    print("4. DELETE /rate-limit/configs/{name} to clean up")


async def simulate_concurrent_requests():
    """Simulate concurrent requests to test rate limiting under load."""
    print("\n🚀 Testing Concurrent Request Rate Limiting")
    print("=" * 50)
    
    async def make_request(client: httpx.AsyncClient, request_id: int):
        """Make a single request and return result."""
        try:
            start_time = time.time()
            response = await client.get(f"{BASE_URL}/health")
            end_time = time.time()
            
            return {
                "id": request_id,
                "status": response.status_code,
                "duration": round(end_time - start_time, 3),
                "remaining": response.headers.get("X-RateLimit-Remaining"),
                "retry_after": response.headers.get("Retry-After")
            }
        except Exception as e:
            return {
                "id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Launch 20 concurrent requests
        tasks = [make_request(client, i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        allowed = sum(1 for r in results if r.get("status") == 200)
        rate_limited = sum(1 for r in results if r.get("status") == 429)
        errors = sum(1 for r in results if r.get("status") == "error")
        
        print(f"Concurrent request results:")
        print(f"  ✅ Allowed: {allowed}")
        print(f"  ⏸️ Rate limited: {rate_limited}")
        print(f"  ❌ Errors: {errors}")
        
        # Show some individual results
        print("\nSample results:")
        for result in results[:5]:
            if result.get("status") == 200:
                print(f"  Request {result['id']}: {result['status']} ({result['duration']}s) - {result.get('remaining', '?')} remaining")
            elif result.get("status") == 429:
                print(f"  Request {result['id']}: {result['status']} - Retry after {result.get('retry_after', '?')}s")
            else:
                print(f"  Request {result['id']}: {result.get('status', 'unknown')}")


async def main():
    """Run all rate limiting tests."""
    print("🔄 IRAM Rate Limiting System Integration Test")
    print("=" * 60)
    print("Testing Redis-backed rate limiting with FastAPI integration")
    print()
    
    try:
        # Test basic middleware functionality
        await test_rate_limit_middleware()
        
        # Test management API
        await test_rate_limit_api()
        
        # Test custom configurations
        await test_custom_rate_limit()
        
        # Test concurrent requests
        await simulate_concurrent_requests()
        
        print("\n✅ Rate limiting integration test completed!")
        print("\nNext steps:")
        print("1. Set up proper authentication for management endpoints")
        print("2. Configure Redis connection in production")
        print("3. Adjust rate limit configurations per your requirements")
        print("4. Monitor rate limiting effectiveness in production")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)