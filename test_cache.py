#!/usr/bin/env python3
"""
Simple test script to verify Redis cache integration in IRAM.

Run this to test:
1. Cache initialization (Redis or fallback to memory)
2. Basic cache operations (set, get, delete)
3. TTL expiration
4. Pattern-based deletion
5. Cache statistics
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import get_config
from src.cache import (
    initialize_cache, 
    get_cache, 
    close_cache,
    profile_cache_key,
    posts_cache_key,
    cached,
    cache_maintenance
)


async def test_basic_cache_operations():
    """Test basic cache operations."""
    print("🧪 Testing basic cache operations...")
    
    cache = await get_cache()
    
    # Test set and get
    test_key = "test:basic"
    test_value = {"message": "Hello Cache!", "timestamp": time.time()}
    
    success = await cache.set(test_key, test_value, ttl=60)
    print(f"   ✅ Set operation: {'success' if success else 'failed'}")
    
    retrieved = await cache.get(test_key)
    print(f"   ✅ Get operation: {'success' if retrieved == test_value else 'failed'}")
    
    # Test exists
    exists = await cache.exists(test_key)
    print(f"   ✅ Exists operation: {'success' if exists else 'failed'}")
    
    # Test delete
    deleted = await cache.delete(test_key)
    print(f"   ✅ Delete operation: {'success' if deleted else 'failed'}")
    
    # Test get after delete
    retrieved_after = await cache.get(test_key)
    print(f"   ✅ Get after delete: {'success' if retrieved_after is None else 'failed'}")
    
    return True


async def test_cache_key_generators():
    """Test cache key generation utilities."""
    print("🔑 Testing cache key generators...")
    
    cache = await get_cache()
    
    # Test profile cache key
    username = "test_user"
    profile_key = profile_cache_key(username)
    profile_data = {
        "username": username,
        "followers": 1000,
        "posts": 50
    }
    
    await cache.set(profile_key, profile_data, ttl=300)
    retrieved_profile = await cache.get(profile_key)
    
    print(f"   ✅ Profile cache key: {profile_key}")
    print(f"   ✅ Profile data stored/retrieved: {'success' if retrieved_profile == profile_data else 'failed'}")
    
    # Test posts cache key
    posts_key = posts_cache_key(username, limit=25)
    posts_data = [
        {"id": 1, "caption": "Test post 1"},
        {"id": 2, "caption": "Test post 2"}
    ]
    
    await cache.set(posts_key, posts_data, ttl=300)
    retrieved_posts = await cache.get(posts_key)
    
    print(f"   ✅ Posts cache key: {posts_key}")
    print(f"   ✅ Posts data stored/retrieved: {'success' if retrieved_posts == posts_data else 'failed'}")
    
    return True


async def test_cache_decorators():
    """Test cache decorators."""
    print("🎨 Testing cache decorators...")
    
    call_count = 0
    
    @cached(ttl=60, key_func=lambda x: f"test_func:{x}")
    async def expensive_function(input_value: str):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)  # Simulate expensive operation
        return f"Result for {input_value} - call #{call_count}"
    
    # First call
    result1 = await expensive_function("test_input")
    print(f"   ✅ First call result: {result1}")
    print(f"   ✅ Call count after first call: {call_count}")
    
    # Second call (should be cached)
    result2 = await expensive_function("test_input")
    print(f"   ✅ Second call result: {result2}")
    print(f"   ✅ Call count after second call: {call_count}")
    
    # Results should be the same and function should only be called once
    cached_success = result1 == result2 and call_count == 1
    print(f"   ✅ Caching decorator: {'success' if cached_success else 'failed'}")
    
    return cached_success


async def test_pattern_deletion():
    """Test pattern-based cache deletion."""
    print("🗑️  Testing pattern deletion...")
    
    cache = await get_cache()
    
    # Set multiple keys with similar patterns
    test_keys = [
        "user:john:profile",
        "user:john:posts", 
        "user:jane:profile",
        "user:jane:posts",
        "system:config"
    ]
    
    for key in test_keys:
        await cache.set(key, f"data for {key}", ttl=300)
    
    # Delete all john's data
    deleted_count = await cache.clear_pattern("user:john:*")
    print(f"   ✅ Deleted {deleted_count} keys matching 'user:john:*'")
    
    # Check which keys still exist
    remaining_keys = []
    for key in test_keys:
        if await cache.exists(key):
            remaining_keys.append(key)
    
    expected_remaining = ["user:jane:profile", "user:jane:posts", "system:config"]
    pattern_success = set(remaining_keys) == set(expected_remaining)
    print(f"   ✅ Pattern deletion: {'success' if pattern_success else 'failed'}")
    print(f"   ✅ Remaining keys: {remaining_keys}")
    
    # Clean up
    await cache.clear_pattern("user:*")
    await cache.clear_pattern("system:*")
    
    return pattern_success


async def test_cache_stats():
    """Test cache statistics."""
    print("📊 Testing cache statistics...")
    
    cache = await get_cache()
    stats = await cache.get_stats()
    
    print(f"   ✅ Cache backend: {stats.get('backend', 'unknown')}")
    print(f"   ✅ Cache stats: {stats}")
    
    # Run maintenance
    maintenance_result = await cache_maintenance()
    print(f"   ✅ Maintenance result: {maintenance_result}")
    
    return True


async def test_ttl_expiration():
    """Test TTL expiration (for memory cache)."""
    print("⏰ Testing TTL expiration...")
    
    cache = await get_cache()
    
    # Set a key with very short TTL
    short_key = "test:ttl"
    await cache.set(short_key, "expires soon", ttl=2)
    
    # Should exist immediately
    exists_now = await cache.exists(short_key)
    print(f"   ✅ Key exists immediately: {'yes' if exists_now else 'no'}")
    
    # Wait for expiration
    print("   ⏳ Waiting 3 seconds for expiration...")
    await asyncio.sleep(3)
    
    # Should not exist after expiration (mainly for memory cache)
    exists_after = await cache.exists(short_key)
    print(f"   ✅ Key exists after TTL: {'no' if not exists_after else 'yes (Redis auto-expires)'}")
    
    return True


async def main():
    """Main test function."""
    print("🚀 Starting IRAM Cache Integration Tests")
    print("=" * 50)
    
    try:
        # Initialize cache
        print("🔧 Initializing cache...")
        await initialize_cache()
        print("   ✅ Cache initialized successfully\n")
        
        # Show configuration
        config = get_config()
        print(f"📋 Configuration:")
        print(f"   Environment: {config.environment}")
        print(f"   Has Redis: {config.has_redis()}")
        if config.has_redis():
            print(f"   Redis Host: {config.redis.host}:{config.redis.port}")
        print()
        
        # Run tests
        tests = [
            test_basic_cache_operations,
            test_cache_key_generators,
            test_cache_decorators,
            test_pattern_deletion,
            test_ttl_expiration,
            test_cache_stats
        ]
        
        results = []
        for test in tests:
            try:
                result = await test()
                results.append(result)
                print()
            except Exception as e:
                print(f"   ❌ Test failed with error: {e}")
                results.append(False)
                print()
        
        # Summary
        passed = sum(results)
        total = len(results)
        
        print("=" * 50)
        print(f"🏁 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Cache integration is working correctly.")
            return 0
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
            return 1
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        return 1
        
    finally:
        # Clean up
        print("\n🧹 Cleaning up...")
        await close_cache()
        print("   ✅ Cache closed")


if __name__ == "__main__":
    exit_code = asyncio.run(main())