#!/usr/bin/env python3
"""
Instagram Research Agent MCP (IRAM) - Main Entry Point

This is the main entry point for the IRAM application, providing both CLI and server modes.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent_orchestrator import create_instagram_agent
from src.mcp_server import main as start_server
from src.utils import get_logger

logger = get_logger(__name__)


async def run_agent_task(task: str, username: str = None, output: str = None):
    """Run a specific agent task."""
    try:
        logger.info(f"Initializing agent for task: {task}")
        
        # Create agent
        agent = create_instagram_agent()
        
        # Execute based on task type
        if task == "analyze-account" and username:
            result = agent.analyze_account_trends(username)
        elif task == "profile" and username:
            result = await agent.execute_task(f"Fetch and analyze profile for @{username}")
        else:
            result = await agent.execute_task(task)
        
        # Handle output
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Results saved to {output}")
        else:
            print("\n" + "="*60)
            print("IRAM ANALYSIS RESULTS")
            print("="*60)
            
            if result.get("success"):
                print(f"✅ Task completed successfully")
                if "result" in result:
                    print(f"\nResult: {result['result']}")
            else:
                print(f"❌ Task failed: {result.get('error', 'Unknown error')}")
            
            print("="*60)
        
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Instagram Research Agent MCP (IRAM) - Autonomous Instagram Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start MCP server
  python main.py server
  
  # Analyze an account
  python main.py run --task "analyze-account" --username "example_user"
  
  # Fetch profile information
  python main.py run --task "profile" --username "example_user" --output results.json
  
  # Custom analysis task
  python main.py run --task "Compare engagement between @user1 and @user2"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the MCP server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    server_parser.add_argument("--port", type=int, default=8000, help="Server port")
    server_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a specific task")
    run_parser.add_argument("--task", required=True, help="Task to execute")
    run_parser.add_argument("--username", help="Instagram username (if applicable)")
    run_parser.add_argument("--output", help="Output file path for results")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run basic tests")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "server":
            logger.info("Starting IRAM MCP Server...")
            
            # Set environment variables if provided
            if hasattr(args, 'host') and args.host:
                os.environ["HOST"] = args.host
            # Railway provides PORT environment variable, don't override it unless explicitly set
            if hasattr(args, 'port') and args.port and args.port != 8000:
                os.environ["PORT"] = str(args.port)
            elif not os.getenv("PORT"):
                os.environ["PORT"] = "8000"
            if hasattr(args, 'debug') and args.debug:
                os.environ["DEBUG"] = "true"
            
            logger.info(f"Server will start on {os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT')}")
            start_server()
            
        elif args.command == "run":
            asyncio.run(run_agent_task(args.task, args.username, args.output))
            
        elif args.command == "config":
            show_config()
            
        elif args.command == "test":
            run_tests()
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


def show_config():
    """Show current configuration."""
    print("\n" + "="*50)
    print("IRAM CONFIGURATION")
    print("="*50)
    
    config_items = [
        ("Instagram Username", os.getenv("INSTAGRAM_USERNAME", "Not set")),
        ("Instagram Password", "Set" if os.getenv("INSTAGRAM_PASSWORD") else "Not set"),
        ("OpenRouter API Key", "Set" if os.getenv("OPENROUTER_API_KEY") else "Not set"),
        ("OpenAI API Key", "Set" if os.getenv("OPENAI_API_KEY") else "Not set"),
        ("LLM Model", os.getenv("LLM_MODEL", "openrouter/sonoma-sky-alpha")),
        ("Host", os.getenv("HOST", "0.0.0.0")),
        ("Port", os.getenv("PORT", "8000")),
        ("Debug", os.getenv("DEBUG", "false")),
        ("Log Level", os.getenv("LOG_LEVEL", "INFO")),
    ]
    
    for key, value in config_items:
        print(f"{key:20}: {value}")
    
    print("\nEnvironment file: .env")
    env_file = Path(".env")
    if env_file.exists():
        print("✅ Environment file exists")
    else:
        print("❌ Environment file not found (create .env from .env.example)")
    
    print("="*50)


def run_tests():
    """Run basic functionality tests."""
    print("\n" + "="*50)
    print("IRAM BASIC TESTS")
    print("="*50)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Import modules
    tests_total += 1
    try:
        from src.agent_orchestrator import create_instagram_agent
        from src.scraping_module import InstagramScraper
        from src.analysis_module import ContentAnalyzer
        print("✅ Module imports successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Module import failed: {e}")
    
    # Test 2: Create agent
    tests_total += 1
    try:
        agent = create_instagram_agent()
        print("✅ Agent creation successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
    
    # Test 3: Username validation
    tests_total += 1
    try:
        from src.utils import validate_instagram_username
        assert validate_instagram_username("test_user") == True
        assert validate_instagram_username("invalid..user") == False
        print("✅ Username validation working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Username validation failed: {e}")
    
    # Test 4: Environment variables
    tests_total += 1
    required_env_vars = ["INSTAGRAM_USERNAME", "INSTAGRAM_PASSWORD"]
    ai_key_set = bool(os.getenv("OPENROUTER_API_KEY")) or bool(os.getenv("OPENAI_API_KEY"))
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if not missing_vars and ai_key_set:
        if os.getenv("OPENROUTER_API_KEY"):
            print("✅ All required environment variables set (using OpenRouter)")
        else:
            print("✅ All required environment variables set (using OpenAI)")
        tests_passed += 1
    else:
        missing_items = missing_vars.copy()
        if not ai_key_set:
            missing_items.append("OPENROUTER_API_KEY or OPENAI_API_KEY")
        print(f"❌ Missing environment variables: {', '.join(missing_items)}")
    
    print(f"\nTests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("🎉 All tests passed! IRAM is ready to use.")
    else:
        print("⚠️  Some tests failed. Check configuration and dependencies.")
    
    print("="*50)


if __name__ == "__main__":
    main()