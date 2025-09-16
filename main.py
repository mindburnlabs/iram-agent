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

import getpass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent_orchestrator import create_instagram_agent
from src.app import run_dev_server
from src.utils import get_logger
from src.config import get_config, IRamConfig

logger = get_logger(__name__)
console = Console()


async def run_agent_task(task: str, username: str = None, output: str = None, public_only: bool = False, rich_output: bool = False):
    """Run a specific agent task."""
    try:
        console.print(f"[bold cyan]Initializing agent for task:[/] {task}")
        
        # Create agent with public_only flag
        config_overrides = {"instagram": {"public_fallback": public_only}}
        agent = create_instagram_agent(config=config_overrides)
        
        # Prompt for credentials if not set
        if not public_only and not agent.iram_config.has_instagram_auth():
            prompt_for_credentials()
            agent = create_instagram_agent(config=config_overrides) # Re-create with new creds
        
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
            console.print(f"[bold green]✅ Results saved to {output}[/]")
        elif rich_output:
            display_rich_results(result)
        else:
            # Simple text output
            console.print(Panel.fit(
                f"[bold green]✅ Task completed successfully[/]" if result.get("success") else f"[bold red]❌ Task failed: {result.get('error', 'Unknown error')}[/]",
                title="IRAM Analysis Results"
            ))
            if result.get("success") and "result" in result:
                console.print(Markdown(result['result']))

    except Exception as e:
        console.print(f"[bold red]❌ Task execution failed: {e}[/]")
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
    run_parser = subparsers.add_parser("run", help=\"Run a specific task\")
    run_parser.add_argument("--task", required=True, help=\"Task to execute\")
    run_parser.add_argument("--username", help=\"Instagram username (if applicable)\")
    run_parser.add_argument("--output", help=\"Output file path for results (JSON)\")
    run_parser.add_argument("--public-only", action="store_true", help=\"Run in public-only mode (no login required)\")
    run_parser.add_argument("--rich-output", action="store_true", help=\"Enable rich, formatted output\")

    # New command for interactive configuration
    config_parser = subparsers.add_parser("configure", help=\"Interactively configure IRAM\")
    
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
            run_dev_server()
            
        elif args.command == "run":
            asyncio.run(run_agent_task(args.task, args.username, args.output))
            
        elif args.command == "config":
            show_config()
            
        elif args.command == "test":
            run_tests()
            
        elif args.command == "configure":
            interactive_configure()
            
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled by user.[/]")
    except Exception as e:
        console.print(f"[bold red]❌ Command failed: {e}[/]")
        sys.exit(1)


def display_rich_results(result: dict):
    """Display analysis results in a rich format."""
    if not result.get("success"):
        console.print(Panel.fit(f"[bold red]❌ Task failed: {result.get('error', 'Unknown error')}[/]"))
        return

    console.print(Panel.fit("[bold green]✅ IRAM Analysis Results[/]", title="Task Complete"))
    
    if "result" in result and isinstance(result["result"], str):
        console.print(Markdown(result["result"]))
    elif isinstance(result.get("result"), dict):
        # For structured results, pretty-print the JSON
        syntax = Syntax(json.dumps(result["result"], indent=2), "json", theme="monokai", line_numbers=True)
        console.print(syntax)


def prompt_for_credentials():
    """Prompt user for Instagram credentials and save to .env."""
    console.print("[bold yellow]Instagram credentials not found.[/]")
    username = st.text_input("Enter your Instagram username:")
    password = st.text_input("Enter your Instagram password:", type="password")
    
    if username and password:
        with open(".env", "a") as f:
            f.write(f"\nINSTAGRAM_USERNAME={username}")
            f.write(f"\nINSTAGRAM_PASSWORD={password}")
        console.print("[bold green]✅ Credentials saved to .env file.[/]")


def interactive_configure():
    """Interactively configure IRAM and save to .env."""
    console.print(Panel.fit("[bold cyan]IRAM Interactive Configuration[/]"))

    # Instagram credentials
    if console.input("[bold]Configure Instagram credentials? (y/n):[/]").lower() == 'y':
        username = console.input("Instagram Username: ")
        password = getpass.getpass("Instagram Password: ")
        with open(".env", "a") as f:
            f.write(f"\nINSTAGRAM_USERNAME={username}")
            f.write(f"\nINSTAGRAM_PASSWORD={password}")

    # LLM Provider
    console.print("\n[bold]LLM Provider Configuration[/]")
    provider = console.input("Choose LLM provider (anthropic, openai, openrouter) [anthropic]: ").lower() or "anthropic"
    api_key = getpass.getpass(f"Enter {provider.upper()} API Key: ")
    with open(".env", "a") as f:
        if provider == "anthropic":
            f.write(f"\nANTHROPIC_API_KEY={api_key}")
        elif provider == "openai":
            f.write(f"\nOPENAI_API_KEY={api_key}")
        elif provider == "openrouter":
            f.write(f"\nOPENROUTER_API_KEY={api_key}")

    console.print("[bold green]✅ Configuration saved to .env file.[/]")


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
    
    # Test 5: Security check with safety
    tests_total += 1
    try:
        import subprocess
        result = subprocess.run(["safety", "check", "--full-report"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Security check (safety) passed")
            tests_passed += 1
        else:
            print(f"❌ Security check (safety) failed:\n{result.stdout}")
    except FileNotFoundError:
        print("⚠️  `safety` command not found, skipping security check. Install with `pip install safety`.")
        tests_total -= 1 # Don't count this test if safety is not installed
    except Exception as e:
        print(f"❌ Security check failed: {e}")
    
    if tests_passed == tests_total:
        print("🎉 All tests passed! IRAM is ready to use.")
    else:
        print("⚠️  Some tests failed. Check configuration and dependencies.")
    
    print("="*50)


if __name__ == "__main__":
    main()