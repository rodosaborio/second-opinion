#!/usr/bin/env python3
"""
Quick setup validation script for Second Opinion.
Run this after setting up your .env file to validate configuration.
"""

from pathlib import Path


def check_env_file():
    """Check if .env file exists and has required keys."""
    env_path = Path(".env")

    if not env_path.exists():
        print("❌ .env file not found")
        print("   Run: cp .env.example .env")
        return False

    print("✅ .env file found")

    # Check for required API key
    with open(env_path) as f:
        content = f.read()

    if "OPENROUTER_API_KEY=your_openrouter_api_key_here" in content:
        print("⚠️  OpenRouter API key not set")
        print(
            "   Edit .env and replace 'your_openrouter_api_key_here' with your actual key"
        )
        return False

    if "OPENROUTER_API_KEY=" not in content or "sk-or-" not in content:
        print("⚠️  OpenRouter API key may not be properly set")
        print("   Make sure your .env contains: OPENROUTER_API_KEY=sk-or-...")
        return False

    print("✅ OpenRouter API key appears to be set")
    return True


def check_dependencies():
    """Check if dependencies are installed."""
    try:
        from src.second_opinion.config.settings import get_settings

        print("✅ Dependencies installed correctly")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Run: uv sync --dev")
        return False


def check_configuration():
    """Test configuration loading."""
    try:
        from src.second_opinion.config.settings import get_settings

        settings = get_settings()

        print("✅ Configuration loaded successfully")
        print(f"   Environment: {settings.environment}")
        print(
            f"   Cost limits: ${settings.cost_management.default_per_request_limit} per request"
        )

        if settings.has_api_key("openrouter"):
            print("✅ OpenRouter API key configured")
        else:
            print("⚠️  OpenRouter API key not found in configuration")
            return False

        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_api_connection():
    """Test API connection with a minimal request."""
    print("\nTesting API connection...")
    try:
        import subprocess

        result = subprocess.run(
            [
                "uv",
                "run",
                "second-opinion",
                "--primary-model",
                "anthropic/claude-3-5-sonnet",
                "Hello",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("✅ API connection successful")
            return True
        else:
            print("❌ API connection failed:")
            print(f"   {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ API request timed out")
        return False
    except Exception as e:
        print(f"❌ API test error: {e}")
        return False


def generate_encryption_keys():
    """Generate encryption keys for the user."""
    try:
        from cryptography.fernet import Fernet

        print("\n🔐 Generated encryption keys (add these to your .env):")
        print(f"DATABASE_ENCRYPTION_KEY={Fernet.generate_key().decode()}")
        print(f"SESSION_ENCRYPTION_KEY={Fernet.generate_key().decode()}")
        print("\nMake sure to use different keys for database and session encryption!")

    except ImportError:
        print("⚠️  cryptography not installed, cannot generate keys")
        print("   Install with: uv add cryptography")


def main():
    """Run all setup validation checks."""
    print("🔧 Second Opinion Setup Validation")
    print("=" * 40)

    checks = [
        ("Environment file", check_env_file),
        ("Dependencies", check_dependencies),
        ("Configuration", check_configuration),
    ]

    all_passed = True
    for name, check_func in checks:
        print(f"\n{name}:")
        if not check_func():
            all_passed = False

    if all_passed:
        print("\n🎉 Setup validation passed!")
        print("\nReady to use Second Opinion:")
        print(
            '   uv run second-opinion --primary-model "anthropic/claude-3-5-sonnet" "What\'s 2+2?"'
        )

        # Offer to test API connection
        response = input("\nWould you like to test the API connection? (y/N): ")
        if response.lower() in ["y", "yes"]:
            test_api_connection()
    else:
        print("\n❌ Setup validation failed")
        print("Please fix the issues above and run this script again.")

        # Offer to generate encryption keys
        response = input("\nWould you like to generate encryption keys? (y/N): ")
        if response.lower() in ["y", "yes"]:
            generate_encryption_keys()

    print("\n📚 For more help, see README.md or run: uv run second-opinion --help")


if __name__ == "__main__":
    main()
