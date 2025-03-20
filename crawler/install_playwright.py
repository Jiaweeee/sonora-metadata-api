"""
Script to install Playwright browsers.
"""

import sys
import subprocess


def main():
    """Install Playwright browsers and dependencies."""
    try:
        print("Installing Playwright browsers...")
        result = subprocess.run(
            ["playwright", "install", "--with-deps", "chromium"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print("Playwright browsers installed successfully.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error installing Playwright browsers: {e}", file=sys.stderr)
        print(f"Error output: {e.stderr}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 