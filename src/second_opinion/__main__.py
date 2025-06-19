"""
Entry point for running Second Opinion as a module.

This allows users to run: python -m second_opinion
"""

from second_opinion.cli.main import app

if __name__ == "__main__":
    app()