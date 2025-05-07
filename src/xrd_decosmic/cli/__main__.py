"""
Entry point for running the CLI directly as a module.
This module allows the CLI to be run directly using the Python module syntax:
    python -m xrd_decosmic.cli
It simply imports and calls the main function from the main module.
"""
from .main import main
if __name__ == "__main__":
    main() 