"""Entry point for the frozen build: GUI with no arguments, CLI otherwise."""
import sys

from spacerocksuite.cli import main

if __name__ == "__main__":
    sys.exit(main())
