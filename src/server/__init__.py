# src/server/__init__.py

"""
LIMA Traffic Counter - Server Package

This package contains the web API server components for collecting
and managing traffic count data from multiple camera installations.
"""

from .main import main

__all__ = ["main"]