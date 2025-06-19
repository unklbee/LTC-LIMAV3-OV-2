# src/services/__init__.py

"""
Services package for LIMA Traffic Counter.

This package contains service layer components that provide higher-level
interfaces for common operations like database access, configuration management,
and other business logic.
"""

from .database_service import (
    DatabaseService,
    CountRecord,
    create_database_service,
    save_vehicle_counts,
    create_count_record_from_detection
)

__all__ = [
    "DatabaseService",
    "CountRecord",
    "create_database_service",
    "save_vehicle_counts",
    "create_count_record_from_detection"
]