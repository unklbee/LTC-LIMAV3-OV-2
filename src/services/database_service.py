# src/services/database_service.py

import logging
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path

from src.object_detector.db import CountDatabase
from src.object_detector.config import Config

logger = logging.getLogger(__name__)


@dataclass
class CountRecord:
    """Data class representing a count record for easier handling."""
    id: Optional[int] = None
    host_id: str = ""
    interval_start: datetime = None
    interval_end: datetime = None
    bicycle: int = 0
    car: int = 0
    truck: int = 0
    motorbike: int = 0
    bus: int = 0

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "host_id": self.host_id,
            "interval_start": self.interval_start,
            "interval_end": self.interval_end,
            "bicycle": self.bicycle,
            "car": self.car,
            "truck": self.truck,
            "motorbike": self.motorbike,
            "bus": self.bus,
        }

    @classmethod
    def from_counts_dict(
            cls,
            host_id: str,
            interval_start: datetime,
            interval_end: datetime,
            counts: Dict[int, int]
    ) -> "CountRecord":
        """Create CountRecord from vehicle counts dictionary."""
        return cls(
            host_id=host_id,
            interval_start=interval_start,
            interval_end=interval_end,
            bicycle=counts.get(1, 0),      # class ID 1 = bicycle
            car=counts.get(2, 0),          # class ID 2 = car
            truck=counts.get(7, 0),        # class ID 7 = truck
            motorbike=counts.get(3, 0),    # class ID 3 = motorbike
            bus=counts.get(5, 0),          # class ID 5 = bus
        )


class DatabaseService:
    """Service class for database operations with cleaner interface."""

    def __init__(self, db_path: str, host_id: str = None, api_url: str = None):
        """Initialize database service."""
        self.db_path = Path(db_path)
        self.host_id = host_id or Config.HOST_ID
        self.api_url = api_url or Config.API_URL

        # Initialize the underlying database
        try:
            self._db = CountDatabase(
                str(self.db_path),
                host_id=self.host_id,
                api_url=self.api_url
            )
            logger.info(f"Database service initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self._db = None
            raise

    def save_count_record(self, record: CountRecord) -> bool:
        """Save a count record to the database."""
        if not self._db:
            logger.warning("Database not available")
            return False

        try:
            # Convert CountRecord to the format expected by CountDatabase
            counts_dict = {
                1: record.bicycle,
                2: record.car,
                3: record.motorbike,
                5: record.bus,
                7: record.truck,
            }

            self._db.insert_interval_counts(
                record.interval_start,
                record.interval_end,
                counts_dict
            )

            logger.debug(f"Saved count record: {record}")
            return True

        except Exception as e:
            logger.error(f"Failed to save count record: {e}")
            return False

    def save_counts(
            self,
            interval_start: datetime,
            interval_end: datetime,
            counts: Dict[int, int]
    ) -> bool:
        """Save vehicle counts for a time interval."""
        if not self._db:
            logger.warning("Database not available")
            return False

        try:
            self._db.insert_interval_counts(
                interval_start,
                interval_end,
                counts
            )
            logger.debug(f"Saved counts: {counts} for interval {interval_start} - {interval_end}")
            return True

        except Exception as e:
            logger.error(f"Failed to save counts: {e}")
            return False

    def update_host_id(self, host_id: str):
        """Update the host ID for this service."""
        self.host_id = host_id
        if self._db:
            self._db.set_host_id(host_id)

    def update_api_url(self, api_url: str):
        """Update the API URL for this service."""
        self.api_url = api_url
        if self._db:
            self._db.set_api_url(api_url)

    def is_available(self) -> bool:
        """Check if database is available."""
        return self._db is not None

    def get_connection_info(self) -> Dict[str, str]:
        """Get database connection information."""
        return {
            "db_path": str(self.db_path),
            "host_id": self.host_id,
            "api_url": self.api_url,
            "status": "connected" if self._db else "disconnected"
        }

    def close(self):
        """Close database connection and cleanup resources."""
        if self._db:
            try:
                self._db.close()
                logger.info("Database service closed")
            except Exception as e:
                logger.warning(f"Error closing database: {e}")
            finally:
                self._db = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_database_service(
        db_path: str = None,
        host_id: str = None,
        api_url: str = None
) -> DatabaseService:
    """Factory function to create a database service instance."""
    db_path = db_path or str(Config.DB_PATH)
    host_id = host_id or Config.HOST_ID
    api_url = api_url or Config.API_URL

    try:
        service = DatabaseService(db_path, host_id, api_url)
        logger.info("Database service created successfully")
        return service
    except Exception as e:
        logger.error(f"Failed to create database service: {e}")
        raise


# Convenience functions for common operations
def save_vehicle_counts(
        counts: Dict[int, int],
        interval_start: datetime,
        interval_end: datetime,
        host_id: str = None
) -> bool:
    """Convenience function to save vehicle counts."""
    try:
        with create_database_service(host_id=host_id) as db_service:
            return db_service.save_counts(interval_start, interval_end, counts)
    except Exception as e:
        logger.error(f"Failed to save vehicle counts: {e}")
        return False


def create_count_record_from_detection(
        host_id: str,
        interval_start: datetime,
        interval_end: datetime,
        detection_counts: Dict[int, int]
) -> CountRecord:
    """Create a CountRecord from detection results."""
    return CountRecord.from_counts_dict(
        host_id=host_id,
        interval_start=interval_start,
        interval_end=interval_end,
        counts=detection_counts
    )