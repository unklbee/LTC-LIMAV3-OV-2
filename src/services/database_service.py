# src/services/database_service.py

import logging
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import asyncio
import aiosqlite

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


class CountDatabase:
    """Simple database implementation for count data."""

    def __init__(self, db_path: str, host_id: str = None, api_url: str = None):
        self.db_path = Path(db_path)
        self.host_id = host_id or "default-host"
        self.api_url = api_url

        # Create database directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Table agregat interval (total per waktu)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vehicle_counts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    host_id TEXT NOT NULL,
                    interval_start TIMESTAMP NOT NULL,
                    interval_end TIMESTAMP NOT NULL,
                    bicycle INTEGER DEFAULT 0,
                    car INTEGER DEFAULT 0,
                    truck INTEGER DEFAULT 0,
                    motorbike INTEGER DEFAULT 0,
                    bus INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Table detail per kendaraan lewat
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vehicle_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    host_id TEXT,
                    camera_id TEXT,
                    line_id TEXT,
                    timestamp TIMESTAMP,
                    vehicle_type TEXT,
                    direction TEXT,
                    confidence REAL,
                    speed REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()


    def insert_interval_counts(self, interval_start: datetime,
                               interval_end: datetime, counts: Dict[int, int]):
        """Insert interval counts into database."""
        record = CountRecord.from_counts_dict(
            self.host_id, interval_start, interval_end, counts
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO vehicle_counts 
                (host_id, interval_start, interval_end, bicycle, car, truck, motorbike, bus)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.host_id,
                record.interval_start,
                record.interval_end,
                record.bicycle,
                record.car,
                record.truck,
                record.motorbike,
                record.bus
            ))
            conn.commit()

    def insert_vehicle_event(self, camera_id: str, line_id: str, timestamp, vehicle_type: str, direction: str, confidence: float, speed: float = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO vehicle_events (host_id, camera_id, line_id, timestamp, vehicle_type, direction, confidence, speed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (self.host_id, camera_id, line_id, timestamp, vehicle_type, direction, confidence, speed))
            conn.commit()

    def set_host_id(self, host_id: str):
        """Set host ID."""
        self.host_id = host_id

    def set_api_url(self, api_url: str):
        """Set API URL."""
        self.api_url = api_url

    def close(self):
        """Close database connection."""
        pass  # SQLite connections are closed automatically


class DatabaseService:
    """Service class for database operations with cleaner interface."""

    def __init__(self, db_path: str, host_id: str = None, api_url: str = None):
        """Initialize database service."""
        self.db_path = Path(db_path)
        self.host_id = host_id or "default-host"
        self.api_url = api_url

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

    async def close(self):
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
        asyncio.run(self.close())

    async def buffer_count(self, record: CountRecord):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.save_count_record, record)

    def insert_vehicle_event(self, camera_id: str, line_id: str, timestamp, vehicle_type: str, direction: str, confidence: float, speed: float = None):
        if self._db:
            self._db.insert_vehicle_event(camera_id, line_id, timestamp, vehicle_type, direction, confidence, speed)

    async def get_statistics(self, start_time, end_time):
        result = {
            'total_counts': {},
            'count_rate': 0.0,
            'avg_speed': 0.0,   # default float, bukan None
            'max_speed': 0.0
            }
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT vehicle_type, COUNT(*) as total
                    FROM vehicle_events
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY vehicle_type
                """, (start_time, end_time)) as cursor:
                    rows = await cursor.fetchall()
                    result['total_counts'] = {row[0]: row[1] for row in rows}

                async with db.execute("""
                    SELECT COUNT(*) FROM vehicle_events
                    WHERE timestamp BETWEEN ? AND ?
                """, (start_time, end_time)) as cursor:
                    total = (await cursor.fetchone())[0]
                    duration_minutes = max((end_time - start_time).total_seconds() / 60, 1)
                    result['count_rate'] = float(total) / duration_minutes

                async with db.execute("""
                    SELECT AVG(speed), MAX(speed)
                    FROM vehicle_events
                    WHERE timestamp BETWEEN ? AND ?
                    AND speed IS NOT NULL
                """, (start_time, end_time)) as cursor:
                    speed_stats = await cursor.fetchone()
                    # Handle None with ternary, PASTI float
                    result['avg_speed'] = float(speed_stats[0]) if speed_stats and speed_stats[0] is not None else 0.0
                    result['max_speed'] = float(speed_stats[1]) if speed_stats and speed_stats[1] is not None else 0.0

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Database statistics error: {e}")

        return result



async def create_database_service(config: dict = None) -> DatabaseService:
    """Factory function to create a database service instance."""
    if config is None:
        config = {}

    # Default configuration
    from ..models.config import get_config
    app_config = get_config()

    db_path = config.get('db_path', str(app_config.db_path))
    host_id = config.get('host_id', app_config.host_id)
    api_url = config.get('api_url', app_config.api_url)

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
        service = DatabaseService(
            db_path="./data/counts.db",
            host_id=host_id or "default-host"
        )
        with service:
            return service.save_counts(interval_start, interval_end, counts)
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