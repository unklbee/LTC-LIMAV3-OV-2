# src/services/database_service.py
"""
Async database service with connection pooling and automatic migrations.
"""

import asyncio
import aiosqlite
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import pandas as pd
from collections import defaultdict
import structlog

logger = structlog.get_logger()


@dataclass
class CountRecord:
    """Single count record"""
    id: Optional[int]
    host_id: str
    camera_id: str
    line_id: str
    timestamp: datetime
    vehicle_type: str
    count: int
    direction: str
    metadata: Optional[Dict] = None


class AsyncConnectionPool:
    """Async SQLite connection pool"""

    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self._initialized = False

    async def initialize(self):
        """Initialize connection pool"""
        if self._initialized:
            return

        # Create pool connections
        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.db_path)

            # Enable WAL mode for better concurrency
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")
            await conn.execute("PRAGMA synchronous=NORMAL")

            await self._pool.put(conn)

        self._initialized = True
        logger.info(f"Initialized connection pool with {self.pool_size} connections")

    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool"""
        if not self._initialized:
            await self.initialize()

        conn = await self._pool.get()
        try:
            yield conn
        finally:
            await self._pool.put(conn)

    async def close(self):
        """Close all connections in pool"""
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()

        self._initialized = False
        logger.info("Connection pool closed")


class DatabaseMigrations:
    """Database migration system"""

    migrations = [
        # Version 1: Initial schema
        {
            'version': 1,
            'sql': """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS vehicle_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                host_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                line_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                vehicle_type TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 1,
                direction TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                INDEX idx_timestamp (timestamp),
                INDEX idx_host_camera (host_id, camera_id),
                INDEX idx_vehicle_type (vehicle_type)
            );
            
            CREATE TABLE IF NOT EXISTS track_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                host_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                INDEX idx_track_timestamp (track_id, timestamp)
            );
            
            CREATE TABLE IF NOT EXISTS statistics_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT UNIQUE NOT NULL,
                data TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                INDEX idx_cache_key (cache_key),
                INDEX idx_expires (expires_at)
            );
            """
        },

        # Version 2: Add speed tracking
        {
            'version': 2,
            'sql': """
            ALTER TABLE vehicle_counts ADD COLUMN speed_kmh REAL;
            ALTER TABLE vehicle_counts ADD COLUMN confidence REAL;
            
            CREATE TABLE IF NOT EXISTS speed_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                host_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                speed_kmh REAL NOT NULL,
                speed_limit_kmh REAL NOT NULL,
                vehicle_type TEXT,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                INDEX idx_violations_timestamp (timestamp),
                INDEX idx_violations_speed (speed_kmh)
            );
            """
        },

        # Version 3: Add zones and analytics
        {
            'version': 3,
            'sql': """
            CREATE TABLE IF NOT EXISTS zone_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                host_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                zone_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,  -- 'entry' or 'exit'
                timestamp TIMESTAMP NOT NULL,
                duration_seconds REAL,  -- for exit events
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                INDEX idx_zone_timestamp (zone_id, timestamp)
            );
            
            CREATE TABLE IF NOT EXISTS hourly_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                host_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                hour_start TIMESTAMP NOT NULL,
                vehicle_type TEXT NOT NULL,
                total_count INTEGER NOT NULL DEFAULT 0,
                avg_speed_kmh REAL,
                max_speed_kmh REAL,
                
                UNIQUE(host_id, camera_id, hour_start, vehicle_type),
                INDEX idx_hourly_time (hour_start)
            );
            """
        }
    ]

    @classmethod
    async def migrate(cls, conn: aiosqlite.Connection):
        """Run database migrations"""
        # Get current version
        try:
            cursor = await conn.execute(
                "SELECT MAX(version) FROM schema_version"
            )
            row = await cursor.fetchone()
            current_version = row[0] if row[0] else 0
        except:
            current_version = 0

        # Apply migrations
        for migration in cls.migrations:
            if migration['version'] > current_version:
                logger.info(f"Applying migration version {migration['version']}")

                # Execute migration SQL
                for statement in migration['sql'].split(';'):
                    if statement.strip():
                        await conn.execute(statement)

                # Record migration
                await conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (migration['version'],)
                )

                await conn.commit()

        logger.info("Database migrations completed")


class DatabaseService:
    """
    Async database service with advanced features:
    - Connection pooling
    - Automatic migrations
    - Caching
    - Batch operations
    - Analytics queries
    """

    def __init__(self, db_path: str, host_id: str, camera_id: str,
                 pool_size: int = 5):
        self.db_path = Path(db_path)
        self.host_id = host_id
        self.camera_id = camera_id

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connection pool
        self.pool = AsyncConnectionPool(str(self.db_path), pool_size)

        # Batch insert buffers
        self.count_buffer: List[CountRecord] = []
        self.event_buffer: List[Dict] = []
        self.buffer_lock = asyncio.Lock()

        # Cache
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_ttl = 60  # seconds

        # Background tasks
        self._flush_task = None
        self._cleanup_task = None

    async def initialize(self):
        """Initialize database service"""
        # Initialize pool
        await self.pool.initialize()

        # Run migrations
        async with self.pool.acquire() as conn:
            await DatabaseMigrations.migrate(conn)

        # Start background tasks
        self._flush_task = asyncio.create_task(self._flush_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Database service initialized")

    async def close(self):
        """Close database service"""
        # Cancel background tasks
        if self._flush_task:
            self._flush_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Flush remaining data
        await self._flush_buffers()

        # Close pool
        await self.pool.close()

        logger.info("Database service closed")

    # ==================== Count Operations ====================

    async def insert_count(self, record: CountRecord):
        """Insert single count record"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO vehicle_counts 
                (host_id, camera_id, line_id, timestamp, vehicle_type, 
                 count, direction, metadata, speed_kmh, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.host_id,
                    record.camera_id,
                    record.line_id,
                    record.timestamp,
                    record.vehicle_type,
                    record.count,
                    record.direction,
                    json.dumps(record.metadata) if record.metadata else None,
                    record.metadata.get('speed_kmh') if record.metadata else None,
                    record.metadata.get('confidence') if record.metadata else None
                )
            )
            await conn.commit()

    async def insert_count_batch(self, records: List[CountRecord]):
        """Insert multiple count records efficiently"""
        if not records:
            return

        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO vehicle_counts 
                (host_id, camera_id, line_id, timestamp, vehicle_type, 
                 count, direction, metadata, speed_kmh, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.host_id, r.camera_id, r.line_id, r.timestamp,
                        r.vehicle_type, r.count, r.direction,
                        json.dumps(r.metadata) if r.metadata else None,
                        r.metadata.get('speed_kmh') if r.metadata else None,
                        r.metadata.get('confidence') if r.metadata else None
                    )
                    for r in records
                ]
            )
            await conn.commit()

        logger.info(f"Inserted {len(records)} count records")

    async def buffer_count(self, record: CountRecord):
        """Add count to buffer for batch insertion"""
        async with self.buffer_lock:
            self.count_buffer.append(record)

            # Auto-flush if buffer is large
            if len(self.count_buffer) >= 100:
                await self._flush_counts()

    async def _flush_counts(self):
        """Flush count buffer to database"""
        async with self.buffer_lock:
            if not self.count_buffer:
                return

            records = self.count_buffer.copy()
            self.count_buffer.clear()

        await self.insert_count_batch(records)

    # ==================== Query Operations ====================

    async def get_counts(self,
                         start_time: datetime,
                         end_time: datetime,
                         vehicle_types: Optional[List[str]] = None,
                         line_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Get counts within time range"""

        # Check cache
        cache_key = f"counts_{start_time}_{end_time}_{vehicle_types}_{line_ids}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Build query
        query = """
            SELECT 
                timestamp,
                line_id,
                vehicle_type,
                direction,
                count,
                speed_kmh,
                confidence
            FROM vehicle_counts
            WHERE host_id = ? AND camera_id = ?
                AND timestamp BETWEEN ? AND ?
        """
        params = [self.host_id, self.camera_id, start_time, end_time]

        if vehicle_types:
            placeholders = ','.join('?' * len(vehicle_types))
            query += f" AND vehicle_type IN ({placeholders})"
            params.extend(vehicle_types)

        if line_ids:
            placeholders = ','.join('?' * len(line_ids))
            query += f" AND line_id IN ({placeholders})"
            params.extend(line_ids)

        query += " ORDER BY timestamp"

        # Execute query
        async with self.pool.acquire() as conn:
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                columns = [d[0] for d in cursor.description]

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Cache result
        self._set_cached(cache_key, df)

        return df

    async def get_statistics(self,
                             start_time: datetime,
                             end_time: datetime,
                             interval: str = 'hour') -> Dict:
        """Get aggregated statistics"""

        # Get raw counts
        df = await self.get_counts(start_time, end_time)

        if df.empty:
            return {
                'total_counts': {},
                'counts_by_time': [],
                'avg_speed': None,
                'peak_hour': None
            }

        # Aggregate by vehicle type
        total_counts = df.groupby('vehicle_type')['count'].sum().to_dict()

        # Time series aggregation
        df.set_index('timestamp', inplace=True)

        if interval == 'hour':
            time_counts = df.groupby([
                pd.Grouper(freq='H'),
                'vehicle_type'
            ])['count'].sum().reset_index()
        elif interval == 'day':
            time_counts = df.groupby([
                pd.Grouper(freq='D'),
                'vehicle_type'
            ])['count'].sum().reset_index()
        else:  # minute
            time_counts = df.groupby([
                pd.Grouper(freq='T'),
                'vehicle_type'
            ])['count'].sum().reset_index()

        # Speed statistics
        speed_data = df[df['speed_kmh'].notna()]['speed_kmh']
        avg_speed = speed_data.mean() if not speed_data.empty else None
        max_speed = speed_data.max() if not speed_data.empty else None

        # Peak hour analysis
        hourly_total = df.groupby(pd.Grouper(freq='H'))['count'].sum()
        peak_hour = hourly_total.idxmax() if not hourly_total.empty else None

        return {
            'total_counts': total_counts,
            'counts_by_time': time_counts.to_dict('records'),
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'peak_hour': peak_hour,
            'total_vehicles': int(df['count'].sum())
        }

    async def get_hourly_statistics(self, date: datetime) -> pd.DataFrame:
        """Get pre-aggregated hourly statistics"""
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        async with self.pool.acquire() as conn:
            async with conn.execute(
                    """
                    SELECT 
                        hour_start,
                        vehicle_type,
                        total_count,
                        avg_speed_kmh,
                        max_speed_kmh
                    FROM hourly_statistics
                    WHERE host_id = ? AND camera_id = ?
                        AND hour_start >= ? AND hour_start < ?
                    ORDER BY hour_start, vehicle_type
                    """,
                    (self.host_id, self.camera_id, start, end)
            ) as cursor:
                rows = await cursor.fetchall()
                columns = [d[0] for d in cursor.description]

        df = pd.DataFrame(rows, columns=columns)
        df['hour_start'] = pd.to_datetime(df['hour_start'])

        return df

    # ==================== Event Operations ====================

    async def insert_track_event(self, track_id: int, event_type: str,
                                 data: Optional[Dict] = None):
        """Insert track event"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO track_events 
                (host_id, camera_id, track_id, event_type, timestamp, data)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.host_id,
                    self.camera_id,
                    track_id,
                    event_type,
                    datetime.now(),
                    json.dumps(data) if data else None
                )
            )
            await conn.commit()

    async def insert_zone_event(self, zone_id: str, track_id: int,
                                event_type: str, duration: Optional[float] = None,
                                metadata: Optional[Dict] = None):
        """Insert zone event"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO zone_events 
                (host_id, camera_id, zone_id, track_id, event_type, 
                 timestamp, duration_seconds, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.host_id,
                    self.camera_id,
                    zone_id,
                    track_id,
                    event_type,
                    datetime.now(),
                    duration,
                    json.dumps(metadata) if metadata else None
                )
            )
            await conn.commit()

    async def insert_speed_violation(self, track_id: int, speed_kmh: float,
                                     speed_limit_kmh: float, vehicle_type: str,
                                     image_path: Optional[str] = None):
        """Insert speed violation record"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO speed_violations 
                (host_id, camera_id, track_id, timestamp, speed_kmh, 
                 speed_limit_kmh, vehicle_type, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.host_id,
                    self.camera_id,
                    track_id,
                    datetime.now(),
                    speed_kmh,
                    speed_limit_kmh,
                    vehicle_type,
                    image_path
                )
            )
            await conn.commit()

    # ==================== Analytics Operations ====================

    async def update_hourly_statistics(self):
        """Update pre-aggregated hourly statistics"""
        # Get last hour's data
        end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(hours=1)

        df = await self.get_counts(start_time, end_time)

        if df.empty:
            return

        # Aggregate by vehicle type
        stats = df.groupby('vehicle_type').agg({
            'count': 'sum',
            'speed_kmh': ['mean', 'max']
        })

        # Insert/update statistics
        async with self.pool.acquire() as conn:
            for vehicle_type, row in stats.iterrows():
                await conn.execute(
                    """
                    INSERT OR REPLACE INTO hourly_statistics 
                    (host_id, camera_id, hour_start, vehicle_type, 
                     total_count, avg_speed_kmh, max_speed_kmh)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.host_id,
                        self.camera_id,
                        start_time,
                        vehicle_type,
                        int(row[('count', 'sum')]),
                        float(row[('speed_kmh', 'mean')]) if pd.notna(row[('speed_kmh', 'mean')]) else None,
                        float(row[('speed_kmh', 'max')]) if pd.notna(row[('speed_kmh', 'max')]) else None
                    )
                )
            await conn.commit()

    # ==================== Maintenance Operations ====================

    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        async with self.pool.acquire() as conn:
            # Delete old counts
            await conn.execute(
                "DELETE FROM vehicle_counts WHERE timestamp < ?",
                (cutoff_date,)
            )

            # Delete old events
            await conn.execute(
                "DELETE FROM track_events WHERE timestamp < ?",
                (cutoff_date,)
            )

            # Delete old cache
            await conn.execute(
                "DELETE FROM statistics_cache WHERE expires_at < ?",
                (datetime.now(),)
            )

            await conn.commit()

        logger.info(f"Cleaned up data older than {days_to_keep} days")

    async def vacuum(self):
        """Vacuum database to reclaim space"""
        async with self.pool.acquire() as conn:
            await conn.execute("VACUUM")

    # ==================== Cache Operations ====================

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self.cache:
            value, expires = self.cache[key]
            if time.time() < expires:
                return value
            else:
                del self.cache[key]
        return None

    def _set_cached(self, key: str, value: Any):
        """Set cached value"""
        expires = time.time() + self.cache_ttl
        self.cache[key] = (value, expires)

    # ==================== Background Tasks ====================

    async def _flush_loop(self):
        """Periodically flush buffers"""
        while True:
            try:
                await asyncio.sleep(5)  # Flush every 5 seconds
                await self._flush_buffers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush loop error: {e}")

    async def _cleanup_loop(self):
        """Periodically run cleanup tasks"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.update_hourly_statistics()
                await self.cleanup_old_data()

                # Clean expired cache
                now = time.time()
                expired_keys = [k for k, (_, exp) in self.cache.items() if exp < now]
                for key in expired_keys:
                    del self.cache[key]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _flush_buffers(self):
        """Flush all buffers to database"""
        await self._flush_counts()


# Factory function
async def create_database_service(config: dict) -> DatabaseService:
    """Create and initialize database service"""
    service = DatabaseService(
        db_path=config['db_path'],
        host_id=config['host_id'],
        camera_id=config['camera_id'],
        pool_size=config.get('pool_size', 5)
    )

    await service.initialize()

    return service