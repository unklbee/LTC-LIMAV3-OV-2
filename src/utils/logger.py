# src/utils/logger.py
"""
Enhanced logging utility with structured logging, rotation, and remote logging support.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
import structlog
from pythonjsonlogger import jsonlogger
import asyncio
import aiohttp
from collections import deque


class AsyncRemoteHandler(logging.Handler):
    """Async handler for sending logs to remote server"""

    def __init__(self, url: str, api_key: Optional[str] = None,
                 batch_size: int = 100, flush_interval: float = 5.0):
        super().__init__()
        self.url = url
        self.api_key = api_key
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Buffer for log records
        self.buffer = deque(maxlen=1000)
        self.session: Optional[aiohttp.ClientSession] = None

        # Start background task
        self.task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self):
        """Periodically flush logs to remote server"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Remote logging error: {e}")

    async def _flush(self):
        """Flush buffered logs"""
        if not self.buffer:
            return

        # Get batch
        batch = []
        while self.buffer and len(batch) < self.batch_size:
            batch.append(self.buffer.popleft())

        if not batch:
            return

        # Create session if needed
        if not self.session:
            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            self.session = aiohttp.ClientSession(headers=headers)

        # Send logs
        try:
            payload = {
                'logs': batch,
                'timestamp': datetime.utcnow().isoformat()
            }

            async with self.session.post(self.url, json=payload) as response:
                if response.status != 200:
                    print(f"Failed to send logs: {response.status}")

        except Exception as e:
            print(f"Error sending logs: {e}")
            # Put logs back in buffer
            self.buffer.extend(batch)

    def emit(self, record):
        """Handle log record"""
        try:
            # Format record to dict
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }

            # Add extra fields
            if hasattr(record, 'extra'):
                log_entry.update(record.extra)

            # Add to buffer
            self.buffer.append(log_entry)

        except Exception:
            self.handleError(record)

    def close(self):
        """Close handler"""
        super().close()

        # Cancel task
        if self.task:
            self.task.cancel()

        # Close session
        if self.session:
            asyncio.create_task(self.session.close())


class ColoredConsoleHandler(logging.StreamHandler):
    """Console handler with colored output"""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }

    RESET = '\033[0m'

    def emit(self, record):
        """Emit colored log record"""
        try:
            # Get color for level
            color = self.COLORS.get(record.levelname, '')

            # Format message
            msg = self.format(record)

            # Add color if supported
            if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
                msg = f"{color}{msg}{self.RESET}"

            # Write to stream
            self.stream.write(msg + self.terminator)
            self.flush()

        except Exception:
            self.handleError(record)


def setup_logging(log_level: int = logging.INFO,
                  log_to_file: bool = True,
                  log_dir: Optional[Path] = None,
                  log_max_size: int = 10 * 1024 * 1024,
                  log_backup_count: int = 5,
                  remote_url: Optional[str] = None,
                  remote_api_key: Optional[str] = None):
    """
    Setup application logging with multiple handlers.
    
    Args:
        log_level: Logging level
        log_to_file: Enable file logging
        log_dir: Directory for log files
        log_max_size: Maximum size per log file
        log_backup_count: Number of backup files to keep
        remote_url: Remote logging endpoint
        remote_api_key: API key for remote logging
    """

    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler with colors
    console_handler = ColoredConsoleHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_to_file and log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

        # JSON format for file logs
        json_formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            timestamp=True
        )

        # Rotating file handler
        log_file = log_dir / f"lima_{datetime.now():%Y%m%d}.jsonl"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_max_size,
            backupCount=log_backup_count
        )
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)

        # Error file handler
        error_file = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=log_max_size,
            backupCount=2
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(json_formatter)
        root_logger.addHandler(error_handler)

    # Remote handler for centralized logging
    if remote_url:
        remote_handler = AsyncRemoteHandler(
            remote_url,
            api_key=remote_api_key
        )
        remote_handler.setLevel(logging.WARNING)  # Only send warnings and above
        root_logger.addHandler(remote_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.dict_tracebacks,
            structlog.processors.CallsiteParameterAdder(
                parameters=[structlog.processors.CallsiteParameter.FILENAME,
                            structlog.processors.CallsiteParameter.FUNC_NAME,
                            structlog.processors.CallsiteParameter.LINENO]
            ),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Log startup
    logger = structlog.get_logger()
    logger.info(
        "Logging initialized",
        level=logging.getLevelName(log_level),
        file_logging=log_to_file,
        remote_logging=bool(remote_url)
    )


class PerformanceLogger:
    """Logger for performance metrics"""

    def __init__(self, name: str):
        self.name = name
        self.logger = structlog.get_logger(name)
        self.metrics: Dict[str, list] = {}

    def log_metric(self, metric_name: str, value: float,
                   metadata: Optional[Dict] = None):
        """Log a performance metric"""
        # Store in memory
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append({
            'value': value,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        })

        # Log to file
        self.logger.info(
            "performance_metric",
            metric=metric_name,
            value=value,
            metadata=metadata
        )

    def log_fps(self, fps: float):
        """Log FPS metric"""
        self.log_metric('fps', fps)

    def log_inference_time(self, time_ms: float, batch_size: int = 1):
        """Log inference time"""
        self.log_metric(
            'inference_time_ms',
            time_ms,
            {'batch_size': batch_size}
        )

    def log_memory_usage(self, memory_mb: float, gpu_memory_mb: Optional[float] = None):
        """Log memory usage"""
        metadata = {'cpu_memory_mb': memory_mb}
        if gpu_memory_mb is not None:
            metadata['gpu_memory_mb'] = gpu_memory_mb

        self.log_metric('memory_usage', memory_mb, metadata)

    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if metric_name not in self.metrics:
            return {}

        values = [m['value'] for m in self.metrics[metric_name]]

        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'latest': values[-1] if values else 0
        }

    def log_summary(self):
        """Log summary of all metrics"""
        summary = {}

        for metric_name in self.metrics:
            stats = self.get_statistics(metric_name)
            summary[metric_name] = stats

        self.logger.info("performance_summary", metrics=summary)

        return summary


# Context manager for timing operations
class Timer:
    """Context manager for timing operations"""

    def __init__(self, name: str, logger: Optional[structlog.BoundLogger] = None):
        self.name = name
        self.logger = logger or structlog.get_logger()
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        self.logger.info(
            f"{self.name}_completed",
            duration_ms=duration_ms,
            success=exc_type is None
        )

        return False

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds() * 1000
        return 0.0