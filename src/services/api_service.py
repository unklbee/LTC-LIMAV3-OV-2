# src/services/api_service.py
"""
API service for remote data submission with retry logic and batching.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import json
import backoff
from dataclasses import dataclass, asdict

import structlog

logger = structlog.get_logger()


@dataclass
class APIRequest:
    """API request data"""
    endpoint: str
    method: str
    data: Dict[str, Any]
    timestamp: datetime
    retry_count: int = 0


class APIService:
    """
    Async API service with:
    - Automatic retry with exponential backoff
    - Request batching
    - Rate limiting
    - Offline queue
    """

    def __init__(self,
                 base_url: str,
                 api_key: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 3,
                 batch_size: int = 100,
                 batch_interval: float = 5.0):

        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.batch_interval = batch_interval

        # Session
        self.session: Optional[aiohttp.ClientSession] = None

        # Request queue
        self.request_queue: deque[APIRequest] = deque(maxlen=10000)
        self.failed_queue: deque[APIRequest] = deque(maxlen=1000)

        # Batch processing
        self.batch_task = None
        self._stop_event = asyncio.Event()

        # Statistics
        self.stats = {
            'requests_sent': 0,
            'requests_failed': 0,
            'bytes_sent': 0,
            'last_error': None,
            'last_success': None
        }

    async def initialize(self):
        """Initialize API service"""
        # Create session with headers
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'LIMA-Traffic-Counter/1.0'
        }

        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=self.timeout
        )

        # Start batch processor
        self.batch_task = asyncio.create_task(self._batch_processor())

        logger.info(f"API service initialized: {self.base_url}")

    async def close(self):
        """Close API service"""
        self._stop_event.set()

        # Wait for batch processor
        if self.batch_task:
            await self.batch_task

        # Process remaining requests
        await self._flush_queue()

        # Close session
        if self.session:
            await self.session.close()

        logger.info("API service closed")

    # ==================== Public Methods ====================

    async def send_count(self, data: Dict[str, Any]):
        """Send vehicle count data"""
        request = APIRequest(
            endpoint='/counts',
            method='POST',
            data=data,
            timestamp=datetime.now()
        )

        self.request_queue.append(request)

    async def send_event(self, event_type: str, data: Dict[str, Any]):
        """Send event data"""
        request = APIRequest(
            endpoint='/events',
            method='POST',
            data={
                'event_type': event_type,
                'timestamp': datetime.now().isoformat(),
                **data
            },
            timestamp=datetime.now()
        )

        self.request_queue.append(request)

    async def send_statistics(self, stats: Dict[str, Any]):
        """Send aggregated statistics"""
        request = APIRequest(
            endpoint='/statistics',
            method='POST',
            data=stats,
            timestamp=datetime.now()
        )

        self.request_queue.append(request)

    async def get_configuration(self) -> Optional[Dict[str, Any]]:
        """Get remote configuration"""
        try:
            async with self.session.get(
                    f"{self.base_url}/configuration"
            ) as response:
                if response.status == 200:
                    return await response.json()

        except Exception as e:
            logger.error(f"Failed to get configuration: {e}")

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get API service statistics"""
        return {
            **self.stats,
            'queue_size': len(self.request_queue),
            'failed_queue_size': len(self.failed_queue)
        }

    # ==================== Private Methods ====================

    async def _batch_processor(self):
        """Process requests in batches"""
        while not self._stop_event.is_set():
            try:
                # Wait for batch interval
                await asyncio.sleep(self.batch_interval)

                # Process batch
                await self._process_batch()

                # Retry failed requests
                await self._retry_failed()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")

    async def _process_batch(self):
        """Process a batch of requests"""
        if not self.request_queue:
            return

        # Get batch
        batch = []
        while self.request_queue and len(batch) < self.batch_size:
            batch.append(self.request_queue.popleft())

        if not batch:
            return

        # Group by endpoint
        grouped = {}
        for request in batch:
            key = (request.endpoint, request.method)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(request)

        # Send grouped requests
        for (endpoint, method), requests in grouped.items():
            await self._send_batch(endpoint, method, requests)

    async def _send_batch(self, endpoint: str, method: str,
                          requests: List[APIRequest]):
        """Send batch of requests to same endpoint"""
        # Prepare batch data
        batch_data = {
            'batch': [req.data for req in requests],
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Send request with retry
            response = await self._send_with_retry(
                endpoint, method, batch_data
            )

            if response:
                self.stats['requests_sent'] += len(requests)
                self.stats['last_success'] = datetime.now()
                logger.info(f"Batch sent: {len(requests)} requests to {endpoint}")
            else:
                # Add to failed queue
                for req in requests:
                    req.retry_count += 1
                    if req.retry_count < self.max_retries:
                        self.failed_queue.append(req)

        except Exception as e:
            logger.error(f"Batch send error: {e}")
            self.stats['last_error'] = str(e)

            # Add to failed queue
            for req in requests:
                req.retry_count += 1
                if req.retry_count < self.max_retries:
                    self.failed_queue.append(req)

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=60
    )
    async def _send_with_retry(self, endpoint: str, method: str,
                               data: Dict[str, Any]) -> Optional[Dict]:
        """Send request with exponential backoff retry"""
        url = f"{self.base_url}{endpoint}"

        async with self.session.request(
                method=method,
                url=url,
                json=data
        ) as response:
            # Log request size
            self.stats['bytes_sent'] += len(json.dumps(data))

            if response.status == 200:
                return await response.json()
            elif response.status == 429:  # Rate limited
                # Get retry after header
                retry_after = response.headers.get('Retry-After', '60')
                await asyncio.sleep(int(retry_after))
                raise aiohttp.ClientError("Rate limited")
            elif response.status >= 500:
                # Server error, retry
                raise aiohttp.ClientError(f"Server error: {response.status}")
            else:
                # Client error, don't retry
                logger.error(f"API error: {response.status} - {await response.text()}")
                return None

    async def _retry_failed(self):
        """Retry failed requests"""
        if not self.failed_queue:
            return

        # Get requests to retry
        retry_batch = []
        while self.failed_queue and len(retry_batch) < self.batch_size // 2:
            request = self.failed_queue.popleft()

            # Check if request is too old
            age = datetime.now() - request.timestamp
            if age < timedelta(hours=24):  # Retry for up to 24 hours
                retry_batch.append(request)
            else:
                self.stats['requests_failed'] += 1

        # Add back to main queue
        self.request_queue.extend(retry_batch)

    async def _flush_queue(self):
        """Flush all pending requests"""
        while self.request_queue:
            await self._process_batch()


# ==================== WebSocket Support ====================

class WebSocketClient:
    """WebSocket client for real-time updates"""

    def __init__(self, url: str, api_key: Optional[str] = None):
        self.url = url
        self.api_key = api_key
        self.websocket = None
        self.running = False

        # Callbacks
        self.on_message_callbacks = []
        self.on_error_callbacks = []

    async def connect(self):
        """Connect to WebSocket server"""
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        self.websocket = await aiohttp.ClientSession().ws_connect(
            self.url,
            headers=headers
        )

        self.running = True

        # Start message handler
        asyncio.create_task(self._message_handler())

        logger.info(f"WebSocket connected: {self.url}")

    async def disconnect(self):
        """Disconnect from WebSocket"""
        self.running = False

        if self.websocket:
            await self.websocket.close()

    async def send(self, message: Dict[str, Any]):
        """Send message via WebSocket"""
        if self.websocket:
            await self.websocket.send_json(message)

    async def _message_handler(self):
        """Handle incoming messages"""
        try:
            async for msg in self.websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    # Trigger callbacks
                    for callback in self.on_message_callbacks:
                        try:
                            await callback(data)
                        except Exception as e:
                            logger.error(f"WebSocket callback error: {e}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg.data}")

                    # Trigger error callbacks
                    for callback in self.on_error_callbacks:
                        try:
                            await callback(msg.data)
                        except Exception as e:
                            logger.error(f"Error callback failed: {e}")

        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")

        finally:
            self.running = False

    def add_message_callback(self, callback):
        """Add message callback"""
        self.on_message_callbacks.append(callback)

    def add_error_callback(self, callback):
        """Add error callback"""
        self.on_error_callbacks.append(callback)