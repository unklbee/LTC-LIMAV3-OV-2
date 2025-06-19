# src/core/counter.py
"""
Advanced vehicle counting system with multi-line support and direction detection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Callable
from collections import defaultdict, deque
from enum import Enum
import numpy as np
import cv2
import time
from datetime import datetime, timedelta

import structlog

logger = structlog.get_logger()


class Direction(Enum):
    """Crossing direction"""
    NONE = 0
    FORWARD = 1
    BACKWARD = -1


@dataclass
class CountingLine:
    """Single counting line definition"""
    line_id: str
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    direction_vector: Optional[Tuple[float, float]] = None
    count_forward: bool = True
    count_backward: bool = True
    active: bool = True

    def __post_init__(self):
        """Calculate line properties"""
        # Calculate line vector
        self.vector = (
            self.end_point[0] - self.start_point[0],
            self.end_point[1] - self.start_point[1]
        )

        # Calculate normal vector (perpendicular)
        length = np.sqrt(self.vector[0]**2 + self.vector[1]**2)
        if length > 0:
            self.normal = (-self.vector[1] / length, self.vector[0] / length)
        else:
            self.normal = (0, 0)

        # If direction vector not specified, use normal
        if self.direction_vector is None:
            self.direction_vector = self.normal

    def get_side(self, point: Tuple[float, float]) -> float:
        """
        Get which side of the line a point is on.
        Returns positive for one side, negative for other, 0 if on line.
        """
        x, y = point
        x1, y1 = self.start_point
        x2, y2 = self.end_point

        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    def check_crossing(self, prev_point: Tuple[float, float],
                       curr_point: Tuple[float, float]) -> Direction:
        """Check if trajectory crosses this line and in which direction"""
        prev_side = self.get_side(prev_point)
        curr_side = self.get_side(curr_point)

        # No crossing if on same side
        if prev_side * curr_side >= 0:
            return Direction.NONE

        # Determine crossing direction based on movement vector
        movement = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
        dot_product = (movement[0] * self.direction_vector[0] +
                       movement[1] * self.direction_vector[1])

        if dot_product > 0:
            return Direction.FORWARD if self.count_forward else Direction.NONE
        else:
            return Direction.BACKWARD if self.count_backward else Direction.NONE


@dataclass
class CountingZone:
    """Advanced counting zone with entry/exit detection"""
    zone_id: str
    polygon: List[Tuple[float, float]]
    entry_lines: List[str] = field(default_factory=list)
    exit_lines: List[str] = field(default_factory=list)
    max_time_inside: float = 300.0  # seconds

    def __post_init__(self):
        """Convert polygon to numpy array"""
        self.polygon_array = np.array(self.polygon, dtype=np.float32)

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside zone"""
        result = cv2.pointPolygonTest(self.polygon_array, point, False)
        return result >= 0


@dataclass
class VehicleCount:
    """Vehicle counting statistics"""
    timestamp: datetime
    line_id: str
    direction: Direction
    vehicle_class: int
    track_id: int
    confidence: float
    speed: Optional[float] = None


class AdvancedCounter:
    """
    Advanced vehicle counting system with multiple features:
    - Multi-line counting with direction detection
    - Zone-based counting (entry/exit)
    - Speed estimation
    - Class-specific counting rules
    - Event callbacks
    """

    def __init__(self,
                 counting_lines: List[CountingLine] = None,
                 counting_zones: List[CountingZone] = None,
                 enable_speed_estimation: bool = True,
                 fps: float = 30.0,
                 pixel_per_meter: float = 10.0):

        # Counting lines and zones
        self.lines: Dict[str, CountingLine] = {}
        self.zones: Dict[str, CountingZone] = {}

        if counting_lines:
            for line in counting_lines:
                self.add_line(line)

        if counting_zones:
            for zone in counting_zones:
                self.add_zone(zone)

        # Track states
        self.track_positions: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        self.track_sides: Dict[int, Dict[str, float]] = defaultdict(dict)
        self.counted_tracks: Dict[str, Set[int]] = defaultdict(set)
        self.zone_tracks: Dict[str, Dict[int, float]] = defaultdict(dict)

        # Counting statistics
        self.counts: Dict[str, Dict[int, Dict[Direction, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self.recent_counts: deque = deque(maxlen=1000)

        # Speed estimation
        self.enable_speed_estimation = enable_speed_estimation
        self.fps = fps
        self.pixel_per_meter = pixel_per_meter

        # Callbacks
        self.on_count_callbacks: List[Callable] = []
        self.on_zone_entry_callbacks: List[Callable] = []
        self.on_zone_exit_callbacks: List[Callable] = []

        # Performance stats
        self.processing_times = deque(maxlen=100)

    def add_line(self, line: CountingLine):
        """Add counting line"""
        self.lines[line.line_id] = line
        logger.info(f"Added counting line: {line.line_id}")

    def add_zone(self, zone: CountingZone):
        """Add counting zone"""
        self.zones[zone.zone_id] = zone
        logger.info(f"Added counting zone: {zone.zone_id}")

    def update(self, tracks: List['Track']) -> Dict[str, int]:
        """
        Update counter with new tracks.

        Returns:
            Current total counts per class
        """
        start_time = time.perf_counter()

        # Process each track
        for track in tracks:
            # Update position history
            center = track.center
            self.track_positions[track.track_id].append((center, time.time()))

            # Check line crossings
            self._check_line_crossings(track)

            # Check zone events
            if self.zones:
                self._check_zone_events(track)

        # Clean old tracks
        self._clean_old_tracks(tracks)

        # Record processing time
        elapsed = time.perf_counter() - start_time
        self.processing_times.append(elapsed)

        # Return current totals
        return self.get_total_counts()

    def _check_line_crossings(self, track: 'Track'):
        """Check if track crosses any counting lines"""
        track_id = track.track_id
        positions = self.track_positions[track_id]

        if len(positions) < 2:
            return

        # Get current and previous positions
        curr_pos, curr_time = positions[-1]
        prev_pos, prev_time = positions[-2]

        # Check each active line
        for line_id, line in self.lines.items():
            if not line.active:
                continue

            # Skip if already counted on this line
            if track_id in self.counted_tracks[line_id]:
                continue

            # Check crossing
            direction = line.check_crossing(prev_pos, curr_pos)

            if direction != Direction.NONE:
                # Record count
                self._record_count(
                    line_id=line_id,
                    direction=direction,
                    track=track,
                    speed=self._estimate_speed(positions) if self.enable_speed_estimation else None
                )

                # Mark as counted
                self.counted_tracks[line_id].add(track_id)

    def _check_zone_events(self, track: 'Track'):
        """Check zone entry/exit events"""
        track_id = track.track_id
        center = track.center
        current_time = time.time()

        for zone_id, zone in self.zones.items():
            inside = zone.contains_point(center)
            was_inside = track_id in self.zone_tracks[zone_id]

            if inside and not was_inside:
                # Entry event
                self.zone_tracks[zone_id][track_id] = current_time
                self._trigger_zone_entry(zone_id, track)

            elif not inside and was_inside:
                # Exit event
                entry_time = self.zone_tracks[zone_id].pop(track_id)
                duration = current_time - entry_time
                self._trigger_zone_exit(zone_id, track, duration)

            elif inside and was_inside:
                # Check if exceeded max time
                entry_time = self.zone_tracks[zone_id][track_id]
                if current_time - entry_time > zone.max_time_inside:
                    logger.warning(f"Track {track_id} exceeded max time in zone {zone_id}")

    def _estimate_speed(self, positions: deque) -> Optional[float]:
        """Estimate speed from position history"""
        if len(positions) < 3:
            return None

        # Use positions over last 0.5 seconds
        current_time = positions[-1][1]
        valid_positions = [(p, t) for p, t in positions
                           if current_time - t <= 0.5]

        if len(valid_positions) < 2:
            return None

        # Calculate distance traveled
        total_distance = 0
        for i in range(1, len(valid_positions)):
            p1, _ = valid_positions[i-1]
            p2, _ = valid_positions[i]
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_distance += distance

        # Convert to meters
        distance_meters = total_distance / self.pixel_per_meter

        # Calculate time elapsed
        time_elapsed = valid_positions[-1][1] - valid_positions[0][1]

        if time_elapsed > 0:
            # Speed in m/s
            speed_mps = distance_meters / time_elapsed
            # Convert to km/h
            return speed_mps * 3.6

        return None

    def _record_count(self, line_id: str, direction: Direction,
                      track: 'Track', speed: Optional[float] = None):
        """Record a counting event"""
        # Update statistics
        self.counts[line_id][track.class_id][direction] += 1

        # Create count record
        count = VehicleCount(
            timestamp=datetime.now(),
            line_id=line_id,
            direction=direction,
            vehicle_class=track.class_id,
            track_id=track.track_id,
            confidence=track.confidence,
            speed=speed
        )

        self.recent_counts.append(count)

        # Trigger callbacks
        for callback in self.on_count_callbacks:
            try:
                callback(count)
            except Exception as e:
                logger.error(f"Count callback error: {e}")

        logger.info(f"Counted vehicle: class={track.class_id}, "
                    f"line={line_id}, direction={direction.name}, "
                    f"speed={speed:.1f} km/h" if speed else "")

    def _trigger_zone_entry(self, zone_id: str, track: 'Track'):
        """Trigger zone entry callbacks"""
        for callback in self.on_zone_entry_callbacks:
            try:
                callback(zone_id, track)
            except Exception as e:
                logger.error(f"Zone entry callback error: {e}")

    def _trigger_zone_exit(self, zone_id: str, track: 'Track', duration: float):
        """Trigger zone exit callbacks"""
        for callback in self.on_zone_exit_callbacks:
            try:
                callback(zone_id, track, duration)
            except Exception as e:
                logger.error(f"Zone exit callback error: {e}")

    def _clean_old_tracks(self, active_tracks: List['Track']):
        """Remove data for tracks that are no longer active"""
        active_ids = {t.track_id for t in active_tracks}

        # Clean position history
        track_ids = list(self.track_positions.keys())
        for track_id in track_ids:
            if track_id not in active_ids:
                # Check if last update was > 30 seconds ago
                if self.track_positions[track_id]:
                    last_time = self.track_positions[track_id][-1][1]
                    if time.time() - last_time > 30:
                        del self.track_positions[track_id]

        # Clean counted tracks
        for line_id in self.counted_tracks:
            self.counted_tracks[line_id] = {
                tid for tid in self.counted_tracks[line_id]
                if tid in active_ids
            }

    def get_total_counts(self) -> Dict[int, int]:
        """Get total counts per vehicle class"""
        totals = defaultdict(int)

        for line_counts in self.counts.values():
            for class_id, direction_counts in line_counts.items():
                # Sum forward and backward counts
                total = (direction_counts[Direction.FORWARD] +
                         direction_counts[Direction.BACKWARD])
                totals[class_id] += total

        return dict(totals)

    def get_line_counts(self, line_id: str) -> Dict[int, Dict[str, int]]:
        """Get counts for specific line"""
        if line_id not in self.counts:
            return {}

        result = {}
        for class_id, direction_counts in self.counts[line_id].items():
            result[class_id] = {
                'forward': direction_counts[Direction.FORWARD],
                'backward': direction_counts[Direction.BACKWARD],
                'total': (direction_counts[Direction.FORWARD] +
                          direction_counts[Direction.BACKWARD])
            }

        return result

    def get_recent_counts(self, minutes: int = 5) -> List[VehicleCount]:
        """Get counts from last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [c for c in self.recent_counts if c.timestamp > cutoff_time]

    def get_statistics(self) -> Dict:
        """Get counting statistics"""
        recent_counts = self.get_recent_counts(minutes=5)

        # Calculate speeds if available
        speeds = [c.speed for c in recent_counts if c.speed is not None]

        stats = {
            'total_counts': self.get_total_counts(),
            'counts_per_line': {
                line_id: self.get_line_counts(line_id)
                for line_id in self.lines
            },
            'recent_count_rate': len(recent_counts) / 5.0,  # per minute
            'avg_speed': np.mean(speeds) if speeds else None,
            'max_speed': np.max(speeds) if speeds else None,
            'active_tracks_in_zones': {
                zone_id: len(tracks)
                for zone_id, tracks in self.zone_tracks.items()
            },
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0
        }

        return stats

    def reset(self):
        """Reset all counts"""
        self.counts.clear()
        self.recent_counts.clear()
        self.counted_tracks.clear()
        self.zone_tracks.clear()
        self.track_positions.clear()

        logger.info("Counter reset")

    def add_count_callback(self, callback: Callable):
        """Add callback for count events"""
        self.on_count_callbacks.append(callback)

    def add_zone_callback(self, entry_callback: Callable = None,
                          exit_callback: Callable = None):
        """Add callbacks for zone events"""
        if entry_callback:
            self.on_zone_entry_callbacks.append(entry_callback)
        if exit_callback:
            self.on_zone_exit_callbacks.append(exit_callback)


# Factory function
def create_counter(config: dict) -> AdvancedCounter:
    """Create counter from configuration"""
    # Parse counting lines
    lines = []
    for line_config in config.get('counting_lines', []):
        line = CountingLine(
            line_id=line_config['id'],
            start_point=tuple(line_config['start']),
            end_point=tuple(line_config['end']),
            direction_vector=tuple(line_config.get('direction', [])) or None,
            count_forward=line_config.get('count_forward', True),
            count_backward=line_config.get('count_backward', True)
        )
        lines.append(line)

    # Parse counting zones
    zones = []
    for zone_config in config.get('counting_zones', []):
        zone = CountingZone(
            zone_id=zone_config['id'],
            polygon=[tuple(p) for p in zone_config['polygon']],
            entry_lines=zone_config.get('entry_lines', []),
            exit_lines=zone_config.get('exit_lines', []),
            max_time_inside=zone_config.get('max_time_inside', 300)
        )
        zones.append(zone)

    # Create counter
    counter = AdvancedCounter(
        counting_lines=lines,
        counting_zones=zones,
        enable_speed_estimation=config.get('enable_speed_estimation', True),
        fps=config.get('fps', 30.0),
        pixel_per_meter=config.get('pixel_per_meter', 10.0)
    )

    return counter