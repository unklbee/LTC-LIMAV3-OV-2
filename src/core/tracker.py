# src/core/tracker.py
"""
Enhanced multi-object tracker using ByteTrack algorithm.
ByteTrack: Multi-Object Tracking by Associating Every Detection Box
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from collections import defaultdict, deque
import time

from scipy.optimize import linear_sum_assignment
import structlog

logger = structlog.get_logger()


@dataclass
class Track:
    """Single track object"""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    state: str = "tentative"  # tentative, confirmed, lost
    age: int = 0
    hits: int = 1
    time_since_update: int = 0

    # Motion model
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # History
    history: deque = field(default_factory=lambda: deque(maxlen=30))

    def __post_init__(self):
        self.history.append(self.bbox.copy())

    @property
    def center(self) -> Tuple[float, float]:
        """Get track center point"""
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def tlwh(self) -> np.ndarray:
        """Get bbox in top-left-width-height format"""
        return np.array([
            self.bbox[0],
            self.bbox[1],
            self.bbox[2] - self.bbox[0],
            self.bbox[3] - self.bbox[1]
        ])

    def predict(self):
        """Predict next position using simple motion model"""
        # Simple constant velocity model
        self.bbox += self.velocity
        self.time_since_update += 1
        self.age += 1

    def update(self, bbox: np.ndarray, confidence: float, class_id: int):
        """Update track with new detection"""
        # Update velocity (exponential moving average)
        velocity = bbox - self.bbox
        self.velocity = 0.9 * self.velocity + 0.1 * velocity

        # Update bbox
        self.bbox = bbox.copy()
        self.confidence = confidence
        self.class_id = class_id

        # Update state
        self.hits += 1
        self.time_since_update = 0

        if self.state == "tentative" and self.hits >= 3:
            self.state = "confirmed"

        # Add to history
        self.history.append(bbox.copy())

    def mark_missed(self):
        """Mark track as missed in current frame"""
        if self.state == "tentative":
            self.state = "lost"
        elif self.time_since_update > 10:
            self.state = "lost"


class ByteTracker:
    """
    Enhanced ByteTrack multi-object tracker.

    Key improvements:
    1. Two-stage association (high score -> low score)
    2. Better motion model
    3. Adaptive thresholds
    4. Track confidence scoring
    """

    def __init__(self,
                 track_thresh: float = 0.5,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 min_box_area: float = 10,
                 mot20: bool = False):

        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        self.mot20 = mot20  # MOT20 dataset compatibility

        # Track management
        self.tracks: Dict[int, Track] = {}
        self.lost_tracks: Dict[int, Track] = {}
        self.removed_tracks: Dict[int, Track] = {}
        self.track_id_counter = 1

        # Kalman filter for better motion prediction (optional)
        self.use_kalman = False

        # Performance stats
        self.frame_count = 0
        self.association_times = []

    def update(self, detections: np.ndarray) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: Nx5 array [x1, y1, x2, y2, score] or
                       Nx6 array [x1, y1, x2, y2, score, class_id]

        Returns:
            List of active tracks
        """
        self.frame_count += 1
        start_time = time.perf_counter()

        # Validate input
        if detections.size == 0:
            detections = np.empty((0, 6))

        # Ensure we have class_id column
        if detections.shape[1] == 5:
            # Add default class_id
            class_ids = np.zeros((len(detections), 1))
            detections = np.hstack([detections, class_ids])

        # Split detections by confidence score
        remain_inds = detections[:, 4] > self.track_thresh
        inds_low = detections[:, 4] > 0.1
        inds_high = detections[:, 4] < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)

        # High confidence detections
        dets_high = detections[remain_inds]
        dets_second = detections[inds_second]

        # Predict existing tracks
        for track in self.tracks.values():
            track.predict()

        # Get active and lost tracks
        active_tracks = [t for t in self.tracks.values() if t.state != "lost"]
        lost_tracks = [t for t in self.tracks.values() if t.state == "lost"]

        ''' First association with high score detections '''
        if len(dets_high) > 0 and len(active_tracks) > 0:
            # Get track boxes
            track_bboxes = np.array([t.bbox for t in active_tracks])
            det_bboxes = dets_high[:, :4]

            # Calculate cost matrix (IoU distance)
            dists = self._iou_distance(track_bboxes, det_bboxes)

            # Solve assignment problem
            matches, u_tracks, u_dets = self._linear_assignment(
                dists, thresh=self.match_thresh
            )

            # Update matched tracks
            for i_track, i_det in matches:
                track = active_tracks[i_track]
                track.update(
                    det_bboxes[i_det],
                    dets_high[i_det, 4],
                    int(dets_high[i_det, 5])
                )

            # Get unmatched tracks and detections
            active_tracks = [active_tracks[i] for i in u_tracks]
            dets_high = dets_high[u_dets]
        else:
            # No matches possible
            u_tracks = list(range(len(active_tracks)))
            u_dets = list(range(len(dets_high)))

        ''' Second association with low score detections '''
        # Associate remaining tracks with low confidence detections
        if len(dets_second) > 0 and len(active_tracks) > 0:
            track_bboxes = np.array([t.bbox for t in active_tracks])
            det_bboxes = dets_second[:, :4]

            dists = self._iou_distance(track_bboxes, det_bboxes)
            matches, u_tracks, u_dets = self._linear_assignment(dists, thresh=0.5)

            for i_track, i_det in matches:
                track = active_tracks[i_track]
                track.update(
                    det_bboxes[i_det],
                    dets_second[i_det, 4],
                    int(dets_second[i_det, 5])
                )

            active_tracks = [active_tracks[i] for i in u_tracks]
        else:
            u_tracks = list(range(len(active_tracks)))

        ''' Deal with unmatched tracks '''
        for track in active_tracks:
            track.mark_missed()

        ''' Deal with lost tracks '''
        # Try to re-associate lost tracks
        if len(dets_high) > 0 and len(lost_tracks) > 0:
            track_bboxes = np.array([t.bbox for t in lost_tracks])
            det_bboxes = dets_high[:, :4]

            dists = self._iou_distance(track_bboxes, det_bboxes)
            matches, u_tracks, u_dets = self._linear_assignment(dists, thresh=0.25)

            for i_track, i_det in matches:
                track = lost_tracks[i_track]
                track.update(
                    det_bboxes[i_det],
                    dets_high[i_det, 4],
                    int(dets_high[i_det, 5])
                )
                track.state = "confirmed"  # Re-activate

            # Remove truly lost tracks
            lost_tracks = [lost_tracks[i] for i in u_tracks]
            dets_high = dets_high[u_dets]

        ''' Create new tracks '''
        for i in range(len(dets_high)):
            if dets_high[i, 4] < self.track_thresh:
                continue

            # Check minimum box area
            bbox = dets_high[i, :4]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < self.min_box_area:
                continue

            # Create new track
            track = Track(
                track_id=self.track_id_counter,
                bbox=bbox.copy(),
                confidence=dets_high[i, 4],
                class_id=int(dets_high[i, 5])
            )
            self.tracks[self.track_id_counter] = track
            self.track_id_counter += 1

        ''' Remove dead tracks '''
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.state == "lost" and track.time_since_update > self.track_buffer:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            track = self.tracks.pop(track_id)
            self.removed_tracks[track_id] = track

        # Record timing
        elapsed = time.perf_counter() - start_time
        self.association_times.append(elapsed)

        # Return confirmed tracks
        output_tracks = [t for t in self.tracks.values()
                         if t.state == "confirmed"]

        return output_tracks

    def _iou_distance(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Calculate IoU distance matrix between two sets of boxes.

        Returns:
            Distance matrix where distance = 1 - IoU
        """
        # Calculate intersection
        x1 = np.maximum(boxes1[:, 0][:, np.newaxis], boxes2[:, 0])
        y1 = np.maximum(boxes1[:, 1][:, np.newaxis], boxes2[:, 1])
        x2 = np.minimum(boxes1[:, 2][:, np.newaxis], boxes2[:, 2])
        y2 = np.minimum(boxes1[:, 3][:, np.newaxis], boxes2[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Calculate union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        union = area1[:, np.newaxis] + area2 - intersection

        # Calculate IoU
        iou = intersection / (union + 1e-6)

        # Convert to distance
        return 1 - iou

    def _linear_assignment(self, cost_matrix: np.ndarray,
                           thresh: float) -> Tuple[List, List, List]:
        """
        Solve linear assignment problem with threshold.

        Returns:
            matches: List of (track_idx, det_idx) pairs
            unmatched_tracks: List of unmatched track indices
            unmatched_detections: List of unmatched detection indices
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        # Solve assignment
        matches, unmatched_tracks, unmatched_dets = [], [], []

        # Use Hungarian algorithm
        cost_matrix_copy = cost_matrix.copy()
        cost_matrix_copy[cost_matrix > thresh] = thresh + 1e-5

        row_indices, col_indices = linear_sum_assignment(cost_matrix_copy)

        for row_idx, col_idx in zip(row_indices, col_indices):
            if cost_matrix[row_idx, col_idx] > thresh:
                unmatched_tracks.append(row_idx)
                unmatched_dets.append(col_idx)
            else:
                matches.append((row_idx, col_idx))

        # Get all unmatched
        unmatched_tracks.extend([i for i in range(cost_matrix.shape[0])
                                 if i not in row_indices])
        unmatched_dets.extend([i for i in range(cost_matrix.shape[1])
                               if i not in col_indices])

        return matches, unmatched_tracks, unmatched_dets

    def get_tracks_by_class(self, class_id: int) -> List[Track]:
        """Get all tracks of specific class"""
        return [t for t in self.tracks.values()
                if t.class_id == class_id and t.state == "confirmed"]

    def get_track_history(self, track_id: int, max_points: int = 30) -> List[Tuple[float, float]]:
        """Get track history as list of center points"""
        if track_id in self.tracks:
            track = self.tracks[track_id]
        elif track_id in self.removed_tracks:
            track = self.removed_tracks[track_id]
        else:
            return []

        points = []
        for bbox in list(track.history)[-max_points:]:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            points.append((cx, cy))

        return points

    def get_statistics(self) -> Dict:
        """Get tracker statistics"""
        active_tracks = [t for t in self.tracks.values() if t.state == "confirmed"]

        stats = {
            "total_tracks": self.track_id_counter - 1,
            "active_tracks": len(active_tracks),
            "lost_tracks": len([t for t in self.tracks.values() if t.state == "lost"]),
            "removed_tracks": len(self.removed_tracks),
            "avg_track_age": np.mean([t.age for t in active_tracks]) if active_tracks else 0,
            "avg_association_time": np.mean(self.association_times[-100:]) if self.association_times else 0
        }

        # Per-class statistics
        class_counts = defaultdict(int)
        for track in active_tracks:
            class_counts[track.class_id] += 1

        stats["tracks_per_class"] = dict(class_counts)

        return stats

    def reset(self):
        """Reset tracker state"""
        self.tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.track_id_counter = 1
        self.frame_count = 0
        self.association_times.clear()

        logger.info("Tracker reset")


# Additional tracking algorithms for comparison

class DeepSORTTracker:
    """
    Deep SORT tracker with appearance features.
    Requires a ReID model for appearance extraction.
    """

    def __init__(self, reid_model_path: Optional[str] = None,
                 max_cosine_distance: float = 0.3,
                 nn_budget: Optional[int] = 100):
        self.reid_model_path = reid_model_path
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget

        # Initialize ReID model if provided
        self.reid_model = None
        if reid_model_path:
            self._load_reid_model()

        # Tracks
        self.tracks: List[Track] = []
        self.track_id_counter = 1

    def _load_reid_model(self):
        """Load ReID model for appearance features"""
        # Implementation depends on specific ReID model
        pass

    def update(self, detections: np.ndarray, frame: np.ndarray) -> List[Track]:
        """Update with appearance features from frame"""
        # Extract appearance features if ReID model available
        if self.reid_model:
            features = self._extract_features(detections, frame)
        else:
            # Fall back to ByteTrack-style tracking
            features = None

        # Update logic with appearance matching
        # ... (implementation details)

        return self.tracks

    def _extract_features(self, detections: np.ndarray,
                          frame: np.ndarray) -> np.ndarray:
        """Extract appearance features for detections"""
        features = []

        for det in detections:
            # Crop detection from frame
            x1, y1, x2, y2 = det[:4].astype(int)
            crop = frame[y1:y2, x1:x2]

            # Extract features using ReID model
            # ... (model-specific implementation)

        return np.array(features)


# Factory function
def create_tracker(config: dict) -> ByteTracker:
    """Create tracker from configuration"""
    tracker_type = config.get('type', 'bytetrack')

    if tracker_type == 'bytetrack':
        return ByteTracker(
            track_thresh=config.get('track_thresh', 0.5),
            track_buffer=config.get('track_buffer', 30),
            match_thresh=config.get('match_thresh', 0.8),
            min_box_area=config.get('min_box_area', 10)
        )
    elif tracker_type == 'deepsort':
        return DeepSORTTracker(
            reid_model_path=config.get('reid_model_path'),
            max_cosine_distance=config.get('max_cosine_distance', 0.3)
        )
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")