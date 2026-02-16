"""
Crowd Data Collection Layer
Collects, structures, and manages tracking data for prediction model training.
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import time
import json


class TrackingPoint:
    """Single tracking point with position and timestamp"""
    
    def __init__(self, x: float, y: float, t: float):
        self.x = x
        self.y = y
        self.t = t
    
    def to_dict(self):
        return {"x": self.x, "y": self.y, "t": self.t}
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["x"], data["y"], data["t"])


class ObjectTrack:
    """Complete track for a single object with computed features"""
    
    def __init__(self, object_id: int, max_history: int = 1000):
        self.object_id = object_id
        self.max_history = max_history
        self.positions: deque = deque(maxlen=max_history)
        self.velocities: deque = deque(maxlen=max_history)
        self.accelerations: deque = deque(maxlen=max_history)
        
    def add_position(self, x: float, y: float, t: float):
        """Add new position and compute velocity/acceleration"""
        point = TrackingPoint(x, y, t)
        
        # Compute velocity if we have previous position
        if len(self.positions) > 0:
            prev = self.positions[-1]
            dt = t - prev.t
            if dt > 0:
                vx = (x - prev.x) / dt
                vy = (y - prev.y) / dt
                self.velocities.append({"vx": vx, "vy": vy, "t": t})
                
                # Compute acceleration if we have previous velocity
                if len(self.velocities) > 1:
                    prev_v = self.velocities[-2]
                    dt_v = t - prev_v["t"]
                    if dt_v > 0:
                        ax = (vx - prev_v["vx"]) / dt_v
                        ay = (vy - prev_v["vy"]) / dt_v
                        self.accelerations.append({"ax": ax, "ay": ay, "t": t})
        
        self.positions.append(point)
    
    def get_speed(self) -> float:
        """Get current speed"""
        if len(self.velocities) == 0:
            return 0.0
        v = self.velocities[-1]
        return np.sqrt(v["vx"]**2 + v["vy"]**2)
    
    def get_direction(self) -> float:
        """Get current movement direction in radians"""
        if len(self.velocities) == 0:
            return 0.0
        v = self.velocities[-1]
        return np.arctan2(v["vy"], v["vx"])
    
    def get_recent_positions(self, time_window: float) -> List[TrackingPoint]:
        """Get positions within the last time_window seconds"""
        if len(self.positions) == 0:
            return []
        
        current_time = self.positions[-1].t
        cutoff_time = current_time - time_window
        
        result = []
        for p in reversed(self.positions):
            if p.t >= cutoff_time:
                result.append(p)
            else:
                break
        return list(reversed(result))
    
    def get_sequence_data(self, seq_length: int) -> Optional[np.ndarray]:
        """Get sequence data for model input [x, y, vx, vy, speed, direction]"""
        if len(self.positions) < seq_length or len(self.velocities) < seq_length:
            return None
        
        sequence = []
        positions = list(self.positions)[-seq_length:]
        velocities = list(self.velocities)[-seq_length:]
        
        for i in range(seq_length):
            p = positions[i]
            v = velocities[i]
            speed = np.sqrt(v["vx"]**2 + v["vy"]**2)
            direction = np.arctan2(v["vy"], v["vx"])
            sequence.append([p.x, p.y, v["vx"], v["vy"], speed, direction])
        
        return np.array(sequence, dtype=np.float32)


class CrowdDataCollector:
    """
    Collects and manages tracking data from the crowd tracker.
    Provides data in formats suitable for model training.
    """
    
    def __init__(self, 
                 max_objects: int = 500,
                 max_history_per_object: int = 1000,
                 time_window: float = 30.0,  # seconds
                 density_radius: float = 100.0):  # pixels
        
        self.max_objects = max_objects
        self.max_history_per_object = max_history_per_object
        self.time_window = time_window
        self.density_radius = density_radius
        
        self.tracks: Dict[int, ObjectTrack] = {}
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 30.0  # Will be updated
        
        # Store completed tracks for training
        self.completed_tracks: List[ObjectTrack] = []
        self.max_completed_tracks = 1000
        
        # Global statistics
        self.global_stats = {
            "avg_speed": 0.0,
            "avg_density": 0.0,
            "total_objects_tracked": 0
        }
    
    def update(self, tracker_objects: dict, frame_time: Optional[float] = None):
        """
        Update with current tracker state.
        
        Args:
            tracker_objects: Dictionary of {object_id: (cx, cy)} from CentroidTracker
            frame_time: Optional explicit timestamp, otherwise uses elapsed time
        """
        self.frame_count += 1
        
        if frame_time is None:
            frame_time = time.time() - self.start_time
        
        current_ids = set(tracker_objects.keys())
        existing_ids = set(self.tracks.keys())
        
        # Handle new objects
        new_ids = current_ids - existing_ids
        for obj_id in new_ids:
            if len(self.tracks) < self.max_objects:
                self.tracks[obj_id] = ObjectTrack(obj_id, self.max_history_per_object)
                self.global_stats["total_objects_tracked"] += 1
        
        # Handle disappeared objects
        disappeared_ids = existing_ids - current_ids
        for obj_id in disappeared_ids:
            if obj_id in self.tracks:
                track = self.tracks[obj_id]
                if len(track.positions) >= 30:  # Only save meaningful tracks
                    self.completed_tracks.append(track)
                    if len(self.completed_tracks) > self.max_completed_tracks:
                        self.completed_tracks.pop(0)
                del self.tracks[obj_id]
        
        # Update existing objects
        for obj_id, centroid in tracker_objects.items():
            if obj_id in self.tracks:
                cx, cy = centroid
                self.tracks[obj_id].add_position(float(cx), float(cy), frame_time)
        
        # Update global statistics
        self._update_global_stats()
    
    def _update_global_stats(self):
        """Update global statistics"""
        if len(self.tracks) == 0:
            return
        
        speeds = [track.get_speed() for track in self.tracks.values()]
        self.global_stats["avg_speed"] = np.mean(speeds) if speeds else 0.0
        
        # Compute average density
        densities = []
        positions = [(t.positions[-1].x, t.positions[-1].y) 
                     for t in self.tracks.values() if len(t.positions) > 0]
        
        for i, (x1, y1) in enumerate(positions):
            count = sum(1 for j, (x2, y2) in enumerate(positions) 
                       if i != j and np.sqrt((x2-x1)**2 + (y2-y1)**2) < self.density_radius)
            densities.append(count)
        
        self.global_stats["avg_density"] = np.mean(densities) if densities else 0.0
    
    def get_local_density(self, x: float, y: float) -> int:
        """Get number of objects within density_radius of (x, y)"""
        count = 0
        for track in self.tracks.values():
            if len(track.positions) > 0:
                p = track.positions[-1]
                dist = np.sqrt((p.x - x)**2 + (p.y - y)**2)
                if dist < self.density_radius:
                    count += 1
        return count
    
    def get_current_state(self) -> Dict:
        """Get current state of all tracked objects"""
        state = {}
        for obj_id, track in self.tracks.items():
            if len(track.positions) > 0:
                p = track.positions[-1]
                state[obj_id] = {
                    "x": p.x,
                    "y": p.y,
                    "t": p.t,
                    "speed": track.get_speed(),
                    "direction": track.get_direction(),
                    "local_density": self.get_local_density(p.x, p.y)
                }
                if len(track.velocities) > 0:
                    v = track.velocities[-1]
                    state[obj_id]["vx"] = v["vx"]
                    state[obj_id]["vy"] = v["vy"]
        return state
    
    def get_training_sequences(self, seq_length: int = 30, 
                                future_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training sequences from collected data.
        
        Returns:
            X: Input sequences [batch, seq_length, features]
            Y: Target future positions [batch, future_steps, 2]
        """
        X_list = []
        Y_list = []
        
        # From active tracks
        for track in self.tracks.values():
            sequences = self._extract_sequences(track, seq_length, future_steps)
            if sequences is not None:
                X_list.extend(sequences[0])
                Y_list.extend(sequences[1])
        
        # From completed tracks
        for track in self.completed_tracks:
            sequences = self._extract_sequences(track, seq_length, future_steps)
            if sequences is not None:
                X_list.extend(sequences[0])
                Y_list.extend(sequences[1])
        
        if len(X_list) == 0:
            return np.array([]), np.array([])
        
        return np.array(X_list), np.array(Y_list)
    
    def _extract_sequences(self, track: ObjectTrack, 
                           seq_length: int, future_steps: int) -> Optional[Tuple[List, List]]:
        """Extract training sequences from a single track"""
        min_length = seq_length + future_steps
        
        if len(track.positions) < min_length or len(track.velocities) < min_length:
            return None
        
        positions = list(track.positions)
        velocities = list(track.velocities)
        
        X_sequences = []
        Y_sequences = []
        
        # Sliding window
        for i in range(len(positions) - min_length + 1):
            # Input sequence
            x_seq = []
            for j in range(seq_length):
                idx = i + j
                p = positions[idx]
                v = velocities[idx] if idx < len(velocities) else {"vx": 0, "vy": 0}
                speed = np.sqrt(v["vx"]**2 + v["vy"]**2)
                direction = np.arctan2(v["vy"], v["vx"])
                x_seq.append([p.x, p.y, v["vx"], v["vy"], speed, direction])
            
            # Target future positions
            y_seq = []
            for j in range(future_steps):
                idx = i + seq_length + j
                if idx < len(positions):
                    p = positions[idx]
                    y_seq.append([p.x, p.y])
            
            if len(y_seq) == future_steps:
                X_sequences.append(x_seq)
                Y_sequences.append(y_seq)
        
        return X_sequences, Y_sequences
    
    def save_data(self, filepath: str):
        """Save collected data to file"""
        data = {
            "tracks": {},
            "completed_tracks": [],
            "global_stats": self.global_stats,
            "frame_count": self.frame_count
        }
        
        for obj_id, track in self.tracks.items():
            data["tracks"][obj_id] = [p.to_dict() for p in track.positions]
        
        for track in self.completed_tracks:
            data["completed_tracks"].append({
                "object_id": track.object_id,
                "positions": [p.to_dict() for p in track.positions]
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load_data(self, filepath: str):
        """Load previously saved data"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.global_stats = data.get("global_stats", self.global_stats)
        self.frame_count = data.get("frame_count", 0)
        
        # Load completed tracks
        for track_data in data.get("completed_tracks", []):
            track = ObjectTrack(track_data["object_id"], self.max_history_per_object)
            for p in track_data["positions"]:
                track.add_position(p["x"], p["y"], p["t"])
            self.completed_tracks.append(track)


class FeatureExtractor:
    """Extract features from tracking data for model input"""
    
    def __init__(self, frame_width: int, frame_height: int, 
                 density_radius: float = 100.0):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.density_radius = density_radius
    
    def normalize_position(self, x: float, y: float) -> Tuple[float, float]:
        """Normalize position to [0, 1]"""
        return x / self.frame_width, y / self.frame_height
    
    def denormalize_position(self, x_norm: float, y_norm: float) -> Tuple[float, float]:
        """Convert normalized position back to pixels"""
        return x_norm * self.frame_width, y_norm * self.frame_height
    
    def extract_features(self, current_state: Dict, 
                         all_positions: List[Tuple[float, float]]) -> Dict:
        """
        Extract features for a single object.
        
        Features:
        - Normalized position (x, y)
        - Velocity (vx, vy)
        - Speed
        - Direction
        - Local density
        - Distance to boundary
        """
        features = {}
        
        x, y = current_state["x"], current_state["y"]
        norm_x, norm_y = self.normalize_position(x, y)
        
        features["norm_x"] = norm_x
        features["norm_y"] = norm_y
        features["vx"] = current_state.get("vx", 0.0)
        features["vy"] = current_state.get("vy", 0.0)
        features["speed"] = current_state.get("speed", 0.0)
        features["direction"] = current_state.get("direction", 0.0)
        features["local_density"] = current_state.get("local_density", 0)
        
        # Distance to boundaries (normalized)
        features["dist_left"] = norm_x
        features["dist_right"] = 1.0 - norm_x
        features["dist_top"] = norm_y
        features["dist_bottom"] = 1.0 - norm_y
        
        return features
    
    def create_feature_vector(self, features: Dict) -> np.ndarray:
        """Create feature vector for model input"""
        return np.array([
            features["norm_x"],
            features["norm_y"],
            features["vx"],
            features["vy"],
            features["speed"],
            features["direction"],
            features["local_density"],
            features["dist_left"],
            features["dist_right"],
            features["dist_top"],
            features["dist_bottom"]
        ], dtype=np.float32)
