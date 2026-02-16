"""
Risk Analysis Module
Evaluates crowd safety based on density, velocity, and movement patterns.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


@dataclass
class RiskReport:
    """Complete risk assessment report"""
    timestamp: float
    risk_level: RiskLevel
    density_risk: float  # 0-1
    velocity_risk: float  # 0-1
    convergence_risk: float  # 0-1
    stampede_probability: float  # 0-1
    overall_risk: float  # 0-1
    hotspots: List[Tuple[float, float, float]]  # (x, y, risk)
    warnings: List[str]
    recommendations: List[str]


class RiskAnalyzer:
    """
    Analyzes crowd risk based on multiple factors:
    - Density: People per unit area
    - Velocity: Sudden speed changes
    - Directional convergence: People moving toward same point
    - Entropy: Randomness in movement vectors
    """
    
    def __init__(self,
                 frame_width: int = 1920,
                 frame_height: int = 1080,
                 density_threshold: float = 4.0,  # people per m²
                 critical_density: float = 6.0,
                 velocity_threshold: float = 2.0,  # m/s
                 pixels_per_meter: float = 50.0):
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.density_threshold = density_threshold
        self.critical_density = critical_density
        self.velocity_threshold = velocity_threshold
        self.pixels_per_meter = pixels_per_meter
        
        # Grid for density calculation
        self.grid_size = 100  # pixels
        self.grid_cols = frame_width // self.grid_size
        self.grid_rows = frame_height // self.grid_size
        
        # History for trend analysis
        self.risk_history = []
        self.max_history = 300  # ~10 seconds at 30fps
        
        # Thresholds for risk levels
        self.risk_thresholds = {
            RiskLevel.SAFE: 0.2,
            RiskLevel.CAUTION: 0.4,
            RiskLevel.WARNING: 0.6,
            RiskLevel.DANGER: 0.8,
            RiskLevel.CRITICAL: 1.0
        }
    
    def set_frame_params(self, width: int, height: int, pixels_per_meter: float = 50.0):
        """Update frame parameters"""
        self.frame_width = width
        self.frame_height = height
        self.pixels_per_meter = pixels_per_meter
        self.grid_cols = width // self.grid_size
        self.grid_rows = height // self.grid_size
    
    def evaluate(self, 
                 current_state: Dict,
                 predicted_positions: Dict = None,
                 timestamp: float = 0.0) -> RiskReport:
        """
        Evaluate current and predicted risk.
        
        Args:
            current_state: Current positions from data collector
            predicted_positions: Optional predicted future positions
            timestamp: Current time
        
        Returns:
            RiskReport with comprehensive analysis
        """
        # Extract current positions and velocities
        positions = []
        velocities = []
        
        for obj_id, state in current_state.items():
            positions.append([state["x"], state["y"]])
            vx = state.get("vx", 0)
            vy = state.get("vy", 0)
            velocities.append([vx, vy])
        
        positions = np.array(positions) if positions else np.array([]).reshape(0, 2)
        velocities = np.array(velocities) if velocities else np.array([]).reshape(0, 2)
        
        # Calculate risk factors
        density_risk, density_grid = self._calculate_density_risk(positions)
        velocity_risk = self._calculate_velocity_risk(velocities)
        convergence_risk = self._calculate_convergence_risk(positions, velocities)
        entropy = self._calculate_movement_entropy(velocities)
        
        # Calculate stampede probability
        stampede_prob = self._calculate_stampede_probability(
            density_risk, velocity_risk, convergence_risk, entropy
        )
        
        # Analyze predictions if available
        prediction_risk = 0.0
        if predicted_positions:
            prediction_risk = self._analyze_predictions(predicted_positions)
        
        # Overall risk (weighted combination)
        weights = {
            "density": 0.35,
            "velocity": 0.20,
            "convergence": 0.20,
            "stampede": 0.15,
            "prediction": 0.10
        }
        
        overall_risk = (
            weights["density"] * density_risk +
            weights["velocity"] * velocity_risk +
            weights["convergence"] * convergence_risk +
            weights["stampede"] * stampede_prob +
            weights["prediction"] * prediction_risk
        )
        
        # Determine risk level
        risk_level = self._get_risk_level(overall_risk)
        
        # Find hotspots
        hotspots = self._find_hotspots(density_grid)
        
        # Generate warnings and recommendations
        warnings = self._generate_warnings(
            density_risk, velocity_risk, convergence_risk, stampede_prob
        )
        recommendations = self._generate_recommendations(risk_level, hotspots)
        
        # Create report
        report = RiskReport(
            timestamp=timestamp,
            risk_level=risk_level,
            density_risk=density_risk,
            velocity_risk=velocity_risk,
            convergence_risk=convergence_risk,
            stampede_probability=stampede_prob,
            overall_risk=overall_risk,
            hotspots=hotspots,
            warnings=warnings,
            recommendations=recommendations
        )
        
        # Store in history
        self.risk_history.append({
            "timestamp": timestamp,
            "overall_risk": overall_risk,
            "density_risk": density_risk,
            "velocity_risk": velocity_risk
        })
        if len(self.risk_history) > self.max_history:
            self.risk_history.pop(0)
        
        return report
    
    def _calculate_density_risk(self, positions: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate density-based risk using grid"""
        if len(positions) == 0:
            return 0.0, np.zeros((self.grid_rows, self.grid_cols))
        
        # Create density grid
        density_grid = np.zeros((self.grid_rows, self.grid_cols))
        
        for x, y in positions:
            col = min(int(x // self.grid_size), self.grid_cols - 1)
            row = min(int(y // self.grid_size), self.grid_rows - 1)
            if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                density_grid[row, col] += 1
        
        # Convert to people per m²
        cell_area_m2 = (self.grid_size / self.pixels_per_meter) ** 2
        density_grid_ppm2 = density_grid / cell_area_m2
        
        # Calculate risk based on max density
        max_density = np.max(density_grid_ppm2)
        
        if max_density < self.density_threshold:
            risk = max_density / self.density_threshold * 0.5
        elif max_density < self.critical_density:
            risk = 0.5 + (max_density - self.density_threshold) / \
                   (self.critical_density - self.density_threshold) * 0.3
        else:
            risk = 0.8 + min(0.2, (max_density - self.critical_density) / 4.0)
        
        return min(1.0, risk), density_grid_ppm2
    
    def _calculate_velocity_risk(self, velocities: np.ndarray) -> float:
        """Calculate risk from velocity (sudden movements, high speeds)"""
        if len(velocities) == 0:
            return 0.0
        
        # Calculate speeds
        speeds = np.sqrt(np.sum(velocities**2, axis=1)) / self.pixels_per_meter
        
        if len(speeds) == 0:
            return 0.0
        
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        speed_std = np.std(speeds)
        
        # Risk factors
        high_speed_risk = min(1.0, max_speed / (self.velocity_threshold * 2))
        variance_risk = min(1.0, speed_std / self.velocity_threshold)
        
        # Combined velocity risk
        risk = 0.5 * high_speed_risk + 0.5 * variance_risk
        
        return min(1.0, risk)
    
    def _calculate_convergence_risk(self, positions: np.ndarray, 
                                    velocities: np.ndarray) -> float:
        """Calculate risk from directional convergence"""
        if len(positions) < 3 or len(velocities) < 3:
            return 0.0
        
        # Find potential convergence points
        # Project positions forward and see if they cluster
        dt = 2.0  # Look 2 seconds ahead
        future_positions = positions + velocities * dt
        
        # Calculate pairwise distances in future
        convergence_count = 0
        total_pairs = 0
        
        for i in range(len(future_positions)):
            for j in range(i + 1, len(future_positions)):
                current_dist = np.linalg.norm(positions[i] - positions[j])
                future_dist = np.linalg.norm(future_positions[i] - future_positions[j])
                
                if current_dist > 10:  # Only consider non-adjacent
                    total_pairs += 1
                    if future_dist < current_dist * 0.5:  # Significant convergence
                        convergence_count += 1
        
        if total_pairs == 0:
            return 0.0
        
        convergence_ratio = convergence_count / total_pairs
        return min(1.0, convergence_ratio * 2)
    
    def _calculate_movement_entropy(self, velocities: np.ndarray) -> float:
        """Calculate entropy of movement directions (chaos indicator)"""
        if len(velocities) < 2:
            return 0.0
        
        # Calculate angles
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        
        # Bin into 8 directions
        bins = np.linspace(-np.pi, np.pi, 9)
        hist, _ = np.histogram(angles, bins=bins)
        
        # Normalize
        hist = hist / (np.sum(hist) + 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        max_entropy = np.log(8)  # Maximum for 8 bins
        
        normalized_entropy = entropy / max_entropy
        return normalized_entropy
    
    def _calculate_stampede_probability(self, density_risk: float,
                                        velocity_risk: float,
                                        convergence_risk: float,
                                        entropy: float) -> float:
        """
        Calculate probability of stampede based on combined factors.
        Stampede conditions:
        - High density + sudden velocity increase
        - Low entropy (everyone moving same direction) + high speed
        - High convergence + high density
        """
        # Factor 1: High density with high velocity
        factor1 = density_risk * velocity_risk
        
        # Factor 2: Coordinated fast movement (low entropy, high speed)
        factor2 = (1 - entropy) * velocity_risk
        
        # Factor 3: Convergence with density
        factor3 = convergence_risk * density_risk
        
        # Combined probability
        prob = max(factor1, factor2, factor3)
        
        # Scale up if multiple factors are high
        if factor1 > 0.5 and factor2 > 0.5:
            prob = min(1.0, prob * 1.3)
        if factor1 > 0.5 and factor3 > 0.5:
            prob = min(1.0, prob * 1.3)
        
        return min(1.0, prob)
    
    def _analyze_predictions(self, predicted_positions: Dict) -> float:
        """Analyze predicted positions for future risk"""
        if not predicted_positions:
            return 0.0
        
        # Look at different time horizons
        time_horizons = [30, 60, 150, 300]  # 1s, 2s, 5s, 10s at 30fps
        max_risk = 0.0
        
        for horizon in time_horizons:
            positions = []
            for obj_id, trajectory in predicted_positions.items():
                if len(trajectory) > horizon:
                    positions.append(trajectory[horizon])
            
            if len(positions) > 0:
                positions = np.array(positions)
                density_risk, _ = self._calculate_density_risk(positions)
                max_risk = max(max_risk, density_risk)
        
        return max_risk
    
    def _get_risk_level(self, overall_risk: float) -> RiskLevel:
        """Convert numerical risk to level"""
        if overall_risk < self.risk_thresholds[RiskLevel.SAFE]:
            return RiskLevel.SAFE
        elif overall_risk < self.risk_thresholds[RiskLevel.CAUTION]:
            return RiskLevel.CAUTION
        elif overall_risk < self.risk_thresholds[RiskLevel.WARNING]:
            return RiskLevel.WARNING
        elif overall_risk < self.risk_thresholds[RiskLevel.DANGER]:
            return RiskLevel.DANGER
        else:
            return RiskLevel.CRITICAL
    
    def _find_hotspots(self, density_grid: np.ndarray, 
                       top_n: int = 5) -> List[Tuple[float, float, float]]:
        """Find high-density hotspots"""
        hotspots = []
        
        # Find cells with density above threshold
        threshold = self.density_threshold * 0.7
        high_density_cells = np.argwhere(density_grid > threshold)
        
        for row, col in high_density_cells:
            x = (col + 0.5) * self.grid_size
            y = (row + 0.5) * self.grid_size
            risk = min(1.0, density_grid[row, col] / self.critical_density)
            hotspots.append((x, y, risk))
        
        # Sort by risk and return top N
        hotspots.sort(key=lambda h: h[2], reverse=True)
        return hotspots[:top_n]
    
    def _generate_warnings(self, density_risk: float, velocity_risk: float,
                          convergence_risk: float, stampede_prob: float) -> List[str]:
        """Generate warning messages based on risk factors"""
        warnings = []
        
        if density_risk > 0.7:
            warnings.append("CRITICAL: Crowd density approaching dangerous levels")
        elif density_risk > 0.5:
            warnings.append("WARNING: High crowd density detected")
        
        if velocity_risk > 0.7:
            warnings.append("ALERT: Rapid crowd movement detected")
        elif velocity_risk > 0.5:
            warnings.append("CAUTION: Elevated crowd velocity")
        
        if convergence_risk > 0.6:
            warnings.append("WARNING: Crowd convergence detected - bottleneck forming")
        
        if stampede_prob > 0.7:
            warnings.append("CRITICAL: High stampede risk - immediate action required")
        elif stampede_prob > 0.5:
            warnings.append("ALERT: Elevated stampede probability")
        
        return warnings
    
    def _generate_recommendations(self, risk_level: RiskLevel,
                                  hotspots: List[Tuple]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if risk_level == RiskLevel.SAFE:
            recommendations.append("Continue normal monitoring")
        
        elif risk_level == RiskLevel.CAUTION:
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Prepare crowd control measures")
        
        elif risk_level == RiskLevel.WARNING:
            recommendations.append("Deploy additional personnel to hotspots")
            recommendations.append("Consider opening additional exits/pathways")
            recommendations.append("Begin crowd dispersal announcements")
        
        elif risk_level == RiskLevel.DANGER:
            recommendations.append("URGENT: Activate crowd control protocols")
            recommendations.append("Open all available exits")
            recommendations.append("Deploy barriers at convergence points")
            recommendations.append("Make emergency announcements")
        
        elif risk_level == RiskLevel.CRITICAL:
            recommendations.append("EMERGENCY: Initiate evacuation procedures")
            recommendations.append("Contact emergency services")
            recommendations.append("Maximum crowd control deployment")
            recommendations.append("Continuous public address guidance")
        
        # Hotspot-specific recommendations
        if len(hotspots) > 0:
            for i, (x, y, risk) in enumerate(hotspots[:3]):
                if risk > 0.7:
                    recommendations.append(
                        f"Priority attention needed at zone ({int(x)}, {int(y)})"
                    )
        
        return recommendations
    
    def get_risk_trend(self) -> str:
        """Analyze risk trend over recent history"""
        if len(self.risk_history) < 30:
            return "STABLE"
        
        recent = [h["overall_risk"] for h in self.risk_history[-30:]]
        older = [h["overall_risk"] for h in self.risk_history[-60:-30]] \
                if len(self.risk_history) >= 60 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        diff = recent_avg - older_avg
        
        if diff > 0.1:
            return "INCREASING"
        elif diff < -0.1:
            return "DECREASING"
        else:
            return "STABLE"
    
    def get_summary(self) -> Dict:
        """Get summary of current risk state"""
        if len(self.risk_history) == 0:
            return {
                "current_risk": 0.0,
                "trend": "STABLE",
                "avg_risk_1min": 0.0,
                "max_risk_1min": 0.0
            }
        
        recent = self.risk_history[-180:]  # ~6 seconds
        
        return {
            "current_risk": self.risk_history[-1]["overall_risk"],
            "trend": self.get_risk_trend(),
            "avg_risk_1min": np.mean([h["overall_risk"] for h in recent]),
            "max_risk_1min": np.max([h["overall_risk"] for h in recent])
        }


# Color mappings for visualization
RISK_COLORS = {
    RiskLevel.SAFE: (0, 255, 0),        # Green
    RiskLevel.CAUTION: (0, 255, 255),   # Yellow
    RiskLevel.WARNING: (0, 165, 255),   # Orange
    RiskLevel.DANGER: (0, 0, 255),      # Red
    RiskLevel.CRITICAL: (0, 0, 139)     # Dark Red
}


def get_risk_color(risk_level: RiskLevel) -> Tuple[int, int, int]:
    """Get BGR color for risk level"""
    return RISK_COLORS.get(risk_level, (128, 128, 128))


def get_risk_color_by_value(risk: float) -> Tuple[int, int, int]:
    """Get BGR color interpolated by risk value (0-1)"""
    if risk < 0.25:
        # Green to Yellow
        t = risk / 0.25
        return (0, 255, int(255 * t))
    elif risk < 0.5:
        # Yellow to Orange
        t = (risk - 0.25) / 0.25
        return (0, int(255 - 90 * t), 255)
    elif risk < 0.75:
        # Orange to Red
        t = (risk - 0.5) / 0.25
        return (0, int(165 - 165 * t), 255)
    else:
        # Red to Dark Red
        t = (risk - 0.75) / 0.25
        return (0, 0, int(255 - 116 * t))
