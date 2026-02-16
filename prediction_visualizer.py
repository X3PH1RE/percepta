"""
Extended Visualization Module
Adds prediction visualization, density maps, and risk indicators.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from risk_analyzer import RiskReport, RiskLevel, get_risk_color, get_risk_color_by_value


class PredictionVisualizer:
    """
    Extended visualizer that shows:
    - Current tracked positions
    - Predicted future positions
    - Movement arrows
    - Density heatmap
    - Risk indicators
    """
    
    def __init__(self, 
                 width: int = 800,
                 height: int = 600,
                 bg_color: Tuple[int, int, int] = (15, 15, 25)):
        
        self.width = width
        self.height = height
        self.bg_color = bg_color
        
        # Source frame dimensions (for scaling)
        self.source_width = 1920
        self.source_height = 1080
        
        # Visual settings
        self.dot_radius = 4
        self.prediction_dot_radius = 3
        self.trail_length = 30
        
        # Color palette for tracked objects
        self.colors = [
            (0, 255, 255),    # Cyan
            (255, 100, 100),  # Light blue
            (100, 255, 100),  # Light green
            (255, 150, 50),   # Orange-blue
            (200, 100, 255),  # Pink
            (100, 255, 255),  # Yellow
            (255, 200, 100),  # Light cyan
            (150, 255, 150),  # Lime
        ]
        
        # Prediction colors (more muted/transparent look)
        self.prediction_color = (180, 180, 100)  # Pale cyan
        self.prediction_warning_color = (100, 100, 255)  # Pale red
        
        # Animation state
        self.animation_frame = 0
        self.prediction_horizon = 0  # Current prediction time being shown
    
    def set_source_dimensions(self, width: int, height: int):
        """Set source video dimensions for coordinate scaling"""
        self.source_width = width
        self.source_height = height
    
    def scale_point(self, x: float, y: float) -> Tuple[int, int]:
        """Scale point from source to visualization coordinates"""
        scale_x = self.width / self.source_width
        scale_y = self.height / self.source_height
        return int(x * scale_x), int(y * scale_y)
    
    def get_object_color(self, obj_id: int) -> Tuple[int, int, int]:
        """Get color for object ID"""
        return self.colors[obj_id % len(self.colors)]
    
    def create_base_canvas(self) -> np.ndarray:
        """Create base canvas with grid"""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = self.bg_color
        
        # Draw grid
        grid_color = (35, 35, 45)
        grid_spacing = 50
        
        for x in range(0, self.width, grid_spacing):
            cv2.line(canvas, (x, 0), (x, self.height), grid_color, 1)
        for y in range(0, self.height, grid_spacing):
            cv2.line(canvas, (0, y), (self.width, y), grid_color, 1)
        
        return canvas
    
    def draw_current_positions(self, canvas: np.ndarray,
                               current_state: Dict,
                               trails: Dict = None):
        """Draw current tracked positions with optional trails"""
        
        for obj_id, state in current_state.items():
            color = self.get_object_color(obj_id)
            x, y = self.scale_point(state["x"], state["y"])
            
            # Draw trail if available
            if trails and obj_id in trails:
                trail = trails[obj_id]
                for i in range(1, len(trail)):
                    pt1 = self.scale_point(trail[i-1][0], trail[i-1][1])
                    pt2 = self.scale_point(trail[i][0], trail[i][1])
                    alpha = i / len(trail)
                    fade_color = tuple(int(c * alpha * 0.5) for c in color)
                    cv2.line(canvas, pt1, pt2, fade_color, 1)
            
            # Draw current position dot
            cv2.circle(canvas, (x, y), self.dot_radius, color, -1)
            
            # Draw ID label
            cv2.putText(canvas, str(obj_id), (x + 6, y + 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    def draw_predictions(self, canvas: np.ndarray,
                         predictions: Dict,
                         time_horizon: int = 60,
                         show_trajectory: bool = True):
        """
        Draw predicted positions.
        
        Args:
            canvas: Canvas to draw on
            predictions: {obj_id: [[x, y], ...]} predicted trajectories
            time_horizon: How many frames into future to show
            show_trajectory: Whether to show full predicted path
        """
        for obj_id, trajectory in predictions.items():
            if len(trajectory) == 0:
                continue
            
            base_color = self.get_object_color(obj_id)
            
            # Draw predicted trajectory
            if show_trajectory:
                max_frames = min(time_horizon, len(trajectory))
                for i in range(1, max_frames, 3):  # Skip frames for cleaner look
                    pt1 = self.scale_point(trajectory[i-1][0], trajectory[i-1][1])
                    pt2 = self.scale_point(trajectory[i][0], trajectory[i][1])
                    
                    # Fade out over time
                    alpha = 1 - (i / max_frames)
                    pred_color = tuple(int(c * alpha * 0.4) for c in base_color)
                    cv2.line(canvas, pt1, pt2, pred_color, 1)
            
            # Draw predicted position at horizon
            if time_horizon < len(trajectory):
                pred_pos = trajectory[time_horizon]
                px, py = self.scale_point(pred_pos[0], pred_pos[1])
                
                # Predicted dot (hollow circle)
                pred_color = tuple(int(c * 0.6) for c in base_color)
                cv2.circle(canvas, (px, py), self.prediction_dot_radius + 2, pred_color, 1)
                cv2.circle(canvas, (px, py), 1, pred_color, -1)
    
    def draw_movement_arrows(self, canvas: np.ndarray,
                             current_state: Dict,
                             arrow_scale: float = 0.5):
        """Draw arrows showing movement direction"""
        
        for obj_id, state in current_state.items():
            x, y = self.scale_point(state["x"], state["y"])
            
            vx = state.get("vx", 0)
            vy = state.get("vy", 0)
            
            speed = np.sqrt(vx**2 + vy**2)
            if speed < 1:  # Skip if not moving
                continue
            
            # Scale velocity for visualization
            scale_x = self.width / self.source_width
            scale_y = self.height / self.source_height
            
            end_x = int(x + vx * arrow_scale * scale_x)
            end_y = int(y + vy * arrow_scale * scale_y)
            
            color = self.get_object_color(obj_id)
            cv2.arrowedLine(canvas, (x, y), (end_x, end_y), color, 1, tipLength=0.3)
    
    def draw_density_heatmap(self, canvas: np.ndarray,
                             positions: List[Tuple[float, float]],
                             radius: float = 50.0,
                             alpha: float = 0.3):
        """Overlay density heatmap on canvas"""
        
        if len(positions) == 0:
            return
        
        # Create heatmap
        heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        
        for px, py in positions:
            x, y = self.scale_point(px, py)
            
            # Add gaussian blob
            y_grid, x_grid = np.ogrid[-y:self.height-y, -x:self.width-x]
            mask = x_grid**2 + y_grid**2 <= radius**2
            heatmap[mask] += 1
        
        # Normalize
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Blend with canvas
        mask = heatmap > 0.1
        for c in range(3):
            canvas[:, :, c] = np.where(
                mask,
                canvas[:, :, c] * (1 - alpha * heatmap) + 
                heatmap_colored[:, :, c] * alpha * heatmap,
                canvas[:, :, c]
            ).astype(np.uint8)
    
    def draw_risk_indicator(self, canvas: np.ndarray,
                            risk_report: RiskReport):
        """Draw risk indicator panel"""
        
        # Panel dimensions
        panel_x = 10
        panel_y = 10
        panel_width = 180
        panel_height = 120
        
        # Background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (30, 30, 40), -1)
        cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)
        
        # Border color based on risk
        border_color = get_risk_color(risk_report.risk_level)
        cv2.rectangle(canvas, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     border_color, 2)
        
        # Title
        cv2.putText(canvas, "RISK ANALYSIS", (panel_x + 10, panel_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Risk level
        level_text = risk_report.risk_level.value.upper()
        cv2.putText(canvas, f"Level: {level_text}", (panel_x + 10, panel_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1)
        
        # Risk bars
        bar_y = panel_y + 55
        bar_height = 8
        bar_max_width = 100
        
        # Density risk bar
        cv2.putText(canvas, "Density", (panel_x + 10, bar_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        bar_width = int(risk_report.density_risk * bar_max_width)
        bar_color = get_risk_color_by_value(risk_report.density_risk)
        cv2.rectangle(canvas, (panel_x + 60, bar_y - 6),
                     (panel_x + 60 + bar_width, bar_y), bar_color, -1)
        
        # Velocity risk bar
        bar_y += 15
        cv2.putText(canvas, "Velocity", (panel_x + 10, bar_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        bar_width = int(risk_report.velocity_risk * bar_max_width)
        bar_color = get_risk_color_by_value(risk_report.velocity_risk)
        cv2.rectangle(canvas, (panel_x + 60, bar_y - 6),
                     (panel_x + 60 + bar_width, bar_y), bar_color, -1)
        
        # Stampede probability
        bar_y += 15
        cv2.putText(canvas, "Stampede", (panel_x + 10, bar_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        bar_width = int(risk_report.stampede_probability * bar_max_width)
        bar_color = get_risk_color_by_value(risk_report.stampede_probability)
        cv2.rectangle(canvas, (panel_x + 60, bar_y - 6),
                     (panel_x + 60 + bar_width, bar_y), bar_color, -1)
        
        # Overall risk
        bar_y += 20
        cv2.putText(canvas, f"Overall: {risk_report.overall_risk:.1%}", 
                   (panel_x + 10, bar_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def draw_hotspots(self, canvas: np.ndarray,
                      hotspots: List[Tuple[float, float, float]]):
        """Draw hotspot indicators"""
        
        for x, y, risk in hotspots:
            sx, sy = self.scale_point(x, y)
            
            # Pulsing effect
            pulse = abs(np.sin(self.animation_frame * 0.1)) * 0.3 + 0.7
            
            color = get_risk_color_by_value(risk)
            radius = int(20 + risk * 15)
            
            # Outer ring (pulsing)
            cv2.circle(canvas, (sx, sy), int(radius * pulse), color, 1)
            
            # Inner warning
            if risk > 0.7:
                cv2.circle(canvas, (sx, sy), 5, (0, 0, 255), -1)
    
    def draw_prediction_timeline(self, canvas: np.ndarray,
                                 current_horizon: int,
                                 max_horizon: int,
                                 fps: int = 30):
        """Draw timeline showing prediction horizon"""
        
        timeline_y = self.height - 30
        timeline_x1 = 50
        timeline_x2 = self.width - 50
        timeline_width = timeline_x2 - timeline_x1
        
        # Background bar
        cv2.rectangle(canvas, (timeline_x1, timeline_y - 5),
                     (timeline_x2, timeline_y + 5), (40, 40, 50), -1)
        
        # Progress bar
        progress = current_horizon / max_horizon if max_horizon > 0 else 0
        progress_x = timeline_x1 + int(timeline_width * progress)
        cv2.rectangle(canvas, (timeline_x1, timeline_y - 3),
                     (progress_x, timeline_y + 3), (100, 200, 100), -1)
        
        # Labels
        cv2.putText(canvas, "Now", (timeline_x1 - 25, timeline_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        time_seconds = max_horizon / fps
        cv2.putText(canvas, f"+{time_seconds:.0f}s", 
                   (timeline_x2 + 5, timeline_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # Current time marker
        current_time = current_horizon / fps
        cv2.putText(canvas, f"+{current_time:.1f}s", 
                   (progress_x - 15, timeline_y - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    def draw_warnings(self, canvas: np.ndarray,
                      warnings: List[str],
                      max_warnings: int = 3):
        """Draw warning messages"""
        
        if not warnings:
            return
        
        warn_x = self.width - 200
        warn_y = 10
        
        for i, warning in enumerate(warnings[:max_warnings]):
            # Truncate long warnings
            display_text = warning[:30] + "..." if len(warning) > 30 else warning
            
            # Determine color based on severity
            if "CRITICAL" in warning:
                color = (0, 0, 255)
            elif "ALERT" in warning or "WARNING" in warning:
                color = (0, 165, 255)
            else:
                color = (0, 255, 255)
            
            # Blinking effect for critical
            if "CRITICAL" in warning and self.animation_frame % 20 < 10:
                color = (100, 100, 255)
            
            cv2.putText(canvas, f"! {display_text}", (warn_x, warn_y + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    
    def create_visualization(self,
                            current_state: Dict,
                            predictions: Dict = None,
                            risk_report: RiskReport = None,
                            trails: Dict = None,
                            prediction_horizon: int = 60,
                            show_heatmap: bool = False,
                            show_arrows: bool = True) -> np.ndarray:
        """
        Create complete visualization frame.
        
        Args:
            current_state: Current object states from data collector
            predictions: Predicted trajectories
            risk_report: Risk analysis report
            trails: Historical trails for objects
            prediction_horizon: How far into predictions to visualize
            show_heatmap: Whether to show density heatmap
            show_arrows: Whether to show movement arrows
        
        Returns:
            Visualization frame
        """
        self.animation_frame += 1
        
        # Create base canvas
        canvas = self.create_base_canvas()
        
        # Draw density heatmap if enabled
        if show_heatmap and current_state:
            positions = [(s["x"], s["y"]) for s in current_state.values()]
            self.draw_density_heatmap(canvas, positions, alpha=0.2)
        
        # Draw predictions first (so current positions are on top)
        if predictions:
            self.draw_predictions(canvas, predictions, prediction_horizon)
        
        # Draw current positions
        if current_state:
            self.draw_current_positions(canvas, current_state, trails)
            
            # Draw movement arrows
            if show_arrows:
                self.draw_movement_arrows(canvas, current_state)
        
        # Draw risk information
        if risk_report:
            self.draw_risk_indicator(canvas, risk_report)
            self.draw_hotspots(canvas, risk_report.hotspots)
            self.draw_warnings(canvas, risk_report.warnings)
        
        # Draw prediction timeline
        if predictions:
            max_len = max(len(t) for t in predictions.values()) if predictions else 0
            self.draw_prediction_timeline(canvas, prediction_horizon, max_len)
        
        # Info text
        cv2.putText(canvas, f"Tracked: {len(current_state)}", 
                   (10, self.height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return canvas
    
    def create_animated_prediction(self,
                                   current_state: Dict,
                                   predictions: Dict,
                                   risk_report: RiskReport = None,
                                   total_frames: int = 60) -> List[np.ndarray]:
        """
        Create animation showing predictions over time.
        
        Returns:
            List of frames for animation
        """
        frames = []
        
        for horizon in range(0, total_frames, 2):  # Skip frames for smoother animation
            frame = self.create_visualization(
                current_state,
                predictions,
                risk_report,
                prediction_horizon=horizon
            )
            frames.append(frame)
        
        return frames


class CombinedDisplay:
    """
    Creates a combined display with multiple panels:
    - Main video with tracking
    - Dot matrix visualization
    - Prediction view
    - Risk dashboard
    """
    
    def __init__(self, 
                 main_width: int = 960,
                 main_height: int = 540,
                 panel_width: int = 400,
                 panel_height: int = 270):
        
        self.main_width = main_width
        self.main_height = main_height
        self.panel_width = panel_width
        self.panel_height = panel_height
        
        # Total display size
        self.total_width = main_width + panel_width
        self.total_height = main_height + panel_height
        
        # Prediction visualizer
        self.pred_vis = PredictionVisualizer(
            width=panel_width,
            height=panel_height
        )
    
    def create_combined_display(self,
                                main_frame: np.ndarray,
                                current_state: Dict,
                                predictions: Dict = None,
                                risk_report: RiskReport = None,
                                trails: Dict = None) -> np.ndarray:
        """Create combined multi-panel display"""
        
        # Create output canvas
        display = np.zeros((self.total_height, self.total_width, 3), dtype=np.uint8)
        display[:] = (20, 20, 30)
        
        # Resize and place main video
        main_resized = cv2.resize(main_frame, (self.main_width, self.main_height))
        display[0:self.main_height, 0:self.main_width] = main_resized
        
        # Create prediction visualization
        pred_frame = self.pred_vis.create_visualization(
            current_state,
            predictions,
            risk_report,
            trails,
            show_heatmap=True
        )
        
        # Place prediction panel
        display[0:self.panel_height, self.main_width:self.total_width] = pred_frame
        
        # Create risk dashboard panel
        dashboard = self._create_risk_dashboard(risk_report)
        display[self.panel_height:self.panel_height*2, 
                self.main_width:self.total_width] = dashboard
        
        # Create stats panel
        stats = self._create_stats_panel(current_state, predictions)
        display[self.main_height:self.total_height, 0:self.main_width] = stats
        
        return display
    
    def _create_risk_dashboard(self, risk_report: RiskReport = None) -> np.ndarray:
        """Create risk dashboard panel"""
        panel = np.zeros((self.panel_height, self.panel_width, 3), dtype=np.uint8)
        panel[:] = (25, 25, 35)
        
        # Title
        cv2.putText(panel, "RISK DASHBOARD", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        if risk_report is None:
            cv2.putText(panel, "No data", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            return panel
        
        # Risk level indicator
        level_color = get_risk_color(risk_report.risk_level)
        cv2.rectangle(panel, (10, 40), (self.panel_width - 10, 80), level_color, 2)
        cv2.putText(panel, risk_report.risk_level.value.upper(), 
                   (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, level_color, 2)
        
        # Risk gauges
        gauge_y = 100
        self._draw_gauge(panel, "Density", risk_report.density_risk, 10, gauge_y)
        self._draw_gauge(panel, "Velocity", risk_report.velocity_risk, 10, gauge_y + 35)
        self._draw_gauge(panel, "Convergence", risk_report.convergence_risk, 10, gauge_y + 70)
        self._draw_gauge(panel, "Stampede", risk_report.stampede_probability, 10, gauge_y + 105)
        
        # Recommendations
        if risk_report.recommendations:
            cv2.putText(panel, "Recommendations:", (10, gauge_y + 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            for i, rec in enumerate(risk_report.recommendations[:2]):
                text = rec[:45] + "..." if len(rec) > 45 else rec
                cv2.putText(panel, f"- {text}", (15, gauge_y + 155 + i * 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
        
        return panel
    
    def _draw_gauge(self, panel: np.ndarray, label: str, 
                    value: float, x: int, y: int):
        """Draw a horizontal gauge"""
        gauge_width = 150
        gauge_height = 12
        
        # Label
        cv2.putText(panel, label, (x, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # Background
        cv2.rectangle(panel, (x + 80, y), (x + 80 + gauge_width, y + gauge_height),
                     (50, 50, 60), -1)
        
        # Fill
        fill_width = int(value * gauge_width)
        color = get_risk_color_by_value(value)
        cv2.rectangle(panel, (x + 80, y), (x + 80 + fill_width, y + gauge_height),
                     color, -1)
        
        # Value text
        cv2.putText(panel, f"{value:.0%}", (x + 80 + gauge_width + 5, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
    def _create_stats_panel(self, current_state: Dict, 
                            predictions: Dict = None) -> np.ndarray:
        """Create statistics panel"""
        panel_height = self.total_height - self.main_height
        panel = np.zeros((panel_height, self.main_width, 3), dtype=np.uint8)
        panel[:] = (25, 25, 35)
        
        # Stats
        num_tracked = len(current_state) if current_state else 0
        
        cv2.putText(panel, f"Tracked Objects: {num_tracked}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if predictions:
            pred_horizon = max(len(t) for t in predictions.values()) / 30
            cv2.putText(panel, f"Prediction Horizon: {pred_horizon:.1f}s", (250, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Average speed
        if current_state:
            speeds = [np.sqrt(s.get("vx", 0)**2 + s.get("vy", 0)**2) 
                     for s in current_state.values()]
            avg_speed = np.mean(speeds) if speeds else 0
            cv2.putText(panel, f"Avg Speed: {avg_speed:.1f} px/s", (500, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return panel
