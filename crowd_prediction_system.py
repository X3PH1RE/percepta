"""
Crowd Prediction System - Main Integration
Combines tracking, data collection, prediction, risk analysis, and visualization.
"""

import cv2
import numpy as np
import time
from typing import Dict, Optional
import os

# Import all modules
from crowd_tracker import CentroidTracker, PersonDetector, DotMatrixVisualizer
from crowd_data_collector import CrowdDataCollector, FeatureExtractor
from crowd_predictor import CrowdPredictor
from risk_analyzer import RiskAnalyzer, RiskLevel
from prediction_visualizer import PredictionVisualizer, CombinedDisplay


class CrowdPredictionSystem:
    """
    Complete system for crowd tracking, prediction, and risk analysis.
    
    Features:
    - Real-time person tracking from video
    - Movement data collection and feature extraction
    - LSTM-based future position prediction
    - Risk analysis and stampede detection
    - Multi-panel visualization
    """
    
    def __init__(self,
                 video_path: str,
                 model_path: str = None,
                 prediction_enabled: bool = True):
        """
        Initialize the crowd prediction system.
        
        Args:
            video_path: Path to input video
            model_path: Path to trained prediction model (optional)
            prediction_enabled: Whether to run predictions
        """
        self.video_path = video_path
        self.prediction_enabled = prediction_enabled
        
        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"Video: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        
        # Calculate scale factor
        self.scale_factor = np.sqrt(
            (self.frame_width * self.frame_height) / (1920 * 1080)
        )
        
        # Initialize components
        self._init_tracker()
        self._init_data_collector()
        self._init_predictor(model_path)
        self._init_risk_analyzer()
        self._init_visualizers()
        
        # State
        self.frame_count = 0
        self.start_time = time.time()
        self.paused = False
        self.show_predictions = True
        self.show_heatmap = False
        self.prediction_horizon = 60  # frames (~2 seconds)
        
        # Prediction cache (don't predict every frame)
        self.last_predictions = {}
        self.prediction_interval = 15  # Predict every 15 frames
        self.last_risk_report = None
    
    def _init_tracker(self):
        """Initialize tracking components"""
        self.detector = PersonDetector(
            method='background_subtraction',
            frame_width=self.frame_width,
            frame_height=self.frame_height
        )
        
        max_distance = int(80 * self.scale_factor)
        self.tracker = CentroidTracker(
            max_disappeared=30,
            max_distance=max_distance
        )
    
    def _init_data_collector(self):
        """Initialize data collection"""
        density_radius = 100 * self.scale_factor
        self.data_collector = CrowdDataCollector(
            max_objects=500,
            max_history_per_object=1000,
            time_window=30.0,
            density_radius=density_radius
        )
        
        self.feature_extractor = FeatureExtractor(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            density_radius=density_radius
        )
    
    def _init_predictor(self, model_path: str = None):
        """Initialize prediction model"""
        self.predictor = CrowdPredictor(
            model_type="lstm",
            input_features=6,
            seq_length=30,
            output_steps=60
        )
        self.predictor.set_frame_params(
            self.frame_width, 
            self.frame_height, 
            self.fps
        )
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.predictor.load_model(model_path)
        else:
            print("No trained model loaded - using simple extrapolation")
    
    def _init_risk_analyzer(self):
        """Initialize risk analyzer"""
        self.risk_analyzer = RiskAnalyzer(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            pixels_per_meter=50.0 * self.scale_factor
        )
    
    def _init_visualizers(self):
        """Initialize visualization components"""
        # Dot matrix visualizer
        dot_width = 500
        dot_height = int(dot_width * self.frame_height / self.frame_width)
        self.dot_visualizer = DotMatrixVisualizer(
            width=dot_width,
            height=dot_height
        )
        
        # Prediction visualizer
        self.pred_visualizer = PredictionVisualizer(
            width=dot_width,
            height=dot_height
        )
        self.pred_visualizer.set_source_dimensions(
            self.frame_width, 
            self.frame_height
        )
        
        # Display scaling for main video
        self.display_width = min(1280, self.frame_width)
        self.display_scale = self.display_width / self.frame_width
        self.display_height = int(self.frame_height * self.display_scale)
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame.
        
        Returns:
            Dictionary with all processed data
        """
        self.frame_count += 1
        frame_time = self.frame_count / self.fps
        
        # Detection
        rects = self.detector.detect(frame)
        
        # Tracking
        self.tracker.update(rects)
        
        # Data collection
        self.data_collector.update(self.tracker.objects, frame_time)
        
        # Get current state
        current_state = self.data_collector.get_current_state()
        
        # Run prediction periodically
        if (self.prediction_enabled and 
            self.frame_count % self.prediction_interval == 0 and
            len(current_state) > 0):
            
            self.last_predictions = self.predictor.predict_future(
                current_state,
                future_seconds=10.0,  # Predict 10 seconds ahead
                data_collector=self.data_collector
            )
        
        # Risk analysis
        self.last_risk_report = self.risk_analyzer.evaluate(
            current_state,
            self.last_predictions,
            frame_time
        )
        
        # Get trails for visualization
        trails = {}
        for obj_id, track in self.data_collector.tracks.items():
            if len(track.positions) > 0:
                trails[obj_id] = [(p.x, p.y) for p in list(track.positions)[-30:]]
        
        return {
            "frame": frame,
            "rects": rects,
            "current_state": current_state,
            "predictions": self.last_predictions,
            "risk_report": self.last_risk_report,
            "trails": trails
        }
    
    def draw_main_frame(self, frame: np.ndarray, data: Dict) -> np.ndarray:
        """Draw tracking overlays on main frame"""
        output = frame.copy()
        
        scale = self.frame_height / 1080
        colors = [
            (0, 255, 255), (255, 100, 100), (100, 255, 100),
            (255, 150, 50), (200, 100, 255), (100, 255, 255)
        ]
        
        # Draw detection boxes
        for (x, y, w, h) in data["rects"]:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 
                         max(1, int(2 * scale)))
        
        # Draw tracked objects
        for obj_id, state in data["current_state"].items():
            color = colors[obj_id % len(colors)]
            cx, cy = int(state["x"]), int(state["y"])
            
            # Crosshair
            size = int(25 * scale)
            thickness = max(2, int(3 * scale))
            cv2.line(output, (cx - size, cy), (cx + size, cy), color, thickness)
            cv2.line(output, (cx, cy - size), (cx, cy + size), color, thickness)
            
            # Circle
            cv2.circle(output, (cx, cy), int(40 * scale), color, thickness)
            
            # ID label
            font_scale = 1.0 * scale
            label = f"ID:{obj_id}"
            cv2.putText(output, label, (cx - 20, cy - int(50 * scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Status overlay
        risk_report = data["risk_report"]
        if risk_report:
            # Risk indicator
            risk_color = self._get_risk_color(risk_report.risk_level)
            cv2.rectangle(output, (10, 10), (250, 90), (0, 0, 0), -1)
            cv2.rectangle(output, (10, 10), (250, 90), risk_color, 2)
            
            cv2.putText(output, f"Risk: {risk_report.risk_level.value.upper()}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
            cv2.putText(output, f"Tracked: {len(data['current_state'])}", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(output, f"Frame: {self.frame_count}", 
                       (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Warnings
            if risk_report.warnings:
                for i, warning in enumerate(risk_report.warnings[:2]):
                    y_pos = 120 + i * 25
                    cv2.putText(output, f"! {warning[:50]}", (20, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return output
    
    def _get_risk_color(self, risk_level: RiskLevel) -> tuple:
        """Get BGR color for risk level"""
        colors = {
            RiskLevel.SAFE: (0, 255, 0),
            RiskLevel.CAUTION: (0, 255, 255),
            RiskLevel.WARNING: (0, 165, 255),
            RiskLevel.DANGER: (0, 0, 255),
            RiskLevel.CRITICAL: (0, 0, 139)
        }
        return colors.get(risk_level, (128, 128, 128))
    
    def run(self):
        """Main processing loop"""
        print("\n" + "="*50)
        print("CROWD PREDICTION SYSTEM")
        print("="*50)
        print("\nControls:")
        print("  Q - Quit")
        print("  P - Pause/Resume")
        print("  R - Reset tracker")
        print("  T - Toggle predictions")
        print("  H - Toggle heatmap")
        print("  +/- - Adjust prediction horizon")
        print("  S - Save collected data")
        print("="*50 + "\n")
        
        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Process frame
                data = self.process_frame(frame)
                
                # Draw main frame with overlays
                main_display = self.draw_main_frame(frame, data)
                main_display = cv2.resize(main_display, 
                                         (self.display_width, self.display_height))
                
                # Create dot matrix visualization
                dot_matrix = self.dot_visualizer.create_visualization(
                    self.tracker, frame.shape
                )
                
                # Create prediction visualization
                if self.show_predictions and data["predictions"]:
                    pred_display = self.pred_visualizer.create_visualization(
                        data["current_state"],
                        data["predictions"],
                        data["risk_report"],
                        data["trails"],
                        self.prediction_horizon,
                        show_heatmap=self.show_heatmap
                    )
                else:
                    pred_display = self.pred_visualizer.create_visualization(
                        data["current_state"],
                        None,
                        data["risk_report"],
                        data["trails"],
                        show_heatmap=self.show_heatmap
                    )
            
            # Display windows
            cv2.imshow('Crowd Tracking - Main View', main_display)
            cv2.imshow('Dot Matrix', dot_matrix)
            cv2.imshow('Prediction & Risk', pred_display)
            
            # Handle input
            key = cv2.waitKey(30 if not self.paused else 100) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('p') or key == ord('P'):
                self.paused = not self.paused
                print("Paused" if self.paused else "Resumed")
            elif key == ord('r') or key == ord('R'):
                max_distance = int(80 * self.scale_factor)
                self.tracker = CentroidTracker(max_disappeared=30, 
                                              max_distance=max_distance)
                print("Tracker reset")
            elif key == ord('t') or key == ord('T'):
                self.show_predictions = not self.show_predictions
                print(f"Predictions: {'ON' if self.show_predictions else 'OFF'}")
            elif key == ord('h') or key == ord('H'):
                self.show_heatmap = not self.show_heatmap
                print(f"Heatmap: {'ON' if self.show_heatmap else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                self.prediction_horizon = min(300, self.prediction_horizon + 30)
                print(f"Prediction horizon: {self.prediction_horizon / self.fps:.1f}s")
            elif key == ord('-') or key == ord('_'):
                self.prediction_horizon = max(30, self.prediction_horizon - 30)
                print(f"Prediction horizon: {self.prediction_horizon / self.fps:.1f}s")
            elif key == ord('s') or key == ord('S'):
                self.data_collector.save_data("crowd_tracking_data.json")
                print("Data saved to crowd_tracking_data.json")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print session summary"""
        runtime = time.time() - self.start_time
        
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Frames processed: {self.frame_count}")
        print(f"Average FPS: {self.frame_count / runtime:.1f}")
        print(f"Total objects tracked: {self.data_collector.global_stats['total_objects_tracked']}")
        
        risk_summary = self.risk_analyzer.get_summary()
        print(f"\nRisk Statistics:")
        print(f"  Current risk: {risk_summary['current_risk']:.1%}")
        print(f"  Average risk: {risk_summary['avg_risk_1min']:.1%}")
        print(f"  Max risk: {risk_summary['max_risk_1min']:.1%}")
        print(f"  Trend: {risk_summary['trend']}")
        print("="*50)


def train_model_from_synthetic():
    """Train a prediction model using synthetic data"""
    from synthetic_crowd_generator import SyntheticCrowdGenerator, prepare_pytorch_data
    
    print("="*50, flush=True)
    print("TRAINING PREDICTION MODEL", flush=True)
    print("="*50, flush=True)
    
    print("\n1. Generating synthetic training data...", flush=True)
    print("   (This may take 1-2 minutes)", flush=True)
    generator = SyntheticCrowdGenerator(width=1920, height=1080)
    
    # Reduced parameters for faster training
    dataset = generator.generate_training_data(
        num_simulations=5,      # Reduced from 20
        agents_per_sim=20,      # Reduced from 50
        duration=20.0           # Reduced from 60
    )
    
    print("\n2. Preparing data for training...", flush=True)
    X, Y, labels = prepare_pytorch_data(dataset, seq_length=30, future_steps=30)
    
    if len(X) == 0:
        print("   ERROR: No training data generated!")
        return None
    
    print(f"   Training samples: {len(X)}")
    print(f"   Input shape: {X.shape}")
    print(f"   Output shape: {Y.shape}")
    
    print("\n3. Training model...")
    predictor = CrowdPredictor(
        model_type="lstm",
        input_features=6,
        seq_length=30,
        output_steps=30         # Reduced from 60
    )
    
    history = predictor.train(
        X, Y,
        epochs=20,              # Reduced from 30
        batch_size=32,
        learning_rate=0.001
    )
    
    print("\n4. Saving model...")
    predictor.save_model("crowd_predictor_model.pth")
    
    print("\n" + "="*50)
    print("Training complete!")
    print("Model saved to: crowd_predictor_model.pth")
    print("="*50)
    
    return predictor


if __name__ == "__main__":
    import sys
    
    # Check for training mode
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_model_from_synthetic()
    else:
        # ============================================
        # CONFIGURATION
        # ============================================
        VIDEO_PATH = r"crowd.mp4"  # Change to your video path
        MODEL_PATH = "crowd_predictor_model.pth"  # Trained model (optional)
        
        # Check if model exists
        model_to_use = MODEL_PATH if os.path.exists(MODEL_PATH) else None
        
        # Run system
        try:
            system = CrowdPredictionSystem(
                video_path=VIDEO_PATH,
                model_path=model_to_use,
                prediction_enabled=True
            )
            system.run()
        except Exception as e:
            print(f"Error: {e}")
            print("\nUsage:")
            print("  python crowd_prediction_system.py          - Run tracking system")
            print("  python crowd_prediction_system.py --train  - Train prediction model")
