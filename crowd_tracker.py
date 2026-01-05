import cv2
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist


class CentroidTracker:
    """Tracks objects using centroid-based tracking with ID persistence"""
    
    def __init__(self, max_disappeared=50, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        # Store previous positions for trail visualization
        self.trails = OrderedDict()
        self.max_trail_length = 30
    
    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.trails[self.next_object_id] = [centroid]
        self.next_object_id += 1
    
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.trails:
            del self.trails[object_id]
    
    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            input_centroids[i] = (cx, cy)
        
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                # Update trail
                self.trails[object_id].append(tuple(input_centroids[col]))
                if len(self.trails[object_id]) > self.max_trail_length:
                    self.trails[object_id].pop(0)
                
                used_rows.add(row)
                used_cols.add(col)
            
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
        
        return self.objects


class PersonDetector:
    """Detects people in frames using multiple detection methods"""
    
    def __init__(self, method='background_subtraction', frame_width=1920, frame_height=1080):
        self.method = method
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Calculate scale factor based on video resolution (relative to 1080p)
        self.scale_factor = (frame_width * frame_height) / (1920 * 1080)
        
        if method == 'hog':
            # HOG detector for pedestrians
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        elif method == 'background_subtraction':
            # Background subtractor - great for top-down views
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=25, detectShadows=True
            )
            # Scale kernel size based on resolution
            kernel_size = max(3, int(5 * np.sqrt(self.scale_factor)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Store the last mask for debug visualization
        self.last_mask = None
    
    def detect(self, frame):
        """Returns list of bounding boxes (x, y, w, h) for detected persons"""
        
        if self.method == 'hog':
            return self._detect_hog(frame)
        elif self.method == 'background_subtraction':
            return self._detect_background(frame)
        else:
            return self._detect_background(frame)
    
    def _detect_hog(self, frame):
        """HOG-based pedestrian detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, weights = self.hog.detectMultiScale(
            gray, winStride=(4, 4), padding=(8, 8), scale=1.05
        )
        rects = [(x, y, w, h) for (x, y, w, h) in boxes]
        return rects
    
    def _detect_background(self, frame):
        """Background subtraction for top-down crowd detection"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (shadows are marked as gray, 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)
        
        # Store mask for debug view
        self.last_mask = fg_mask.copy()
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Scale area thresholds based on video resolution
        min_area = int(300 * self.scale_factor)
        max_area = int(100000 * self.scale_factor)
        
        rects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area - scaled for video resolution
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by aspect ratio for top-down view (people appear roughly circular)
                aspect_ratio = w / float(h)
                if 0.2 < aspect_ratio < 5.0:
                    rects.append((x, y, w, h))
        
        return rects


class DotMatrixVisualizer:
    """Creates a dot matrix visualization of tracked positions"""
    
    def __init__(self, width=600, height=400, bg_color=(15, 15, 25)):
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.dot_radius = 4
        self.trail_radius = 1
        
        # Color palette for different people
        self.colors = [
            (0, 255, 255),    # Cyan
            (255, 100, 100),  # Light blue
            (100, 255, 100),  # Light green
            (255, 150, 50),   # Orange-blue
            (200, 100, 255),  # Pink
            (100, 255, 255),  # Yellow
            (255, 200, 100),  # Light cyan
            (150, 255, 150),  # Lime
            (255, 100, 200),  # Magenta
            (100, 200, 255),  # Gold
        ]
    
    def get_color(self, object_id):
        return self.colors[object_id % len(self.colors)]
    
    def create_visualization(self, tracker, frame_shape):
        """Create dot matrix visualization from tracker data"""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = self.bg_color
        
        # Draw grid
        grid_color = (40, 40, 50)
        for x in range(0, self.width, 40):
            cv2.line(canvas, (x, 0), (x, self.height), grid_color, 1)
        for y in range(0, self.height, 40):
            cv2.line(canvas, (0, y), (self.width, y), grid_color, 1)
        
        # Scale factors
        scale_x = self.width / frame_shape[1]
        scale_y = self.height / frame_shape[0]
        
        # Draw trails and dots
        for object_id, centroid in tracker.objects.items():
            color = self.get_color(object_id)
            
            # Draw trail
            if object_id in tracker.trails:
                trail = tracker.trails[object_id]
                for i in range(1, len(trail)):
                    pt1 = (int(trail[i-1][0] * scale_x), int(trail[i-1][1] * scale_y))
                    pt2 = (int(trail[i][0] * scale_x), int(trail[i][1] * scale_y))
                    # Fade effect based on position in trail
                    alpha = i / len(trail)
                    fade_color = tuple(int(c * alpha * 0.5) for c in color)
                    cv2.line(canvas, pt1, pt2, fade_color, 1)
            
            # Draw current position dot
            x = int(centroid[0] * scale_x)
            y = int(centroid[1] * scale_y)
            
            # Simple small dot
            cv2.circle(canvas, (x, y), self.dot_radius, color, -1)
            
            # Small ID label
            cv2.putText(canvas, f"{object_id}", (x + 6, y + 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        # Draw stats
        cv2.putText(canvas, f"TRACKED: {len(tracker.objects)}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(canvas, "DOT MATRIX VISUALIZATION", (10, self.height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        return canvas


def draw_detections(frame, tracker, detector_rects):
    """Draw bounding boxes and IDs on the main frame"""
    output = frame.copy()
    
    # Calculate scale for high-res videos
    frame_height = frame.shape[0]
    scale = frame_height / 1080  # Scale relative to 1080p
    
    # Color palette
    colors = [
        (0, 255, 255), (255, 100, 100), (100, 255, 100),
        (255, 150, 50), (200, 100, 255), (100, 255, 255),
        (255, 200, 100), (150, 255, 150), (255, 100, 200), (100, 200, 255)
    ]
    
    # Draw detection boxes
    for (x, y, w, h) in detector_rects:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), max(1, int(2 * scale)))
    
    # Draw tracked persons with IDs
    for object_id, centroid in tracker.objects.items():
        color = colors[object_id % len(colors)]
        cx, cy = centroid
        
        # Scale sizes for high-res
        crosshair_size = int(25 * scale)
        circle_radius = int(40 * scale)
        font_scale = 1.0 * scale
        thickness = max(2, int(3 * scale))
        
        # Draw crosshair at centroid
        cv2.line(output, (cx - crosshair_size, cy), (cx + crosshair_size, cy), color, thickness)
        cv2.line(output, (cx, cy - crosshair_size), (cx, cy + crosshair_size), color, thickness)
        
        # Draw circle around person
        cv2.circle(output, (cx, cy), circle_radius, color, thickness)
        
        # Draw ID tag
        label = f"ID:{object_id}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tag_y = cy - circle_radius - int(10 * scale)
        cv2.rectangle(output, (cx - text_w//2 - int(5*scale), tag_y - text_h - int(5*scale)), 
                     (cx + text_w//2 + int(5*scale), tag_y + int(5*scale)), color, -1)
        cv2.putText(output, label, (cx - text_w//2, tag_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    
    # Draw stats on frame
    cv2.putText(output, f"People Tracked: {len(tracker.objects)}", (10, int(40*scale)),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (0, 255, 255), max(2, int(3*scale)))
    
    return output


def main(video_path, detection_method='background_subtraction'):
    """Main function to run crowd tracking"""
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate scale factor for 4K and other high-res videos
    scale_factor = np.sqrt((frame_width * frame_height) / (1920 * 1080))
    
    print(f"Video: {frame_width}x{frame_height} @ {fps}fps")
    print(f"Detection method: {detection_method}")
    print(f"Scale factor: {scale_factor:.2f}x")
    print("\nControls:")
    print("  Q - Quit")
    print("  P - Pause/Resume")
    print("  R - Reset tracker")
    print("  D - Toggle debug mask view")
    
    # Initialize components with resolution-aware settings
    detector = PersonDetector(method=detection_method, frame_width=frame_width, frame_height=frame_height)
    
    # Scale tracking distance for high-res videos
    max_track_distance = int(80 * scale_factor)
    tracker = CentroidTracker(max_disappeared=30, max_distance=max_track_distance)
    
    # Scale dot matrix to match video aspect ratio
    dot_width = 600
    dot_height = int(dot_width * frame_height / frame_width)
    visualizer = DotMatrixVisualizer(width=dot_width, height=dot_height)
    
    paused = False
    frame_count = 0
    show_debug = True  # Show mask by default to help debug
    
    # For resizing display (4K is too big for most monitors)
    display_width = min(1280, frame_width)
    display_scale = display_width / frame_width
    display_height = int(frame_height * display_scale)
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
            
            # Detect people
            rects = detector.detect(frame)
            
            # Update tracker
            tracker.update(rects)
            
            # Draw on main frame
            output_frame = draw_detections(frame, tracker, rects)
            
            # Create dot matrix visualization
            dot_matrix = visualizer.create_visualization(tracker, frame.shape)
            
            # Add frame counter and detection count
            cv2.putText(output_frame, f"Frame: {frame_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
            cv2.putText(output_frame, f"Detections: {len(rects)}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
            
            # Resize for display
            output_display = cv2.resize(output_frame, (display_width, display_height))
            
            # Create debug mask view if available
            if show_debug and detector.last_mask is not None:
                mask_display = cv2.resize(detector.last_mask, (display_width // 2, display_height // 2))
                mask_colored = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
                # Add text
                cv2.putText(mask_colored, "Detection Mask (D to toggle)", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Display windows
        cv2.imshow('Crowd Tracking - Main View', output_display)
        cv2.imshow('Dot Matrix Visualization', dot_matrix)
        
        if show_debug and detector.last_mask is not None:
            cv2.imshow('Debug: Detection Mask', mask_colored)
        
        # Handle keyboard input
        key = cv2.waitKey(30 if not paused else 100) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('p') or key == ord('P'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r') or key == ord('R'):
            tracker = CentroidTracker(max_disappeared=30, max_distance=max_track_distance)
            print("Tracker reset")
        elif key == ord('d') or key == ord('D'):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow('Debug: Detection Mask')
            print(f"Debug view: {'ON' if show_debug else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nTracking ended.")


if __name__ == "__main__":
    VIDEO_PATH = r"crowd.mp4" 
    
    # Detection method: 'background_subtraction' (best for top-down) or 'hog'
    DETECTION_METHOD = 'background_subtraction'
    
    main(VIDEO_PATH, DETECTION_METHOD)

