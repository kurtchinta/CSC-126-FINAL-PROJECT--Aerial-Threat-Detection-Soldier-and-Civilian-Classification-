"""
Real-time Stream Detection System
Process live video streams (webcam, drone feed, RTSP)
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from datetime import datetime
import time
from collections import deque


PROJECT_ROOT = Path(__file__).parent.parent.parent
BEST_PT_PATH = PROJECT_ROOT / 'backend' / 'civilian_soldier_working' / 'runs' / 'train' / 'custom_aerial_detection' / 'weights' / 'best.pt'
YOLOV11_PATH = PROJECT_ROOT / 'backend' / 'civilian_soldier_working' / 'yolo11n.pt'
OUTPUT_DIR = PROJECT_ROOT / "output" / "streams"


class StreamDetector:
    """Real-time stream detection with YOLO"""
    
    def __init__(self, model_path: Path = None, conf_threshold: float = 0.50, 
                 iou_threshold: float = 0.45, buffer_size: int = 30):
        """
        Initialize stream detector
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            buffer_size: Size of FPS calculation buffer
        """
        # Use provided model path or default to best.pt
        if model_path and Path(model_path).exists():
            self.model_path = Path(model_path)
        elif BEST_PT_PATH.exists():
            self.model_path = BEST_PT_PATH
        elif YOLOV11_PATH.exists():
            self.model_path = YOLOV11_PATH
        else:
            raise FileNotFoundError("No model weights found!")
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        
        # Class configuration (based on your dataset)
        self.class_names = {
            0: 'Civilian',
            1: 'Soldier'
        }
        
        self.class_colors = {
            0: (0, 255, 0),      # Green for civilians
            1: (0, 0, 255)       # Red for soldiers
        }
        
        # Performance tracking
        self.frame_count = 0
        self.fps_buffer = deque(maxlen=buffer_size)
        self.detection_history = deque(maxlen=100)
        
        # Cumulative detection counts
        self.total_detections = {0: 0, 1: 0}
        
        # Recording
        self.is_recording = False
        self.video_writer = None
    
    def load_model(self):
        """Load YOLO model"""
        print(f"Loading model: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        print("✓ Model loaded successfully")
    
    def draw_detections(self, frame: np.ndarray, results) -> tuple:
        """Draw bounding boxes and collect detection data"""
        annotated_frame = frame.copy()
        current_detections = {
            0: 0,  # Civilian
            1: 0   # Soldier
        }
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Apply higher confidence threshold for civilians (class 0)
                if class_id == 0 and confidence < 0.65:  # Civilian needs 65% confidence
                    continue
                
                # Count detection (both current and cumulative)
                current_detections[class_id] += 1
                self.total_detections[class_id] += 1
                
                # Get color and label
                color = self.class_colors.get(class_id, (255, 255, 255))
                label = f"{self.class_names.get(class_id, 'Unknown')}: {confidence:.2f}"
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_height - 10),
                    (x1 + label_width, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        
        return annotated_frame, current_detections
    
    def draw_ui(self, frame: np.ndarray, fps: float, detections: dict) -> np.ndarray:
        """Draw user interface overlay"""
        height, width = frame.shape[:2]
        
        # Top panel - Statistics
        overlay = frame.copy()
        panel_height = 180
        cv2.rectangle(overlay, (10, 10), (350, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Title
        cv2.putText(frame, "AERIAL SURVEILLANCE SYSTEM", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Current detections (show cumulative totals)
        y_offset = 115
        for class_id in [0, 1]:  # Civilian, Soldier
            color = self.class_colors.get(class_id, (255, 255, 255))
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            count = self.total_detections.get(class_id, 0)
            cv2.putText(frame, f"{class_name}: {count}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        # Total cumulative detections
        cv2.putText(frame, f"Total: {sum(self.total_detections.values())}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Recording indicator
        if self.is_recording:
            cv2.circle(frame, (width - 40, 40), 15, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (width - 80, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Controls help
        help_y = height - 80
        cv2.putText(frame, "Controls:", (10, help_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Q - Quit | R - Record | S - Screenshot", (10, help_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def start_recording(self, width: int, height: int, fps: int):
        """Start video recording"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = OUTPUT_DIR / f"stream_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        self.is_recording = True
        print(f"\n✓ Recording started: {output_path}")
    
    def stop_recording(self):
        """Stop video recording"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        print("\n✓ Recording stopped")
    
    def save_screenshot(self, frame: np.ndarray):
        """Save current frame as screenshot"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = OUTPUT_DIR / f"screenshot_{timestamp}.jpg"
        cv2.imwrite(str(screenshot_path), frame)
        print(f"\n✓ Screenshot saved: {screenshot_path}")
    
    def process_stream(self, source: str):
        """
        Process live video stream
        
        Args:
            source: Video source (0 for webcam, RTSP URL, etc.)
        """
        print("\n" + "="*60)
        print("LIVE STREAM DETECTION")
        print("="*60)
        
        # Parse source
        if source.isdigit():
            source = int(source)
            print(f"Source: Webcam {source}")
        else:
            print(f"Source: {source}")
        
        # Open stream
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open stream: {source}")
        
        # Get stream properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print("="*60)
        print("\nStream started. Press 'Q' to quit\n")
        
        # Reset statistics
        self.frame_count = 0
        self.fps_buffer.clear()
        self.detection_history.clear()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("\n⚠ Stream ended or connection lost")
                    break
                
                self.frame_count += 1
                start_time = time.time()
                
                # Run detection
                results = self.model(frame, conf=self.conf_threshold,
                                   iou=self.iou_threshold, verbose=False)[0]
                
                # Calculate FPS
                inference_time = time.time() - start_time
                current_fps = 1 / inference_time if inference_time > 0 else 0
                self.fps_buffer.append(current_fps)
                avg_fps = np.mean(self.fps_buffer)
                
                # Draw detections
                annotated_frame, detections = self.draw_detections(frame, results)
                self.detection_history.append(detections)
                
                # Draw UI
                annotated_frame = self.draw_ui(annotated_frame, avg_fps, detections)
                
                # Record if enabled
                if self.is_recording and self.video_writer:
                    self.video_writer.write(annotated_frame)
                
                # Print periodic updates
                if self.frame_count % 100 == 0:
                    print(f"Frame: {self.frame_count} | FPS: {avg_fps:.1f} | "
                          f"Detections - Soldiers: {detections[0]}, "
                          f"Civilians: {detections[1]}, Persons: {detections[2]}")
        
        except KeyboardInterrupt:
            print("\n\nStream interrupted by user")
        
        finally:
            # Cleanup
            if self.is_recording:
                self.stop_recording()
            cap.release()
        
        # Print summary
        print("\n" + "="*60)
        print("STREAM DETECTION COMPLETE")
        print("="*60)
        print(f"Total Frames: {self.frame_count}")
        print(f"Average FPS: {np.mean(self.fps_buffer):.2f}")
        print("="*60 + "\n")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Real-time stream detection')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam, RTSP URL, etc.)')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to model weights (defaults to best.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU threshold for NMS')
    
    args = parser.parse_args()
    
    # Determine model path
    model_path = Path(args.weights) if args.weights else None
    
    # Initialize detector
    detector = StreamDetector(
        model_path=model_path,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Load model
    detector.load_model()
    
    # Process stream
    detector.process_stream(source=args.source)


if __name__ == "__main__":
    main()
