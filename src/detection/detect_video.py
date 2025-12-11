"""
Real-time Video Detection System
Process video files with trained YOLO model
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


PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output" / "videos"


class VideoDetector:
    """Real-time video detection with YOLO"""
    
    def __init__(self, model_path: Path, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize detector
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        
        # Class names and colors (based on your dataset)
        self.class_names = {
            0: 'Civilian',
            1: 'Soldier'
        }
        
        self.class_colors = {
            0: (0, 255, 0),      # Green for civilians
            1: (0, 0, 255)       # Red for soldiers
        }
        
        # Statistics
        self.frame_count = 0
        self.total_detections = {
            0: 0,  # Civilian
            1: 0   # Soldier
        }
        self.fps_history = []
    
    def load_model(self):
        """Load YOLO model"""
        print(f"Loading model: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        print("✓ Model loaded successfully")
    
    def draw_detections(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            results: YOLO detection results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
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
                
                # Update statistics
                self.total_detections[class_id] = self.total_detections.get(class_id, 0) + 1
                
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
        
        return annotated_frame
    
    def draw_stats(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw statistics overlay on frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        stats_height = 150
        cv2.rectangle(overlay, (10, 10), (300, stats_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw statistics
        y_offset = 35
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 30
        for class_id, count in self.total_detections.items():
            color = self.class_colors.get(class_id, (255, 255, 255))
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            cv2.putText(frame, f"{class_name}: {count}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        return frame
    
    def process_video(self, video_path: str, output_path: str = None, display: bool = True):
        """
        Process video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            display: Whether to display video during processing
        """
        print("\n" + "="*60)
        print("PROCESSING VIDEO")
        print("="*60)
        print(f"Input: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total Frames: {total_frames}")
        
        # Setup output video
        if output_path:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output: {output_path}")
        else:
            out = None
        
        print("="*60)
        print("\nProcessing... Press 'q' to quit\n")
        
        # Reset statistics
        self.frame_count = 0
        self.total_detections = {
            0: 0,  # Civilian
            1: 0   # Soldier
        }
        self.fps_history = []
        
        # Process frames
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                start_time = time.time()
                
                # Run detection
                results = self.model(frame, conf=self.conf_threshold, 
                                   iou=self.iou_threshold, verbose=False)[0]
                
                # Calculate FPS
                inference_time = time.time() - start_time
                current_fps = 1 / inference_time if inference_time > 0 else 0
                self.fps_history.append(current_fps)
                
                # Keep only last 30 FPS measurements
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                
                avg_fps = np.mean(self.fps_history)
                
                # Draw detections and stats
                annotated_frame = self.draw_detections(frame, results)
                annotated_frame = self.draw_stats(annotated_frame, avg_fps)
                
                # Write output
                if out:
                    out.write(annotated_frame)
                
                # Display
                if display:
                    cv2.imshow('Aerial Surveillance Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user")
                        break
                
                # Progress
                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | FPS: {avg_fps:.1f} | "
                          f"Civilians: {self.total_detections[0]} | "
                          f"Soldiers: {self.total_detections[1]}")
        
        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Frames Processed: {self.frame_count}")
        print(f"Average FPS: {np.mean(self.fps_history):.2f}")
        print(f"\nTotal Detections:")
        for class_id, count in self.total_detections.items():
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            print(f"  {class_name}: {count}")
        print("="*60 + "\n")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Real-time video detection')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--weights', type=str, default=str(MODELS_DIR / 'best.pt'),
                       help='Path to model weights')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output video')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU threshold for NMS')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    
    args = parser.parse_args()
    
    # Setup output path
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = str(OUTPUT_DIR / f"detection_{timestamp}.mp4")
    
    # Initialize detector
    detector = VideoDetector(
        model_path=Path(args.weights),
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Load model
    detector.load_model()
    
    # Process video
    detector.process_video(
        video_path=args.source,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
