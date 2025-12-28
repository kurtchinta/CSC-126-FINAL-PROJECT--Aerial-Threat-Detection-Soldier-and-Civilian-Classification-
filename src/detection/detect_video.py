"""
Unified Real-time Detection System
Supports both video file and live stream (webcam, RTSP, etc.)
Works with Electron app via Flask streaming
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
from threading import Thread, Lock, Event
from flask import Flask, Response, jsonify
import json


# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
BEST_PT_PATH = PROJECT_ROOT / 'backend' / 'civilian_soldier_working' / 'runs' / 'train' / 'custom_aerial_detection' / 'weights' / 'best.pt'
YOLOV11_PATH = PROJECT_ROOT / 'backend' / 'civilian_soldier_working' / 'yolo11n.pt'
OUTPUT_VIDEO_DIR = PROJECT_ROOT / 'output' / 'videos'

# Flask app for streaming
flask_app = Flask(__name__)

# Global state for streaming
class StreamState:
    def __init__(self):
        self.original_frame = None
        self.detected_frame = None
        self.stats = {
            'fps': 0,
            'frame_count': 0,
            'civilians': 0,
            'soldiers': 0,
            'total': 0,
            'progress': 0,
            'status': 'idle'
        }
        self.lock = Lock()
        self.running = False
        self.processing_complete = Event()
        self.video_fps = 30  # Target FPS for streaming

stream_state = StreamState()


class UnifiedDetector:
    """Unified detector for video and stream with YOLO"""
    
    def __init__(self, model_path: Path = None, conf_threshold: float = 0.25, 
                 iou_threshold: float = 0.45, buffer_size: int = 30):
        """
        Initialize detector
        
        Args:
            model_path: Path to YOLO model weights (defaults to best.pt)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            buffer_size: FPS buffer size
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
        self.buffer_size = buffer_size
        
        # Class configuration
        self.class_names = {0: 'Civilian', 1: 'Soldier'}
        self.class_colors = {0: (0, 255, 0), 1: (0, 0, 255)}  # Green for civilian, Red for soldier
        
        # Statistics
        self.frame_count = 0
        self.total_detections = {0: 0, 1: 0}
        self.fps_history = []
        self.fps_buffer = deque(maxlen=buffer_size)
        
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
        """Draw bounding boxes and collect detection data, labeling each detection with a unique number."""
        annotated_frame = frame.copy()
        current_detections = {0: 0, 1: 0}
        # Track instance numbers for each class
        instance_counters = {0: 1, 1: 1}
        
        if results.boxes is not None and len(results.boxes) > 0:
            # Sort boxes by y1 (top to bottom) for consistent numbering
            sorted_boxes = sorted(results.boxes, key=lambda b: int(b.xyxy[0][1]))
            for box in sorted_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                # Apply higher confidence threshold for civilians
                if class_id == 0 and confidence < 0.65:
                    continue
                if confidence < self.conf_threshold:
                    continue
                # Update statistics
                self.total_detections[class_id] = self.total_detections.get(class_id, 0) + 1
                current_detections[class_id] += 1
                # Draw bounding box
                color = self.class_colors.get(class_id, (255, 255, 255))
                # Label with unique number per class
                label = f"{self.class_names.get(class_id, 'Unknown')} {instance_counters[class_id]}: {confidence:.2f}"
                instance_counters[class_id] += 1
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return annotated_frame, current_detections
    
    def draw_stats(self, frame: np.ndarray, fps: float, detections: dict = None) -> np.ndarray:
        """Draw statistics overlay on frame with cumulative counts"""
        height, width = frame.shape[:2]
        overlay = frame.copy()
        stats_height = 180
        cv2.rectangle(overlay, (10, 10), (350, stats_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Title
        cv2.putText(frame, "AERIAL SURVEILLANCE SYSTEM", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Cumulative detections (use total_detections for running totals)
        y_offset = 115
        for class_id in [0, 1]:  # Civilian, Soldier
            color = self.class_colors.get(class_id, (255, 255, 255))
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            count = self.total_detections.get(class_id, 0)
            cv2.putText(frame, f"{class_name}: {count}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        # Total detections
        cv2.putText(frame, f"Total: {sum(self.total_detections.values())}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_video(self, video_path: str, output_path: str = None, display: bool = True, electron_mode: bool = False):
        """
        Process video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            display: Whether to display video during processing
            electron_mode: Whether to stream to Electron app
        """
        global stream_state
        
        print("\n" + "="*60, flush=True)
        print("PROCESSING VIDEO", flush=True)
        print("="*60, flush=True)
        print(f"Input: {video_path}", flush=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Resolution: {width}x{height}", flush=True)
        print(f"FPS: {fps}", flush=True)
        print(f"Total Frames: {total_frames}", flush=True)
        
        # Setup output video
        out = None
        if output_path:
            OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output: {output_path}", flush=True)
        
        print("="*60, flush=True)
        print("\nProcessing... Press 'q' to quit\n", flush=True)
        
        # Reset statistics
        self.frame_count = 0
        self.total_detections = {0: 0, 1: 0}
        self.fps_history = []
        
        # Set streaming state
        stream_state.running = True
        stream_state.video_fps = fps
        stream_state.processing_complete.clear()
        with stream_state.lock:
            stream_state.stats['status'] = 'processing'
        
        # Calculate frame delay for real-time playback in electron mode
        frame_delay = 1.0 / fps if electron_mode else 0
        
        try:
            while cap.isOpened() and stream_state.running:
                frame_start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                start_time = time.time()
                
                # Run detection
                results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)[0]
                
                # Calculate FPS
                inference_time = time.time() - start_time
                current_fps = 1 / inference_time if inference_time > 0 else 0
                self.fps_history.append(current_fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = np.mean(self.fps_history)
                
                # Draw detections
                annotated_frame, detections = self.draw_detections(frame, results)
                annotated_frame = self.draw_stats(annotated_frame, avg_fps, detections)
                
                # Update stream state for Electron
                if electron_mode:
                    with stream_state.lock:
                        stream_state.original_frame = frame.copy()
                        stream_state.detected_frame = annotated_frame.copy()
                        stream_state.stats = {
                            'fps': round(avg_fps, 1),
                            'frame_count': self.frame_count,
                            'total_frames': total_frames,
                            'civilians': self.total_detections[0],
                            'soldiers': self.total_detections[1],
                            'total': sum(self.total_detections.values()),
                            'progress': round((self.frame_count / total_frames) * 100, 1) if total_frames > 0 else 0,
                            'status': 'processing'
                        }
                    
                    # Pace the output to match video FPS for smooth streaming
                    elapsed = time.time() - frame_start_time
                    if elapsed < frame_delay:
                        time.sleep(frame_delay - elapsed)
                
                # Write output
                if out:
                    out.write(annotated_frame)
                
                # Display if enabled (non-electron mode)
                if display and not electron_mode:
                    cv2.imshow('Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nProcessing interrupted by user", flush=True)
                        break
                
                # Progress updates
                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"Progress: {progress:.1f}% | FPS: {avg_fps:.1f} | Civilians: {self.total_detections[0]} | Soldiers: {self.total_detections[1]}", flush=True)
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user", flush=True)
        
        finally:
            cap.release()
            if out:
                out.release()
            if display and not electron_mode:
                cv2.destroyAllWindows()
            
            # Update final stats
            with stream_state.lock:
                stream_state.stats['status'] = 'complete'
                stream_state.stats['progress'] = 100
            
            # Signal processing complete but keep running for final frame display
            stream_state.processing_complete.set()
        
        # Print summary
        print("\n" + "="*60, flush=True)
        print("PROCESSING COMPLETE", flush=True)
        print("="*60, flush=True)
        print(f"Frames Processed: {self.frame_count}", flush=True)
        print(f"Average FPS: {np.mean(self.fps_history):.2f}" if self.fps_history else "Average FPS: N/A", flush=True)
        print(f"\nTotal Detections:", flush=True)
        for class_id, count in self.total_detections.items():
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            print(f"  {class_name}: {count}", flush=True)
        print(f"  Total: {sum(self.total_detections.values())}", flush=True)
        print("="*60 + "\n", flush=True)
    
    def process_stream(self, source: str, electron_mode: bool = False):
        """
        Process live video stream (webcam, RTSP, etc.)
        
        Args:
            source: Video source (0 for webcam, RTSP URL, etc.)
            electron_mode: Whether to stream to Electron app
        """
        global stream_state
        
        print("\n" + "="*60, flush=True)
        print("LIVE STREAM DETECTION", flush=True)
        print("="*60, flush=True)
        
        # Parse source
        if str(source).isdigit():
            source = int(source)
            print(f"Source: Webcam {source}", flush=True)
        else:
            print(f"Source: {source}", flush=True)
        
        # Open stream
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open stream: {source}")
        
        # Get stream properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"Resolution: {width}x{height}", flush=True)
        print(f"FPS: {fps}", flush=True)
        print("="*60, flush=True)
        print("\nStream started. Press 'Q' to quit\n", flush=True)
        
        # Reset statistics
        self.frame_count = 0
        self.fps_buffer.clear()
        self.total_detections = {0: 0, 1: 0}
        stream_state.running = True
        stream_state.video_fps = fps
        with stream_state.lock:
            stream_state.stats['status'] = 'streaming'
        
        try:
            while stream_state.running:
                ret, frame = cap.read()
                if not ret:
                    print("\n⚠ Stream ended or connection lost", flush=True)
                    break
                
                self.frame_count += 1
                start_time = time.time()
                
                # Run detection
                results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)[0]
                
                # Calculate FPS
                inference_time = time.time() - start_time
                current_fps = 1 / inference_time if inference_time > 0 else 0
                self.fps_buffer.append(current_fps)
                avg_fps = np.mean(self.fps_buffer)
                
                # Draw detections
                annotated_frame, detections = self.draw_detections(frame, results)
                annotated_frame = self.draw_stats(annotated_frame, avg_fps, detections)
                
                # Update stream state for Electron
                if electron_mode:
                    with stream_state.lock:
                        stream_state.original_frame = frame.copy()
                        stream_state.detected_frame = annotated_frame.copy()
                        stream_state.stats = {
                            'fps': round(avg_fps, 1),
                            'frame_count': self.frame_count,
                            'civilians': self.total_detections[0],
                            'soldiers': self.total_detections[1],
                            'total': sum(self.total_detections.values()),
                            'status': 'streaming'
                        }
                else:
                    # Display locally
                    cv2.imshow('Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStream interrupted by user", flush=True)
                        break
                
                # Periodic updates
                if self.frame_count % 100 == 0:
                    print(f"Frame: {self.frame_count} | FPS: {avg_fps:.1f} | Civilians: {detections[0]}, Soldiers: {detections[1]}", flush=True)
        
        except KeyboardInterrupt:
            print("\nStream interrupted by user", flush=True)
        
        finally:
            cap.release()
            if not electron_mode:
                cv2.destroyAllWindows()
            with stream_state.lock:
                stream_state.stats['status'] = 'stopped'
            stream_state.processing_complete.set()
        
        # Print summary
        print("\n" + "="*60, flush=True)
        print("STREAM DETECTION COMPLETE", flush=True)
        print("="*60, flush=True)
        print(f"Total Frames: {self.frame_count}", flush=True)
        print(f"Average FPS: {np.mean(self.fps_buffer):.2f}" if self.fps_buffer else "Average FPS: N/A", flush=True)
        print(f"\nTotal Detections:", flush=True)
        for class_id, count in self.total_detections.items():
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            print(f"  {class_name}: {count}", flush=True)
        print(f"  Total: {sum(self.total_detections.values())}", flush=True)
        print("="*60 + "\n", flush=True)


# Flask routes for Electron streaming
@flask_app.route('/video/original')
def video_original():
    """Stream original video frames"""
    def generate():
        last_frame = None
        while stream_state.running or not stream_state.processing_complete.is_set():
            with stream_state.lock:
                frame = stream_state.original_frame
            
            if frame is not None:
                # Only encode and send if we have a new frame
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                last_frame = id(frame)
            
            time.sleep(0.025)  # ~40 FPS max
        
        # Send final frame if available
        with stream_state.lock:
            if stream_state.original_frame is not None:
                _, buffer = cv2.imencode('.jpg', stream_state.original_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@flask_app.route('/video/detected')
def video_detected():
    """Stream detected video frames"""
    def generate():
        last_frame = None
        while stream_state.running or not stream_state.processing_complete.is_set():
            with stream_state.lock:
                frame = stream_state.detected_frame
            
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                last_frame = id(frame)
            
            time.sleep(0.025)  # ~40 FPS max
        
        # Send final frame if available
        with stream_state.lock:
            if stream_state.detected_frame is not None:
                _, buffer = cv2.imencode('.jpg', stream_state.detected_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@flask_app.route('/stats')
def get_stats():
    """Get current detection statistics"""
    with stream_state.lock:
        return jsonify(stream_state.stats)


@flask_app.route('/stop')
def stop_detection():
    """Stop detection"""
    stream_state.running = False
    return jsonify({'status': 'stopped'})


def run_flask_server():
    """Run Flask server in background"""
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)  # Reduce Flask logging noise
    flask_app.run(host='127.0.0.1', port=5000, threaded=True, use_reloader=False)


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Unified real-time detection (video/stream)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to input video file or stream (0 for webcam, RTSP URL, etc.)')
    parser.add_argument('--mode', type=str, choices=['video', 'stream'], default='video',
                       help='Detection mode: video or stream')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to model weights (defaults to best.pt)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output video (video mode only)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU threshold for NMS')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    parser.add_argument('--electron', action='store_true',
                       help='Enable Electron mode with Flask streaming')
    
    args = parser.parse_args()
    
    # Setup output path for video mode
    if args.output is None and args.mode == 'video':
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        args.output = str(OUTPUT_VIDEO_DIR / f"detection_{timestamp}.mp4")
    
    # Determine model path
    model_path = None
    if args.weights:
        model_path = Path(args.weights)
    
    # Initialize detector
    detector = UnifiedDetector(
        model_path=model_path,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Load model
    detector.load_model()
    
    # Start Flask server if electron mode
    if args.electron:
        print("Starting Flask server for Electron streaming...", flush=True)
        
        # Pre-set running state so Flask generators don't exit immediately
        stream_state.running = True
        
        flask_thread = Thread(target=run_flask_server, daemon=True)
        flask_thread.start()
        print("Flask server started on http://127.0.0.1:5000", flush=True)
        time.sleep(0.5)  # Give Flask time to start
        
        # Run processing in a separate thread for electron mode
        def run_detection():
            if args.mode == 'video':
                detector.process_video(
                    video_path=args.source,
                    output_path=args.output,
                    display=not args.no_display,
                    electron_mode=True
                )
            else:
                detector.process_stream(
                    source=args.source,
                    electron_mode=True
                )
            # Keep running briefly after processing to allow final frame display
            time.sleep(2)
            stream_state.running = False
        
        detection_thread = Thread(target=run_detection)
        detection_thread.start()
        
        # Wait for detection to complete
        detection_thread.join()
        
        # Give a moment for final stats to be fetched
        time.sleep(1)
    else:
        # Non-electron mode: run directly
        if args.mode == 'video':
            detector.process_video(
                video_path=args.source,
                output_path=args.output,
                display=not args.no_display,
                electron_mode=False
            )
        else:
            detector.process_stream(
                source=args.source,
                electron_mode=False
            )


if __name__ == "__main__":
    main()
