#!/usr/bin/env python3
"""
Dynamic Object Tracking

Tracks objects (vehicles, pedestrians) across all cameras and time using:
- YOLOv8 for detection
- ByteTrack for multi-object tracking
- SAM2 for instance segmentation
- Multi-camera association
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import torch

sys.path.append(str(Path(__file__).parent.parent))

from utils import setup_logger, load_config


@dataclass
class Detection:
    """Single object detection"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    mask: np.ndarray = None
    features: np.ndarray = None


@dataclass
class Track:
    """Tracked object across frames"""
    track_id: int
    camera_id: str
    detections: Dict[int, Detection] = field(default_factory=dict)  # frame_idx -> Detection
    class_name: str = ""
    start_frame: int = 0
    end_frame: int = 0
    
    def add_detection(self, frame_idx: int, detection: Detection):
        """Add detection to track"""
        self.detections[frame_idx] = detection
        if not self.class_name:
            self.class_name = detection.class_name
        self.start_frame = min(self.start_frame, frame_idx) if self.start_frame > 0 else frame_idx
        self.end_frame = max(self.end_frame, frame_idx)


class DynamicObjectTracker:
    """Track dynamic objects across cameras and time"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type != "cuda":
            raise RuntimeError("GPU required! No CUDA available.")
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._init_detector()
        self._init_tracker()
        self._init_segmentor()
        
        # Tracking state
        self.tracks_per_camera: Dict[str, Dict[int, Track]] = {}  # camera_id -> track_id -> Track
        self.global_track_id = 0
        
    def _init_detector(self):
        """Initialize YOLOv8 detector"""
        from ultralytics import YOLO
        
        model_name = self.config['pass2_dynamic']['tracking'].get('detector', 'yolov8x')
        self.logger.info(f"Loading YOLO detector: {model_name}")
        
        self.detector = YOLO(f"{model_name}.pt")
        self.detector.to(self.device)
        
        # Classes to track
        self.pedestrian_classes = self.config['pass2_dynamic']['tracking']['pedestrian_classes']
        self.vehicle_classes = self.config['pass2_dynamic']['tracking']['vehicle_classes']
        self.track_classes = self.pedestrian_classes + self.vehicle_classes
        
        self.logger.info(f"Tracking classes: {self.track_classes}")
        
    def _init_tracker(self):
        """Initialize ByteTrack"""
        sys.path.insert(0, str(Path(__file__).parent.parent / "ByteTrack"))
        
        from yolox.tracker.byte_tracker import BYTETracker
        from types import SimpleNamespace
        
        # ByteTrack args
        self.tracker_args = SimpleNamespace()
        self.tracker_args.track_thresh = self.config['pass2_dynamic']['tracking'].get('conf_threshold', 0.3)
        self.tracker_args.track_buffer = self.config['pass2_dynamic']['tracking'].get('track_buffer', 30)
        self.tracker_args.match_thresh = self.config['pass2_dynamic']['tracking'].get('iou_threshold', 0.5)
        self.tracker_args.mot20 = False
        
        self.trackers = {}  # One tracker per camera
        
        self.logger.info("ByteTrack initialized")
        
    def _init_segmentor(self):
        """Initialize SAM2 for instance segmentation"""
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            checkpoint = "sam2_hiera_large.pt"
            
            # Load from local checkpoint
            sam2_repo_path = Path(__file__).parent.parent / "sam2_repo"
            checkpoint_path = sam2_repo_path / "checkpoints" / checkpoint
            
            if checkpoint_path.exists():
                self.segmentor = SAM2ImagePredictor.from_pretrained(
                    str(sam2_repo_path),
                    cfg=str(model_cfg),
                    checkpoint=str(checkpoint_path),
                    device=self.device
                )
                self.logger.info(f"SAM2 segmentor initialized from {checkpoint_path}")
            else:
                # Fallback to huggingface
                self.segmentor = SAM2ImagePredictor.from_pretrained(
                    f"facebook/{checkpoint.replace('.pt', '')}",
                    device=self.device
                )
                self.logger.info("SAM2 segmentor initialized from huggingface")
            
        except Exception as e:
            self.logger.warning(f"SAM2 init failed: {e}. Segmentation disabled.")
            self.segmentor = None
    
    def track_camera(
        self,
        camera_id: str,
        video_path: Path,
        sample_rate: int = 1
    ) -> Dict[int, Track]:
        """
        Track objects in single camera video
        
        Args:
            camera_id: Camera identifier
            video_path: Path to video file
            sample_rate: Process every Nth frame
            
        Returns:
            Dictionary of track_id -> Track
        """
        self.logger.info(f"Tracking objects in {camera_id}: {video_path}")
        
        # Initialize ByteTrack for this camera
        from yolox.tracker.byte_tracker import BYTETracker
        tracker = BYTETracker(self.tracker_args, frame_rate=30)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return {}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"  Video: {fps} FPS, {total_frames} frames")
        
        tracks = {}
        frame_idx = 0
        processed = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue
            
            # Detect objects
            detections = self._detect_frame(frame)
            
            # Track objects with ByteTrack
            if detections:
                online_targets = tracker.update(
                    output_results=self._format_detections_for_bytetrack(detections),
                    img_info=frame.shape[:2],
                    img_size=frame.shape[:2]
                )
                
                # Update tracks
                for target in online_targets:
                    track_id = target.track_id
                    
                    # Get matching detection
                    bbox = target.tlbr  # [x1, y1, x2, y2]
                    detection = self._match_detection(bbox, detections)
                    
                    if detection:
                        # Segment if SAM2 available
                        if self.segmentor:
                            detection.mask = self._segment_object(frame, detection.bbox)
                        
                        # Add to track
                        if track_id not in tracks:
                            global_id = self._get_global_track_id()
                            tracks[track_id] = Track(
                                track_id=global_id,
                                camera_id=camera_id
                            )
                        
                        tracks[track_id].add_detection(frame_idx, detection)
            
            processed += 1
            if processed % 100 == 0:
                self.logger.info(f"  Processed {processed}/{total_frames//sample_rate} frames, {len(tracks)} tracks")
            
            frame_idx += 1
        
        cap.release()
        
        self.logger.info(f"✓ Camera {camera_id}: {len(tracks)} tracks found")
        
        # Store tracks for this camera
        self.tracks_per_camera[camera_id] = tracks
        
        return tracks
    
    def _detect_frame(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in frame using YOLO"""
        results = self.detector(frame, verbose=False, device=self.device)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                class_name = r.names[class_id]
                
                # Filter for tracked classes
                if class_name not in self.track_classes:
                    continue
                
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                
                detections.append(Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name
                ))
        
        return detections
    
    def _format_detections_for_bytetrack(self, detections: List[Detection]) -> np.ndarray:
        """Format detections for ByteTrack: [x1, y1, x2, y2, score]"""
        if not detections:
            return np.empty((0, 5))
        
        output = np.zeros((len(detections), 5))
        for i, det in enumerate(detections):
            output[i, :4] = det.bbox
            output[i, 4] = det.confidence
        
        return output
    
    def _match_detection(self, bbox: np.ndarray, detections: List[Detection]) -> Detection:
        """Match ByteTrack bbox to original detection"""
        best_iou = 0
        best_det = None
        
        for det in detections:
            iou = self._compute_iou(bbox, det.bbox)
            if iou > best_iou:
                best_iou = iou
                best_det = det
        
        return best_det
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def _segment_object(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Segment object using SAM2"""
        try:
            self.segmentor.set_image(frame)
            
            masks, scores, _ = self.segmentor.predict(
                box=bbox,
                multimask_output=False
            )
            
            return masks[0]  # Return best mask
            
        except Exception as e:
            self.logger.warning(f"Segmentation failed: {e}")
            return None
    
    def _get_global_track_id(self) -> int:
        """Get unique global track ID"""
        self.global_track_id += 1
        return self.global_track_id
    
    def save_tracks(self, output_dir: Path):
        """Save tracking results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import pickle
        
        # Save tracks per camera
        for camera_id, tracks in self.tracks_per_camera.items():
            output_file = output_dir / f"tracks_{camera_id}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(tracks, f)
            
            self.logger.info(f"Saved {len(tracks)} tracks for {camera_id} to {output_file}")
        
        # Save summary
        summary_file = output_dir / "tracking_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=== Tracking Summary ===\n\n")
            for camera_id, tracks in self.tracks_per_camera.items():
                f.write(f"{camera_id}: {len(tracks)} tracks\n")
                
                # Class distribution
                class_counts = {}
                for track in tracks.values():
                    class_counts[track.class_name] = class_counts.get(track.class_name, 0) + 1
                
                for cls, count in sorted(class_counts.items()):
                    f.write(f"  {cls}: {count}\n")
                f.write("\n")
        
        self.logger.info(f"Saved tracking summary to {summary_file}")


def main():
    """Main tracking entry point"""
    config = load_config()
    logger = setup_logger("DynamicTracking")
    
    logger.info("=== PASS 2: Dynamic Object Tracking ===\n")
    
    # Initialize tracker
    tracker = DynamicObjectTracker(config, logger)
    
    # Get video files
    video_dir = Path(config['data']['video_dir'])
    video_files = sorted(video_dir.glob("*.mp4"))
    
    if not video_files:
        logger.error(f"No videos found in {video_dir}")
        return
    
    logger.info(f"Found {len(video_files)} videos")
    
    # Track each camera
    for video_file in video_files:
        camera_id = video_file.stem
        tracker.track_camera(
            camera_id=camera_id,
            video_path=video_file,
            sample_rate=config['pass2_dynamic'].get('sample_rate', 5)
        )
    
    # Save results
    output_dir = Path(config['data']['output_dir']) / "pass2_dynamic"
    tracker.save_tracks(output_dir)
    
    logger.info("\n✓ Dynamic object tracking complete")


if __name__ == "__main__":
    main()
