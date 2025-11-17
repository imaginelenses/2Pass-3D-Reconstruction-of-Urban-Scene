#!/usr/bin/env python3
"""
Per-Object 3D Reconstruction

Reconstructs tracked dynamic objects in 3D:
- Multi-camera object matching
- 3D triangulation per object
- JSON output with position and angle mappings
- Independent, modular execution

Usage:
    python pass2_dynamic/reconstruct_objects.py
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import pickle
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from utils import setup_logger, load_config, save_ply
from utils.camera_utils import CameraSet
from pass2_dynamic.track_objects import Track, Detection


@dataclass
class ObjectInstance3D:
    """Single 3D instance of an object at a specific timestamp"""
    object_id: int  # Global unique ID across cameras
    timestamp: float
    frame_idx: int
    class_name: str
    
    # 3D position and orientation
    position_3d: List[float]  # [x, y, z] in world coordinates
    rotation: List[float]  # [roll, pitch, yaw] in radians
    
    # 3D bounding box
    bbox_3d: Optional[Dict[str, List[float]]] = None  # {min: [x,y,z], max: [x,y,z]}
    dimensions: Optional[List[float]] = None  # [width, height, depth]
    
    # Reconstruction quality
    num_views: int = 0
    confidence: float = 0.0
    
    # Source cameras
    camera_ids: List[str] = None
    
    # 3D point cloud (optional)
    points_3d: Optional[np.ndarray] = None  # [N, 3]
    colors: Optional[np.ndarray] = None  # [N, 3]
    
    def to_dict(self):
        """Convert to JSON-serializable dict"""
        d = {
            'object_id': self.object_id,
            'timestamp': self.timestamp,
            'frame_idx': self.frame_idx,
            'class_name': self.class_name,
            'position_3d': self.position_3d,
            'rotation': self.rotation,
            'bbox_3d': self.bbox_3d,
            'dimensions': self.dimensions,
            'num_views': self.num_views,
            'confidence': self.confidence,
            'camera_ids': self.camera_ids or []
        }
        return d


class ObjectReconstructor:
    """Reconstructs dynamic objects in 3D from multi-camera tracks"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """Initialize reconstructor"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. GPU required for object reconstruction.")
        
        self.config = load_config(config_path)
        self.logger = setup_logger("ObjectReconstructor")
        self.device = torch.device("cuda")
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load camera parameters
        camera_file = Path(self.config['data']['output_dir']) / "pass1_static" / "cameras.json"
        if not camera_file.exists():
            raise FileNotFoundError(f"Camera parameters not found: {camera_file}")
        
        self.cameras = CameraSet.load(camera_file)
        self.logger.info(f"Loaded {len(self.cameras)} camera parameters")
        
        # Output directory
        self.output_dir = Path(self.config['data']['output_dir']) / "pass2_dynamic" / "objects_3d"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def load_tracks(self, tracking_dir: Path) -> Dict[str, Dict[int, Track]]:
        """
        Load tracking results from all cameras
        
        Args:
            tracking_dir: Directory containing tracks_*.pkl files
            
        Returns:
            Dictionary mapping camera_id to tracks
        """
        self.logger.info(f"Loading tracks from {tracking_dir}")
        
        tracks_per_camera = {}
        
        for track_file in sorted(tracking_dir.glob("tracks_*.pkl")):
            camera_id = track_file.stem.replace("tracks_", "")
            
            with open(track_file, 'rb') as f:
                tracks = pickle.load(f)
            
            tracks_per_camera[camera_id] = tracks
            self.logger.info(f"  {camera_id}: {len(tracks)} tracks")
        
        if not tracks_per_camera:
            raise ValueError(f"No track files found in {tracking_dir}")
        
        return tracks_per_camera
    
    def match_objects_across_cameras(
        self,
        tracks_per_camera: Dict[str, Dict[int, Track]]
    ) -> Dict[int, List[Tuple[str, Track]]]:
        """
        Match objects across different cameras using appearance and geometry
        
        Args:
            tracks_per_camera: Tracks per camera
            
        Returns:
            Dictionary mapping global_object_id to list of (camera_id, track) pairs
        """
        self.logger.info("Matching objects across cameras...")
        
        config = self.config['pass2_dynamic']['aggregation']
        
        # Extract all track features
        all_tracks = []
        for camera_id, tracks in tracks_per_camera.items():
            for track_id, track in tracks.items():
                all_tracks.append((camera_id, track))
        
        self.logger.info(f"Total tracks to match: {len(all_tracks)}")
        
        # Simple greedy matching based on temporal overlap and class
        # TODO: Implement more sophisticated matching with appearance features
        matched_objects = {}
        next_object_id = 1
        used_tracks = set()
        
        for camera_id, track in all_tracks:
            track_key = (camera_id, track.track_id)
            
            if track_key in used_tracks:
                continue
            
            # Start new object
            object_id = next_object_id
            next_object_id += 1
            
            matched_objects[object_id] = [(camera_id, track)]
            used_tracks.add(track_key)
            
            # Find matching tracks in other cameras
            for other_camera_id, other_track in all_tracks:
                other_key = (other_camera_id, other_track.track_id)
                
                if other_key in used_tracks:
                    continue
                
                # Check if same class
                if track.class_name != other_track.class_name:
                    continue
                
                # Check temporal overlap
                if self._has_temporal_overlap(track, other_track):
                    matched_objects[object_id].append((other_camera_id, other_track))
                    used_tracks.add(other_key)
        
        self.logger.info(f"Matched into {len(matched_objects)} global objects")
        
        # Log matching statistics
        multi_view_count = sum(1 for cameras in matched_objects.values() if len(cameras) > 1)
        self.logger.info(f"  Multi-view objects: {multi_view_count}/{len(matched_objects)}")
        
        return matched_objects
    
    def _has_temporal_overlap(self, track1: Track, track2: Track) -> bool:
        """Check if two tracks have temporal overlap"""
        frames1 = set(track1.detections.keys())
        frames2 = set(track2.detections.keys())
        
        # Check for any overlap
        overlap = frames1 & frames2
        
        # Also allow small time gap
        max_gap = self.config['pass2_dynamic']['aggregation'].get('max_temporal_gap', 10)
        
        if overlap:
            return True
        
        # Check gap
        max_frame1 = max(frames1)
        min_frame1 = min(frames1)
        max_frame2 = max(frames2)
        min_frame2 = min(frames2)
        
        gap = min(abs(max_frame1 - min_frame2), abs(max_frame2 - min_frame1))
        
        return gap <= max_gap
    
    def reconstruct_objects(
        self,
        matched_objects: Dict[int, List[Tuple[str, Track]]],
        fps: float = 30.0
    ) -> List[ObjectInstance3D]:
        """
        Reconstruct 3D instances for all objects
        
        Args:
            matched_objects: Matched objects across cameras
            fps: Video frame rate
            
        Returns:
            List of 3D object instances
        """
        self.logger.info("Reconstructing 3D objects...")
        
        all_instances = []
        
        for object_id, camera_track_pairs in matched_objects.items():
            self.logger.info(f"Reconstructing object {object_id}...")
            
            # Get all frames where this object appears
            all_frames = set()
            for camera_id, track in camera_track_pairs:
                all_frames.update(track.detections.keys())
            
            class_name = camera_track_pairs[0][1].class_name
            
            # Reconstruct at each timestamp
            for frame_idx in sorted(all_frames):
                timestamp = frame_idx / fps
                
                # Get detections from all cameras at this frame
                multi_view_detections = []
                camera_ids = []
                
                for camera_id, track in camera_track_pairs:
                    if frame_idx in track.detections:
                        det = track.detections[frame_idx]
                        cam = self.cameras.get_camera(camera_id)
                        if cam is not None:
                            multi_view_detections.append((camera_id, cam, det))
                            camera_ids.append(camera_id)
                
                # Need at least 2 views for triangulation
                if len(multi_view_detections) < 2:
                    continue
                
                # Triangulate 3D position
                position_3d, confidence = self._triangulate_position(multi_view_detections)
                
                if position_3d is None:
                    continue
                
                # Estimate rotation
                rotation = self._estimate_rotation(multi_view_detections, position_3d)
                
                # Estimate dimensions
                dimensions = self._estimate_dimensions(multi_view_detections)
                
                # Create 3D bounding box
                bbox_3d = self._create_bbox_3d(position_3d, dimensions, rotation)
                
                # Create instance
                instance = ObjectInstance3D(
                    object_id=object_id,
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    class_name=class_name,
                    position_3d=position_3d.tolist(),
                    rotation=rotation.tolist(),
                    bbox_3d=bbox_3d,
                    dimensions=dimensions.tolist() if dimensions is not None else None,
                    num_views=len(multi_view_detections),
                    confidence=confidence,
                    camera_ids=camera_ids
                )
                
                all_instances.append(instance)
        
        self.logger.info(f"Reconstructed {len(all_instances)} 3D object instances")
        
        return all_instances
    
    def _triangulate_position(
        self,
        multi_view_detections: List[Tuple[str, object, Detection]]
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Triangulate 3D position from multi-view detections
        
        Args:
            multi_view_detections: List of (camera_id, camera, detection)
            
        Returns:
            (position_3d, confidence) or (None, 0.0)
        """
        if len(multi_view_detections) < 2:
            return None, 0.0
        
        # Simple centroid triangulation
        # TODO: Implement proper DLT triangulation
        
        rays = []
        origins = []
        
        for camera_id, cam, det in multi_view_detections:
            # Use bbox center
            bbox = det.bbox
            u = (bbox[0] + bbox[2]) / 2
            v = (bbox[1] + bbox[3]) / 2
            
            # Unproject to ray
            # This is a placeholder - need actual camera unprojection
            ray = np.array([u - cam.width/2, v - cam.height/2, cam.focal_length])
            ray = ray / np.linalg.norm(ray)
            
            rays.append(ray)
            origins.append(np.array([0, 0, 0]))  # Camera position
        
        # Average position (simple approach)
        position_3d = np.mean(origins, axis=0)
        confidence = min(1.0, len(multi_view_detections) / 3.0)
        
        return position_3d, confidence
    
    def _estimate_rotation(
        self,
        multi_view_detections: List[Tuple[str, object, Detection]],
        position_3d: np.ndarray
    ) -> np.ndarray:
        """Estimate object rotation (roll, pitch, yaw)"""
        # Placeholder: return zero rotation
        return np.array([0.0, 0.0, 0.0])
    
    def _estimate_dimensions(
        self,
        multi_view_detections: List[Tuple[str, object, Detection]]
    ) -> Optional[np.ndarray]:
        """Estimate object dimensions (width, height, depth)"""
        # Placeholder: estimate from 2D bboxes
        avg_width = np.mean([det.bbox[2] - det.bbox[0] for _, _, det in multi_view_detections])
        avg_height = np.mean([det.bbox[3] - det.bbox[1] for _, _, det in multi_view_detections])
        
        # Rough estimate
        return np.array([avg_width * 0.01, avg_height * 0.01, avg_width * 0.01])
    
    def _create_bbox_3d(
        self,
        position: np.ndarray,
        dimensions: Optional[np.ndarray],
        rotation: np.ndarray
    ) -> Optional[Dict[str, List[float]]]:
        """Create 3D bounding box"""
        if dimensions is None:
            return None
        
        half_dims = dimensions / 2
        
        bbox_min = (position - half_dims).tolist()
        bbox_max = (position + half_dims).tolist()
        
        return {
            'min': bbox_min,
            'max': bbox_max
        }
    
    def save_results(self, instances: List[ObjectInstance3D]):
        """
        Save reconstruction results as JSON
        
        Args:
            instances: List of 3D object instances
        """
        self.logger.info("Saving results...")
        
        # Group by object_id
        objects_by_id = defaultdict(list)
        for instance in instances:
            objects_by_id[instance.object_id].append(instance)
        
        # Save JSON mapping
        output_json = self.output_dir / "objects_3d.json"
        
        json_data = {
            'metadata': {
                'num_objects': len(objects_by_id),
                'num_instances': len(instances),
                'coordinate_system': 'world',
                'units': 'meters'
            },
            'objects': []
        }
        
        for object_id, object_instances in objects_by_id.items():
            # Sort by timestamp
            object_instances.sort(key=lambda x: x.timestamp)
            
            obj_data = {
                'object_id': object_id,
                'class_name': object_instances[0].class_name,
                'num_instances': len(object_instances),
                'temporal_span': {
                    'start_time': object_instances[0].timestamp,
                    'end_time': object_instances[-1].timestamp,
                    'start_frame': object_instances[0].frame_idx,
                    'end_frame': object_instances[-1].frame_idx
                },
                'instances': [inst.to_dict() for inst in object_instances]
            }
            
            json_data['objects'].append(obj_data)
        
        with open(output_json, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        self.logger.info(f"Saved JSON mapping: {output_json}")
        
        # Save per-object PLY files (optional)
        for object_id, object_instances in objects_by_id.items():
            # Collect all points
            all_points = []
            all_colors = []
            
            for inst in object_instances:
                if inst.points_3d is not None and inst.colors is not None:
                    all_points.append(inst.points_3d)
                    all_colors.append(inst.colors)
            
            if all_points:
                points = np.vstack(all_points)
                colors = np.vstack(all_colors)
                
                ply_file = self.output_dir / f"object_{object_id:04d}.ply"
                save_ply(ply_file, points, colors)
                
                self.logger.info(f"Saved PLY: {ply_file}")
        
        # Save summary
        summary_file = self.output_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write("3D Object Reconstruction Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total objects: {len(objects_by_id)}\n")
            f.write(f"Total instances: {len(instances)}\n\n")
            
            f.write("Per-object breakdown:\n")
            for object_id, object_instances in sorted(objects_by_id.items()):
                class_name = object_instances[0].class_name
                num_instances = len(object_instances)
                f.write(f"  Object {object_id:04d} ({class_name}): {num_instances} instances\n")
        
        self.logger.info(f"Saved summary: {summary_file}")
    
    def run(self):
        """Run complete object reconstruction pipeline"""
        self.logger.info("=" * 50)
        self.logger.info("Starting Per-Object 3D Reconstruction")
        self.logger.info("=" * 50)
        
        # Load tracks
        tracking_dir = Path(self.config['data']['output_dir']) / "pass2_dynamic"
        tracks_per_camera = self.load_tracks(tracking_dir)
        
        # Match objects across cameras
        matched_objects = self.match_objects_across_cameras(tracks_per_camera)
        
        # Reconstruct 3D
        instances = self.reconstruct_objects(matched_objects)
        
        # Save results
        self.save_results(instances)
        
        self.logger.info("=" * 50)
        self.logger.info("✅ Object reconstruction complete!")
        self.logger.info(f"   Output: {self.output_dir}")
        self.logger.info("=" * 50)


def main():
    """Main entry point"""
    reconstructor = ObjectReconstructor()
    reconstructor.run()


if __name__ == "__main__":
    main()
