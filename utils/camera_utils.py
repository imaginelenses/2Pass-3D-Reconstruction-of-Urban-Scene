#!/usr/bin/env python3
"""
Camera utilities for multi-view reconstruction
"""

import numpy as np
import torch
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class CameraParams:
    """Camera parameters container"""
    K: np.ndarray  # Intrinsic matrix [3, 3]
    R: np.ndarray  # Rotation matrix [3, 3]
    t: np.ndarray  # Translation vector [3]
    width: int
    height: int
    camera_id: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'K': self.K.tolist(),
            'R': self.R.tolist(),
            't': self.t.tolist(),
            'width': self.width,
            'height': self.height,
            'camera_id': self.camera_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Load from dictionary"""
        return cls(
            K=np.array(data['K']),
            R=np.array(data['R']),
            t=np.array(data['t']),
            width=data['width'],
            height=data['height'],
            camera_id=data['camera_id']
        )
    
    def get_projection_matrix(self) -> np.ndarray:
        """Get projection matrix P = K[R|t]"""
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return self.K @ Rt
    
    def get_camera_center(self) -> np.ndarray:
        """Get camera center in world coordinates"""
        return -self.R.T @ self.t


class CameraSet:
    """Manage multiple cameras"""
    
    def __init__(self):
        self.cameras: Dict[str, CameraParams] = {}
    
    def add_camera(self, camera: CameraParams):
        """Add a camera to the set"""
        self.cameras[camera.camera_id] = camera
    
    def get_camera(self, camera_id: str) -> Optional[CameraParams]:
        """Get camera by ID"""
        return self.cameras.get(camera_id)
    
    def __len__(self) -> int:
        """Return number of cameras"""
        return len(self.cameras)
    
    def __iter__(self):
        """Iterate over camera values"""
        return iter(self.cameras.values())
    
    def save(self, output_path: str):
        """Save cameras to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            cam_id: cam.to_dict() 
            for cam_id, cam in self.cameras.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, input_path: str):
        """Load cameras from JSON file"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        camera_set = cls()
        for cam_id, cam_data in data.items():
            camera = CameraParams.from_dict(cam_data)
            camera_set.add_camera(camera)
        
        return camera_set
    
    def __len__(self) -> int:
        return len(self.cameras)
    
    def __iter__(self):
        return iter(self.cameras.values())


def project_points(
    points_3d: np.ndarray,
    camera: CameraParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to image plane
    
    Args:
        points_3d: [N, 3] world coordinates
        camera: Camera parameters
    
    Returns:
        points_2d: [N, 2] image coordinates
        depth: [N] depth values
    """
    # Transform to camera coordinates
    points_cam = (camera.R @ points_3d.T).T + camera.t
    
    # Get depth
    depth = points_cam[:, 2]
    
    # Project to image plane
    points_2d_h = (camera.K @ points_cam.T).T
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]
    
    return points_2d, depth


def unproject_points(
    points_2d: np.ndarray,
    depth: np.ndarray,
    camera: CameraParams
) -> np.ndarray:
    """
    Unproject 2D points with depth to 3D world coordinates
    
    Args:
        points_2d: [N, 2] image coordinates
        depth: [N] depth values
        camera: Camera parameters
    
    Returns:
        points_3d: [N, 3] world coordinates
    """
    # Convert to homogeneous coordinates
    points_2d_h = np.concatenate([
        points_2d,
        np.ones((len(points_2d), 1))
    ], axis=1)
    
    # Unproject to camera coordinates
    K_inv = np.linalg.inv(camera.K)
    points_cam = (K_inv @ points_2d_h.T).T * depth[:, None]
    
    # Transform to world coordinates
    R_inv = camera.R.T
    points_3d = (R_inv @ points_cam.T).T - R_inv @ camera.t
    
    return points_3d


def is_in_view(
    points_3d: np.ndarray,
    camera: CameraParams,
    margin: int = 10
) -> np.ndarray:
    """
    Check if 3D points are visible in camera view
    
    Args:
        points_3d: [N, 3] world coordinates
        camera: Camera parameters
        margin: Pixel margin from image border
    
    Returns:
        mask: [N] boolean mask of visible points
    """
    points_2d, depth = project_points(points_3d, camera)
    
    # Check depth
    depth_valid = depth > 0
    
    # Check bounds
    x_valid = (points_2d[:, 0] >= margin) & (points_2d[:, 0] < camera.width - margin)
    y_valid = (points_2d[:, 1] >= margin) & (points_2d[:, 1] < camera.height - margin)
    
    return depth_valid & x_valid & y_valid


def compute_camera_frustum(
    camera: CameraParams,
    depth_range: Tuple[float, float] = (0.1, 100.0)
) -> np.ndarray:
    """
    Compute camera frustum corners for visualization
    
    Args:
        camera: Camera parameters
        depth_range: (near, far) depth values
    
    Returns:
        corners: [8, 3] frustum corners in world coordinates
    """
    near, far = depth_range
    w, h = camera.width, camera.height
    
    # Image corners
    corners_2d = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)
    
    # Unproject at near and far
    near_corners = unproject_points(corners_2d, np.full(4, near), camera)
    far_corners = unproject_points(corners_2d, np.full(4, far), camera)
    
    # Combine
    corners = np.vstack([near_corners, far_corners])
    
    return corners
