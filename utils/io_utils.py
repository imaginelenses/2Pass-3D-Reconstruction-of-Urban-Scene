#!/usr/bin/env python3
"""
I/O utilities for saving and loading 3D data
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import struct


def save_ply(
    filepath: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    ascii_format: bool = False
):
    """
    Save point cloud to PLY file
    
    Args:
        filepath: Output file path
        points: [N, 3] point coordinates
        colors: [N, 3] RGB colors (0-255) or None
        normals: [N, 3] normal vectors or None
        ascii_format: Save as ASCII instead of binary
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    n_points = len(points)
    
    # Build header
    header = "ply\n"
    header += f"format {'ascii' if ascii_format else 'binary_little_endian'} 1.0\n"
    header += f"element vertex {n_points}\n"
    header += "property float x\n"
    header += "property float y\n"
    header += "property float z\n"
    
    if normals is not None:
        header += "property float nx\n"
        header += "property float ny\n"
        header += "property float nz\n"
    
    if colors is not None:
        header += "property uchar red\n"
        header += "property uchar green\n"
        header += "property uchar blue\n"
    
    header += "end_header\n"
    
    if ascii_format:
        with open(filepath, 'w') as f:
            f.write(header)
            for i in range(n_points):
                line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
                
                if normals is not None:
                    line += f" {normals[i, 0]:.6f} {normals[i, 1]:.6f} {normals[i, 2]:.6f}"
                
                if colors is not None:
                    line += f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"
                
                f.write(line + "\n")
    else:
        with open(filepath, 'wb') as f:
            f.write(header.encode('ascii'))
            
            for i in range(n_points):
                # Position
                f.write(struct.pack('fff', *points[i]))
                
                # Normal
                if normals is not None:
                    f.write(struct.pack('fff', *normals[i]))
                
                # Color
                if colors is not None:
                    f.write(struct.pack('BBB', *colors[i].astype(np.uint8)))


def load_ply(filepath: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load point cloud from PLY file
    
    Args:
        filepath: Input file path
    
    Returns:
        points: [N, 3] point coordinates
        colors: [N, 3] RGB colors or None
        normals: [N, 3] normal vectors or None
    """
    from plyfile import PlyData
    
    plydata = PlyData.read(filepath)
    vertex = plydata['vertex']
    
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    
    # Try to load colors
    colors = None
    if 'red' in vertex.data.dtype.names:
        colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T
    
    # Try to load normals
    normals = None
    if 'nx' in vertex.data.dtype.names:
        normals = np.vstack([vertex['nx'], vertex['ny'], vertex['nz']]).T
    
    return points, colors, normals


def save_cameras_colmap(
    filepath: str,
    cameras: Dict[str, Dict],
    image_size: Tuple[int, int]
):
    """
    Save cameras in COLMAP format
    
    Args:
        filepath: Output file path
        cameras: Dictionary of camera parameters
        image_size: (width, height)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        # Write header
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        
        for cam_id, cam in cameras.items():
            # Assuming PINHOLE model
            K = cam['K']
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            w, h = image_size
            f.write(f"{cam_id} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")


def save_images_colmap(
    filepath: str,
    cameras: Dict[str, Dict],
    image_names: Dict[str, str]
):
    """
    Save images (camera poses) in COLMAP format
    
    Args:
        filepath: Output file path
        cameras: Dictionary of camera parameters  
        image_names: Dictionary mapping camera IDs to image filenames
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        # Write header
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        from utils.geometry_utils import matrix_to_quaternion
        
        for img_id, (cam_id, cam) in enumerate(cameras.items(), start=1):
            R = cam['R']
            t = cam['t']
            
            # Convert rotation to quaternion
            q = matrix_to_quaternion(R)
            qw, qx, qy, qz = q
            
            name = image_names.get(cam_id, f"{cam_id}.png")
            
            f.write(f"{img_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {cam_id} {name}\n")
            f.write("\n")  # Empty POINTS2D line


def save_json(filepath: str, data: Dict):
    """Save dictionary to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict:
    """Load dictionary from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_npz(filepath: str, **arrays):
    """Save multiple arrays to compressed NPZ file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(filepath, **arrays)


def load_npz(filepath: str) -> Dict[str, np.ndarray]:
    """Load arrays from NPZ file"""
    data = np.load(filepath, allow_pickle=True)
    return {key: data[key] for key in data.keys()}
