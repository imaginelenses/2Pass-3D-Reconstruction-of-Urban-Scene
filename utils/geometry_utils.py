#!/usr/bin/env python3
"""
Geometry utilities for 3D reconstruction and visibility analysis
"""

import numpy as np
import torch
from typing import Tuple, Optional, List
from scipy.spatial.transform import Rotation


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector(s)"""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-8)


def compute_ray_direction(
    origin: np.ndarray,
    target: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Compute ray direction and distance from origin to target
    
    Args:
        origin: [3] or [N, 3] ray origin
        target: [3] or [N, 3] ray target
    
    Returns:
        direction: Normalized direction vector
        distance: Distance from origin to target
    """
    diff = target - origin
    distance = np.linalg.norm(diff, axis=-1)
    direction = normalize_vector(diff)
    return direction, distance


def ray_sphere_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float
) -> Tuple[bool, Optional[float]]:
    """
    Compute ray-sphere intersection
    
    Args:
        ray_origin: [3] ray origin
        ray_direction: [3] normalized ray direction
        sphere_center: [3] sphere center
        sphere_radius: sphere radius
    
    Returns:
        intersects: Whether ray intersects sphere
        t: Distance along ray to intersection (None if no intersection)
    """
    oc = ray_origin - sphere_center
    
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere_radius * sphere_radius
    
    discriminant = b*b - 4*a*c
    
    if discriminant < 0:
        return False, None
    
    t = (-b - np.sqrt(discriminant)) / (2.0 * a)
    
    if t < 0:
        t = (-b + np.sqrt(discriminant)) / (2.0 * a)
    
    if t < 0:
        return False, None
    
    return True, t


def ray_box_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Compute ray-AABB intersection using slab method
    
    Args:
        ray_origin: [3] ray origin
        ray_direction: [3] normalized ray direction
        box_min: [3] box minimum corner
        box_max: [3] box maximum corner
    
    Returns:
        intersects: Whether ray intersects box
        t_near: Near intersection distance
        t_far: Far intersection distance
    """
    inv_dir = 1.0 / (ray_direction + 1e-8)
    
    t0 = (box_min - ray_origin) * inv_dir
    t1 = (box_max - ray_origin) * inv_dir
    
    t_min = np.minimum(t0, t1)
    t_max = np.maximum(t0, t1)
    
    t_near = np.max(t_min)
    t_far = np.min(t_max)
    
    if t_near > t_far or t_far < 0:
        return False, None, None
    
    return True, t_near, t_far


def compute_bounding_box(points: np.ndarray, padding: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box for point cloud
    
    Args:
        points: [N, 3] point coordinates
        padding: Additional padding around box
    
    Returns:
        box_min: [3] minimum corner
        box_max: [3] maximum corner
    """
    box_min = points.min(axis=0) - padding
    box_max = points.max(axis=0) + padding
    return box_min, box_max


def point_to_line_distance(
    point: np.ndarray,
    line_point: np.ndarray,
    line_direction: np.ndarray
) -> float:
    """
    Compute perpendicular distance from point to line
    
    Args:
        point: [3] point coordinates
        line_point: [3] point on line
        line_direction: [3] line direction (normalized)
    
    Returns:
        distance: Perpendicular distance
    """
    v = point - line_point
    proj = np.dot(v, line_direction) * line_direction
    perp = v - proj
    return np.linalg.norm(perp)


def rodrigues_rotation(
    axis: np.ndarray,
    angle: float
) -> np.ndarray:
    """
    Compute rotation matrix using Rodrigues' formula
    
    Args:
        axis: [3] rotation axis (will be normalized)
        angle: Rotation angle in radians
    
    Returns:
        R: [3, 3] rotation matrix
    """
    axis = normalize_vector(axis.reshape(-1))
    
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix
    
    Args:
        q: [4] quaternion [w, x, y, z]
    
    Returns:
        R: [3, 3] rotation matrix
    """
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x,y,z,w]
    return rot.as_matrix()


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion
    
    Args:
        R: [3, 3] rotation matrix
    
    Returns:
        q: [4] quaternion [w, x, y, z]
    """
    rot = Rotation.from_matrix(R)
    q_scipy = rot.as_quat()  # [x, y, z, w]
    return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])


def compute_plane_from_points(
    points: np.ndarray,
    robust: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Fit plane to 3D points
    
    Args:
        points: [N, 3] point coordinates
        robust: Use RANSAC for robust fitting
    
    Returns:
        normal: [3] plane normal vector
        d: Plane offset (plane equation: normal · x = d)
    """
    if robust and len(points) > 10:
        # Simple RANSAC
        best_inliers = 0
        best_normal = None
        best_d = None
        
        for _ in range(100):
            # Sample 3 points
            idx = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[idx]
            
            # Compute normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal = normalize_vector(normal)
            
            d = np.dot(normal, p1)
            
            # Count inliers
            distances = np.abs(points @ normal - d)
            inliers = np.sum(distances < 0.1)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal
                best_d = d
        
        return best_normal, best_d
    else:
        # SVD-based fitting
        centroid = points.mean(axis=0)
        centered = points - centroid
        
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]
        d = np.dot(normal, centroid)
        
        return normal, d


def project_point_to_plane(
    point: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float
) -> np.ndarray:
    """
    Project point onto plane
    
    Args:
        point: [3] point coordinates
        plane_normal: [3] plane normal
        plane_d: Plane offset
    
    Returns:
        projected: [3] projected point
    """
    distance = np.dot(point, plane_normal) - plane_d
    projected = point - distance * plane_normal
    return projected


def compute_convex_hull_2d(points_2d: np.ndarray) -> np.ndarray:
    """
    Compute 2D convex hull using Graham scan
    
    Args:
        points_2d: [N, 2] points
    
    Returns:
        hull_indices: Indices of hull vertices
    """
    from scipy.spatial import ConvexHull
    
    if len(points_2d) < 3:
        return np.arange(len(points_2d))
    
    hull = ConvexHull(points_2d)
    return hull.vertices


def point_in_polygon_2d(
    point: np.ndarray,
    polygon: np.ndarray
) -> bool:
    """
    Check if 2D point is inside polygon using ray casting
    
    Args:
        point: [2] point coordinates
        polygon: [N, 2] polygon vertices
    
    Returns:
        inside: Whether point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside
