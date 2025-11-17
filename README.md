# Traffic Intersection 3D Reconstruction

Modular pipeline for 3D reconstruction of traffic scenes with dynamic object tracking.

## 🎯 Overview

Multi-camera 3D reconstruction system with two independent phases:

1. **Static Scene Reconstruction**: DUSt3R-based point cloud generation
2. **Dynamic Object Tracking & 3D Reconstruction**: Per-object 3D reconstruction with JSON position/angle mapping

**Key Features:**
- Modular, independent scripts that run sequentially
- GPU-only processing (CUDA required)
- Multi-camera object matching
- JSON output for easy integration

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────┐
│   INPUT: Multi-Camera Videos    │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│   PHASE 1: Static Scene         │
│   - DUSt3R 3D reconstruction    │
│   - Camera pose estimation      │
│   - Output: Point cloud + JSON  │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│   PHASE 2A: Object Tracking     │
│   - YOLOv8 detection            │
│   - ByteTrack multi-object      │
│   - SAM2 segmentation           │
│   - Output: Tracks per camera   │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│   PHASE 2B: 3D Reconstruction   │
│   - Multi-camera matching       │
│   - 3D triangulation per object │
│   - Position & angle mapping    │
│   - Output: JSON + PLY files    │
└─────────────────────────────────┘
```

## 🚀 Installation

```bash
# Clone repository
cd /home/rsu/Project/3DReconstruction/static-scene

# Create conda environment
mamba create -n acv2 python=3.10 -y
mamba activate acv2

# Install dependencies (GPU required)
./install_dependencies.sh
```

**Requirements:**
- NVIDIA GPU with CUDA (RTX 3080+ recommended)
- 11+ GB VRAM
- Linux with CUDA drivers

## 📂 Data Preparation

Place your camera videos in `data/videos/`:
```
data/videos/
├── s1-left.mp4
├── s1-right.mp4
├── s2-left.mp4
├── s2-right.mp4
├── s3-left.mp4
├── s3-right.mp4
├── s4-left.mp4
└── s4-right.mp4
```

## 🎬 Usage - Modular Execution

Each script can be run **independently and sequentially**:

### Phase 1: Static Scene Reconstruction

```bash
python pass1_static/reconstruct_static_scene.py
```

**Output:**
- `outputs/pass1_static/dust3r_pointcloud.ply` - Static scene point cloud
- `outputs/pass1_static/cameras.json` - Camera parameters

### Phase 2A: Dynamic Object Tracking

```bash
python pass2_dynamic/track_objects.py
```

**Output:**
- `outputs/pass2_dynamic/tracks_<camera_id>.pkl` - Tracks per camera

### Phase 2B: 3D Object Reconstruction

```bash
python pass2_dynamic/reconstruct_objects.py
```

**Output:**
- `outputs/pass2_dynamic/objects_3d/objects_3d.json` - **Main JSON mapping**
- `outputs/pass2_dynamic/objects_3d/object_<id>.ply` - Per-object point clouds
- `outputs/pass2_dynamic/objects_3d/summary.txt` - Summary statistics

## 📊 JSON Output Format

The main output `objects_3d.json` contains position and angle mappings:

```json
{
  "metadata": {
    "num_objects": 25,
    "num_instances": 487,
    "coordinate_system": "world",
    "units": "meters"
  },
  "objects": [
    {
      "object_id": 1,
      "class_name": "person",
      "num_instances": 45,
      "temporal_span": {
        "start_time": 0.0,
        "end_time": 1.5,
        "start_frame": 0,
        "end_frame": 45
      },
      "instances": [
        {
          "object_id": 1,
          "timestamp": 0.0,
          "frame_idx": 0,
          "class_name": "person",
          "position_3d": [10.5, 2.3, 0.0],
          "rotation": [0.0, 0.0, 1.57],
          "bbox_3d": {
            "min": [10.2, 2.0, -0.5],
            "max": [10.8, 2.6, 0.5]
          },
          "dimensions": [0.6, 1.2, 0.5],
          "num_views": 3,
          "confidence": 0.85,
          "camera_ids": ["s1-left", "s1-right", "s2-left"]
        }
      ]
    }
  ]
}
```

**Key Fields:**
- `position_3d`: [x, y, z] in world coordinates (meters)
- `rotation`: [roll, pitch, yaw] in radians
- `bbox_3d`: 3D bounding box (min/max corners)
- `dimensions`: [width, height, depth]
- `num_views`: Number of cameras that saw this object
- `confidence`: Reconstruction confidence [0-1]

## 📁 Directory Structure

```
static-scene/
├── config/
│   └── pipeline_config.yaml          # Configuration
├── data/
│   └── videos/                        # Input videos (gitignored)
├── pass1_static/
│   └── reconstruct_static_scene.py   # Static reconstruction script
├── pass2_dynamic/
│   ├── track_objects.py              # Tracking script
│   └── reconstruct_objects.py        # 3D reconstruction script
├── utils/
│   ├── camera_utils.py               # Camera utilities
│   └── logger.py                     # Logging
├── outputs/                          # All outputs (gitignored)
│   ├── pass1_static/
│   │   ├── dust3r_pointcloud.ply
│   │   └── cameras.json
│   └── pass2_dynamic/
│       ├── tracks_*.pkl
│       └── objects_3d/
│           ├── objects_3d.json       # ← Main output
│           ├── object_*.ply
│           └── summary.txt
├── install_dependencies.sh
├── requirements.txt
├── .gitignore                        # Excludes repos, models, data
└── README.md
```

## ⚙️ Configuration

Edit `config/pipeline_config.yaml`:

```yaml
pass1_static:
  static_gaussians:
    iterations: 1000  # DUSt3R iterations

pass2_dynamic:
  sample_rate: 5  # Process every 5th frame
  
  tracking:
    detector: "yolov8x"  # YOLO model
    conf_threshold: 0.3
    iou_threshold: 0.5
    
    pedestrian_classes: ["person"]
    vehicle_classes: ["car", "bus", "truck", "motorcycle", "bicycle"]
  
  segmentation:
    method: "sam2"
    use_box_prompts: true
  
  aggregation:
    max_temporal_gap: 10  # Max frames between observations
    min_observations: 2   # Min cameras to reconstruct
```

## 🔧 Technical Details

### Phase 1: Static Scene
- **DUSt3R**: Neural multi-view 3D reconstruction
- **Output**: 1.3M point cloud with confidence filtering
- **Cameras**: Automatic pose estimation

### Phase 2A: Object Tracking
- **YOLOv8x**: Object detection (pedestrians, vehicles)
- **ByteTrack**: Multi-object tracking with occlusion handling
- **SAM2**: Instance segmentation for precise masks
- **Per-camera tracking**: Independent tracks saved

### Phase 2B: 3D Reconstruction
- **Multi-camera matching**: Temporal and spatial association
- **Triangulation**: 3D position from multiple views
- **Geometry estimation**: Position, rotation, dimensions
- **JSON export**: Easy integration with static scene

## 🎯 Performance

**Processing Time (RTX 3080):**
- Static reconstruction: 2-3 minutes
- Tracking (per camera): 5-10 minutes
- 3D reconstruction: 1-2 minutes

**Expected Output:**
- Static: 1.3M points
- Tracking: 50-200 tracks per camera
- Reconstruction: 10-50 global objects

## 🐛 Troubleshooting

### "CUDA not available"
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### "SAM2 not found"
```bash
cd sam2_repo
pip install --no-build-isolation -e .
```

### "DUSt3R import failed"
```bash
cd dust3r
git submodule update --init --recursive
pip install -r requirements.txt
```

### "No tracks found"
Check that Phase 2A completed successfully:
```bash
ls outputs/pass2_dynamic/tracks_*.pkl
```

## 📚 Dependencies

- **PyTorch** 2.7+ (CUDA 11.8+)
- **ultralytics** (YOLOv8)
- **ByteTrack** (multi-object tracking)
- **SAM2** (segmentation)
- **DUSt3R** (3D reconstruction)
- **Open3D**, NumPy, OpenCV

## 🔗 References

- [DUSt3R](https://github.com/naver/dust3r) - 3D reconstruction
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Multi-object tracking
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - Segmentation

## 📄 License

This project combines multiple open-source components, each with their own licenses.
Please refer to individual repositories for license details.
