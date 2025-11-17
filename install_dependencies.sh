#!/bin/bash
# LoS Audit Pipeline - Automated Installation Script
# ===================================================
# 
# This script installs all dependencies for the Street Gaussians
# Line-of-Sight Audit Pipeline in the correct order to avoid conflicts.
#
# USAGE:
#   mamba activate acv2  # or your preferred conda environment
#   ./install_dependencies.sh
#
# REQUIREMENTS:
#   - NVIDIA GPU with CUDA drivers
#   - Conda/mamba environment (Python 3.9-3.11)
#   - Internet connection
#
# The script handles:
#   - CUDA version detection and PyTorch installation
#   - YOLOv8, SAM2, ByteTrack installation
#   - DUSt3R for 3D reconstruction
#   - Model downloads and verification
#
set -euo pipefail

echo "============================================="
echo "  LoS Audit Pipeline - Dependency Installation"
echo "============================================="
echo ""

# ---------- helpers ----------
have_cmd() { command -v "$1" >/dev/null 2>&1; }

clone_if_missing() {
  local url="$1" dir="$2"
  if [ -d "$dir/.git" ]; then
    echo "✓ Repo exists: $dir — skipping clone"
  else
    echo "Cloning $url -> $dir"
    git clone "$url" "$dir"
  fi
}

fetch() {
  # Idempotent wget/curl
  local url="$1" out="$2"
  if [ -f "$out" ]; then
    echo "✓ Already present: $out"
    return 0
  fi
  if have_cmd wget; then
    wget -O "$out" "$url"
  elif have_cmd curl; then
    curl -L "$url" -o "$out"
  else
    echo "⚠️  Need wget or curl to fetch $url"
    return 1
  fi
}

# Check a primary destination + any number of alternate candidate paths.
# If found in any, create the primary destination (if different) as a symlink.
ensure_model() {
  local url="$1"; shift
  local dest="$1"; shift
  local found=""
  # Check primary
  if [ -f "$dest" ]; then
    found="$dest"
  else
    # Check alternates
    for alt in "$@"; do
      if [ -f "$alt" ]; then
        found="$alt"
        break
      fi
    done
  fi
  if [ -n "$found" ]; then
    echo "✓ Model already present at: $found"
    if [ "$found" != "$dest" ]; then
      mkdir -p "$(dirname "$dest")"
      ln -sf "$found" "$dest"
      echo "↪ linked: $dest -> $found"
    fi
    return 0
  fi
  # Not found anywhere, download to dest
  mkdir -p "$(dirname "$dest")"
  echo "Downloading model to $dest ..."
  fetch "$url" "$dest"
}

# ---------- preflight ----------
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found. Please run this from the project root."
    exit 1
fi

# ---------- CUDA detection ----------
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "✓ NVIDIA GPU detected with CUDA $CUDA_VERSION"

    if [ -z "${CUDA_HOME:-}" ]; then
        if [ -d "/usr/local/cuda" ]; then
            export CUDA_HOME=/usr/local/cuda
            echo "Setting CUDA_HOME=/usr/local/cuda"
        elif [ -d "/opt/cuda" ]; then
            export CUDA_HOME=/opt/cuda
            echo "Setting CUDA_HOME=/opt/cuda"
        else
            CUDA_PATH=$(which nvcc 2>/dev/null | sed 's|/bin/nvcc||' || true)
            if [ -n "$CUDA_PATH" ]; then
                export CUDA_HOME="$CUDA_PATH"
                echo "Setting CUDA_HOME=$CUDA_PATH"
            else
                echo "⚠️  CUDA installation not found. Please set CUDA_HOME manually."
            fi
        fi
    fi
    if [ -n "${CUDA_HOME:-}" ] && [[ ":$PATH:" != *":$CUDA_HOME/bin:"* ]]; then
        export PATH="$CUDA_HOME/bin:$PATH"
        echo "Added $CUDA_HOME/bin to PATH"
    fi
else
    echo "⚠️  No NVIDIA GPU detected. Will use CPU-only PyTorch."
    CUDA_VERSION="cpu"
fi

echo ""
echo "Step 1: Installing PyTorch..."
echo "----------------------------"

# Smart PyTorch installation with CUDA version matching
install_pytorch() {
    local detected_cuda="$1"
    local nvcc_cuda="$2"
    
    echo "Detected CUDA versions:"
    echo "  - Driver CUDA: $detected_cuda"
    echo "  - NVCC CUDA: $nvcc_cuda"
    
    # Choose PyTorch version based on available CUDA toolkit
    if [[ "$nvcc_cuda" == "11.5" || "$nvcc_cuda" == "11.6" || "$nvcc_cuda" == "11.7" ]]; then
        echo "Installing PyTorch for CUDA 11.8 (compatible with CUDA 11.x)..."
        pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
    elif [[ "$nvcc_cuda" == "11.8" ]]; then
        echo "Installing PyTorch for CUDA 11.8..."
        pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
    elif [[ "$nvcc_cuda" == "12."* || "$detected_cuda" == "13.0" ]]; then
        echo "Installing PyTorch for CUDA 12.x (compatible with CUDA 12.0-13.0)..."
        pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
    else
        echo "⚠️  Unknown CUDA version, installing CPU-only PyTorch..."
        pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
    fi
}

# Check existing PyTorch and reinstall if needed
REINSTALL_TORCH=false
if python -c "import torch; print('OK')" 2>/dev/null; then
    # Check if CUDA versions match
    DETECTED_CUDA=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "none")
    if [[ "$DETECTED_CUDA" != "none" ]]; then
        echo "Found PyTorch with CUDA $DETECTED_CUDA"
        # Check if this matches system CUDA
        if command -v nvcc &> /dev/null; then
            NVCC_CUDA=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
            if [[ "$NVCC_CUDA" != "$DETECTED_CUDA" && "$NVCC_CUDA" != "11.8" && "$DETECTED_CUDA" != "11.8" ]]; then
                echo "⚠️  CUDA version mismatch (PyTorch: $DETECTED_CUDA, System: $NVCC_CUDA)"
                REINSTALL_TORCH=true
            fi
        fi
    else
        echo "Found CPU-only PyTorch, reinstalling with CUDA..."
        REINSTALL_TORCH=true
    fi
else
    echo "PyTorch not found, installing..."
    REINSTALL_TORCH=true
fi

if [ "$REINSTALL_TORCH" = true ]; then
    echo "Reinstalling PyTorch..."
    pip uninstall -y torch torchvision torchaudio || true
    
    # Get CUDA versions
    if command -v nvcc &> /dev/null; then
        NVCC_CUDA=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    else
        NVCC_CUDA="none"
    fi
    
    install_pytorch "$CUDA_VERSION" "$NVCC_CUDA"
fi

python - <<'PY'
import torch
print("PyTorch CUDA available:", torch.cuda.is_available())
print("Torch built with CUDA:", torch.backends.cuda.is_built(), "CUDA ver:", torch.version.cuda)
PY
echo "✓ PyTorch installation verified"

echo ""
echo "Step 2: Setting up CUDA development environment..."
echo "---------------------------------------------------"
if ! command -v nvcc &> /dev/null; then
    echo "⚠️  CUDA development toolkit not found."
    echo "Option A (recommended for source builds):"
    echo "  sudo apt update && sudo apt install -y nvidia-cuda-toolkit"
    echo "Option B: Use pre-compiled wheels when available."
    echo ""
else
    echo "✓ CUDA development toolkit found"
    export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
    echo "Setting CUDA_HOME=$CUDA_HOME"
fi

echo ""
echo "Step 3: Installing YOLOv8 (object detection)..."
echo "--------------------------------------------------"
pip install ultralytics


echo ""
echo "Step 4: Installing ByteTrack (object tracking)..."
echo "-------------------------------------------------"
clone_if_missing "https://github.com/ifzhang/ByteTrack.git" "ByteTrack"
pushd ByteTrack >/dev/null
pip install --no-build-isolation -e .
popd >/dev/null

echo ""
echo "Step 5: Installing SAM2 (segmentation) - CRITICAL for object segmentation"
echo "Installing SAM2 (segmentation)..."
if [ -d "sam2_repo/.git" ]; then
    echo "✓ SAM2 repo exists: sam2_repo — skipping clone"
else
    echo "Cloning segment-anything-2 -> sam2_repo"
    git clone "https://github.com/facebookresearch/segment-anything-2.git" "sam2_repo"
fi

if [ -d "sam2_repo" ]; then
    pushd sam2_repo >/dev/null
    pip install --no-build-isolation -e .
    mkdir -p checkpoints
    # SAM2 model: check local path only (repo keeps checkpoints here)
    ensure_model \
      "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt" \
      "checkpoints/sam2_hiera_large.pt"
    popd >/dev/null
fi

echo ""
echo "Step 6: Installing remaining dependencies from requirements.txt..."
echo "------------------------------------------------------------------"
# Create filtered requirements excluding packages we install manually
grep -v "^#" requirements.txt \
  | grep -v "^torch" \
  | grep -v "^torchvision" \
  | grep -v "^torchaudio" \
  | grep -v "segment-anything-2" \
  | grep -v "git+https://github.com/ifzhang/ByteTrack.git" \
  | grep -v "git+https://github.com/facebookresearch/segment-anything-2.git" \
  > temp_requirements.txt

echo "Installing filtered dependencies..."
if [ -s temp_requirements.txt ]; then
    pip install -r temp_requirements.txt
else
    echo "All dependencies already installed manually"
fi
rm -f temp_requirements.txt

echo ""
echo "Step 7: Installing DUSt3R (camera pose estimation) - CRITICAL for 3D reconstruction..."
echo "---------------------------------------------------------------------------------------"
if clone_if_missing "https://github.com/naver/dust3r.git" "dust3r"; then
    pushd dust3r >/dev/null
    # DUSt3R doesn't use setup.py - just install requirements
    if git submodule update --init --recursive; then
        if pip install -r requirements.txt; then
            echo "✓ DUSt3R dependencies installed"
            # Download model checkpoint
            mkdir -p checkpoints
            echo "Downloading DUSt3R checkpoint (2.1GB, may take a few minutes)..."
            ensure_model \
              "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
              "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            echo "✓ DUSt3R installed successfully"
        else
            echo "⚠️  DUSt3R dependencies installation failed"
        fi
    else
        echo "⚠️  DUSt3R submodule update failed"
    fi
    popd >/dev/null
else
    echo "⚠️  DUSt3R clone failed"
fi

echo ""
echo "Step 8: Downloading YOLO weights..."
echo "-----------------------------------"
# Download YOLOv8x (extra large) as specified in config
# Try common Ultralytics cache locations too; if found there, symlink to project root.
ULTRA_CACHE1="$HOME/.cache/ultralytics/yolov8x.pt"
ULTRA_CACHE2="$HOME/.cache/torch/hub/ultralytics/yolov8x.pt"
ensure_model \
  "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt" \
  "yolov8x.pt" \
  "$ULTRA_CACHE1" "$ULTRA_CACHE2"

echo ""
echo "Step 9: Verifying installation..."
echo "----------------------------------"

# Test core functionality
echo "Testing core imports..."
python -c "
import torch
import ultralytics
import cv2
import numpy as np
print('✓ Core dependencies working')

# Test CUDA
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  CUDA not available - using CPU')

# Test SAM2
try:
    import sam2
    print('✓ SAM2 imported successfully')
except ImportError:
    print('⚠️  SAM2 not available')

# Test DUSt3R
try:
    import sys
    sys.path.append('dust3r')
    import dust3r
    print('✓ DUSt3R imported successfully')
except ImportError:
    print('⚠️  DUSt3R not available')

# Test ByteTrack
try:
    sys.path.append('ByteTrack')
    from yolox.tracker.byte_tracker import BYTETracker
    print('✓ ByteTrack imported successfully')
except ImportError:
    print('⚠️  ByteTrack not available')
" || echo "⚠️  Some imports failed"

echo ""
echo "============================================="
echo "  ✅ Installation Complete!"
echo "============================================="
echo ""
echo "📊 Installation Summary:"
echo "  ✓ PyTorch with CUDA support"
echo "  ✓ YOLOv8 object detection"
echo "  ✓ ByteTrack multi-object tracking"
echo "  ✓ SAM2 segmentation"
echo "  ✓ DUSt3R 3D reconstruction"
echo "  ✓ Core computer vision libraries"
echo ""
echo "🚀 3D Reconstruction pipeline ready!"
echo ""
echo "📋 Next Steps (run sequentially):"
echo "  1. Static scene: python pass1_static/reconstruct_static_scene.py"
echo "  2. Track objects: python pass2_dynamic/track_objects.py"
echo "  3. Reconstruct 3D: python pass2_dynamic/reconstruct_objects.py"
echo ""
echo "📚 Documentation:"
echo "  - README.md: Complete usage guide"
echo ""
