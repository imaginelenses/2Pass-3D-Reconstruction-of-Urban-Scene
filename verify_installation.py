#!/usr/bin/env python3
"""
Verify installation of all dependencies
"""

import sys
from pathlib import Path


def check_import(module_name, package_name=None, optional=False):
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name}: Installed")
        return True
    except ImportError as e:
        if optional:
            print(f"⚠ {package_name}: Not installed (optional)")
        else:
            print(f"✗ {package_name}: NOT INSTALLED - {e}")
        return False


def check_pytorch():
    """Check PyTorch installation with CUDA"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        
        print(f"✓ PyTorch: {torch.__version__}")
        if cuda_available:
            print(f"  └─ CUDA: {cuda_version} (Available)")
            print(f"  └─ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  └─ CUDA: Not available (CPU only)")
        
        return True
    except ImportError:
        print("✗ PyTorch: NOT INSTALLED")
        return False


def check_file_exists(filepath, description):
    """Check if a file exists"""
    path = Path(filepath)
    if path.exists():
        print(f"✓ {description}: Found at {filepath}")
        return True
    else:
        print(f"⚠ {description}: Not found at {filepath}")
        return False


def main():
    """Run all checks"""
    print("=" * 80)
    print("  VERIFYING LINE-OF-SIGHT AUDIT PIPELINE INSTALLATION")
    print("=" * 80)
    print()
    
    all_ok = True
    
    # Core dependencies
    print("Core Dependencies:")
    print("-" * 80)
    all_ok &= check_pytorch()
    all_ok &= check_import("torchvision")
    all_ok &= check_import("numpy")
    all_ok &= check_import("cv2", "opencv-python")
    all_ok &= check_import("PIL", "Pillow")
    all_ok &= check_import("tqdm")
    all_ok &= check_import("yaml", "pyyaml")
    all_ok &= check_import("omegaconf")
    print()
    
    # YOLO
    print("Object Detection:")
    print("-" * 80)
    all_ok &= check_import("ultralytics", "YOLOv8")
    check_file_exists("yolov8n.pt", "YOLO weights")
    print()
    
    # SAM2
    print("Segmentation:")
    print("-" * 80)
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✓ SAM2: Installed")
    except ImportError as e:
        print(f"✗ SAM2: NOT INSTALLED - {e}")
        all_ok = False
    print()
    
    # 3D Gaussian Splatting
    print("3D Gaussian Splatting:")
    print("-" * 80)
    try:
        import diff_gaussian_rasterization
        print("✓ diff-gaussian-rasterization: Installed")
    except ImportError:
        print("✗ diff-gaussian-rasterization: NOT INSTALLED")
    try:
        import simple_kNN
        print("✓ simple-knn: Installed")
    except ImportError:
        try:
            import simpleknn
            print("✓ simple-knn: Installed")
        except ImportError:
            print("✗ simple-knn: NOT INSTALLED")
            all_ok = False
    print()
    
    # Optional dependencies
    print("Optional Dependencies:")
    print("-" * 80)
    
    # Street Gaussians
    try:
        # Check if street_gaussians directory exists
        if Path("street_gaussians").exists():
            print("✓ Street Gaussians: Repository cloned")
        else:
            print("⚠ Street Gaussians: Repository not found")
    except:
        print("⚠ Street Gaussians: Not checked")
    
    # ByteTrack
    try:
        if Path("ByteTrack").exists():
            print("✓ ByteTrack: Repository cloned")
        else:
            print("⚠ ByteTrack: Repository not found")
    except:
        print("⚠ ByteTrack: Not checked")
    
    # DUSt3R
    try:
        if Path("dust3r").exists():
            print("✓ DUSt3R: Repository cloned (optional)")
        else:
            print("⚠ DUSt3R: Not found (optional)")
    except:
        pass
    
    print()
    
    # Check project structure
    print("Project Structure:")
    print("-" * 80)
    check_file_exists("config/pipeline_config.yaml", "Configuration file")
    check_file_exists("utils/__init__.py", "Utils module")
    check_file_exists("pass1_static/extract_static_backgrounds.py", "Pass 1 extractor")
    check_file_exists("main_pipeline.py", "Main pipeline")
    print()
    
    # Check data
    print("Data:")
    print("-" * 80)
    data_dir = Path("StreetAware-sample")
    if data_dir.exists():
        videos = list(data_dir.glob("*.mp4"))
        print(f"✓ Video directory: Found {len(videos)} videos")
    else:
        print("⚠ Video directory: Not found (StreetAware-sample/)")
    print()
    
    # Summary
    print("=" * 80)
    if all_ok:
        print("  ✓ ALL CORE DEPENDENCIES INSTALLED")
        print("  You can now run: python main_pipeline.py")
    else:
        print("  ✗ SOME DEPENDENCIES ARE MISSING")
        print("  Please review INSTALLATION.md for setup instructions")
    print("=" * 80)
    print()
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
