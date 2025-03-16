import sys
import os
# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
import cv2
from shottomatte import MatteExtractor

def test_initialization():
    """Test that the MatteExtractor can be initialized."""
    print("Testing MatteExtractor initialization...")
    extractor = MatteExtractor()
    print("✓ MatteExtractor initialized successfully")
    print(f"✓ Using device: {extractor.device}")
    
    # Test flow estimator
    print("\nTesting flow estimator initialization...")
    flow_estimator = extractor.flow_estimator
    print(f"✓ Flow estimator initialized: {flow_estimator.__class__.__name__}")
    
    # Test scene detector
    print("\nTesting scene detector initialization...")
    scene_detector = extractor.scene_detector
    print(f"✓ Scene detector initialized: {scene_detector.__class__.__name__}")
    
    # Test feature matcher
    print("\nTesting feature matcher initialization...")
    if extractor.use_loftr:
        print(f"✓ Using LoFTR feature matcher")
    else:
        print("✗ LoFTR not available, using OpenCV fallback")
    
    return extractor

def test_flow_computation():
    """Test optical flow computation with random frames."""
    print("\nTesting optical flow computation...")
    
    # Create two random frames
    frame1 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    # Initialize flow estimator
    extractor = MatteExtractor()
    
    # Compute flow
    flow_result = extractor.flow_estimator(frame1, frame2)
    
    # Extract flow field from result dictionary
    flow = flow_result['flow']
    magnitude = flow_result['magnitude']
    vector = flow_result['vector']
    
    print(f"✓ Flow shape: {flow.shape}")
    print(f"✓ Flow mean magnitude: {magnitude:.4f}")
    print(f"✓ Flow vector: {vector}")
    
    return flow_result

if __name__ == "__main__":
    print("=== Testing ShotToMatte Project ===\n")
    
    # Test initialization
    extractor = test_initialization()
    
    # Test flow computation
    flow = test_flow_computation()
    
    print("\n=== All tests completed ===") 