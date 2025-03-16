#!/usr/bin/env python3
"""Integration tests for the enhanced queue-based MatteExtractor pipeline."""

import sys
import os
from pathlib import Path
import time
import numpy as np
import cv2
import torch
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

from shottomatte import MatteExtractor

class MockProgressCallback:
    """Mock progress callback for testing."""
    def __init__(self):
        self.updates = []
        
    def update(self, status):
        self.updates.append(status)
        
def create_test_video(output_path, width=640, height=480, num_frames=20, fps=30.0):
    """Create a test video with panning motion."""
    # Create a wider background to pan across
    background_width = width * 4  # Make it wider for more obvious panning
    background = np.zeros((height, background_width, 3), dtype=np.uint8)
    
    # Add some recognizable patterns to the background
    # Green rectangle
    cv2.rectangle(background, (100, 100), (300, 300), (0, 255, 0), -1)
    # Red circle
    cv2.circle(background, (background_width//2, height//2), 100, (0, 0, 255), -1)
    # Blue diagonal line
    cv2.line(background, (background_width-400, 50), (background_width-100, height-50), (255, 0, 0), 10)
    # Add text
    cv2.putText(background, "TEST PATTERN", (background_width//4, height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Calculate step size for smooth panning
    x_step = (background_width - width) / (num_frames - 1)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Generate frames with panning motion
    for i in range(num_frames):
        # Calculate current x position
        x = int(i * x_step)
        
        # Extract current view
        frame = background[:, x:x+width].copy()
        
        # Add frame number and timestamp
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {i/fps:.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
    
    # Release video writer
    out.release()
    
    print(f"Created test video at {output_path}")

def test_queue_integration():
    """Test the integration of the queue system with the main pipeline."""
    print("\n=== Testing Queue Integration ===\n")
    
    # Create output directory
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    
    # Create test video
    video_path = test_dir / "test_pan.mp4"
    create_test_video(str(video_path))
    
    # Initialize extractor with test configuration
    config = {
        'scene_threshold': 0.3,  # Lower threshold for testing
        'scan_threshold': 0.1,  # Lower threshold for testing
        'min_scan_duration': 2,  # Lower duration for testing
        'flow_consistency_threshold': 0.3,  # Lower threshold for testing
        'max_frames_for_stitch': 3,  # Lower threshold for testing
        'debug_output': True,
        'output_dir': str(test_dir)  # Set output directory
    }
    
    print("\n=== Initializing MatteExtractor ===")
    extractor = MatteExtractor(config)
    
    # Process video
    print("\nProcessing test video...")
    metrics = extractor.process_video(str(video_path), str(test_dir), sample_rate=2)
    
    # Verify queue metrics
    print("\nVerifying queue metrics...")
    assert metrics is not None, "No metrics returned"
    assert 'detection' in metrics, "No detection metrics"
    assert 'flow' in metrics, "No flow metrics"
    assert 'stitch' in metrics, "No stitch metrics"
    
    # Check queue health
    print("\nChecking queue health...")
    for queue in ['detection', 'flow', 'stitch']:
        assert metrics[queue]['processed'] > 0, f"No frames were processed in {queue} queue"
        assert metrics[queue]['dropped'] == 0, f"Frames were dropped in {queue} queue"
        assert metrics[queue]['total'] > 0, f"No total frames in {queue} queue"
    
    # Verify scene tracking
    print("\nVerifying scene tracking...")
    assert len(extractor.timers['scene_detection']) > 0, "No scene detection was performed"
    
    # Check performance metrics
    print("\nChecking performance metrics...")
    print(f"Scene detection times: {len(extractor.timers['scene_detection'])} operations")
    print(f"Flow analysis times: {len(extractor.timers['flow_analysis'])} operations")
    
    # Verify output files
    print("\nVerifying output files...")
    panorama_files = list(test_dir.glob("panorama_*.jpg"))
    assert len(panorama_files) > 0, "No panoramas were created"
    
    # Clean up
    def remove_dir_tree(path):
        for item in path.glob("*"):
            if item.is_file():
                item.unlink()
            else:
                remove_dir_tree(item)
                item.rmdir()
    
    remove_dir_tree(test_dir)
    test_dir.rmdir()
    
    return metrics

def test_memory_management():
    """Test memory management and cleanup."""
    print("\n=== Testing Memory Management ===")
    
    if not torch.cuda.is_available():
        print("Skipping memory test - CUDA not available")
        return
    
    # Record initial GPU memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    
    # Create test video
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    video_path = str(test_dir / "test_memory.mp4")
    create_test_video(video_path, num_frames=50)
    
    # Process video
    config = {
        'debug_output': False,  # Minimize extra memory usage
        'detection_maxlen': 5,  # Smaller queue sizes for testing
        'flow_maxlen': 5,
        'stitch_maxlen': 5
    }
    extractor = MatteExtractor(config)
    extractor.process_video(video_path, str(test_dir))
    
    # Clear references to help garbage collection
    extractor._flow_estimator = None
    extractor._scene_detector = None
    extractor._matcher = None
    extractor = None
    
    # Force garbage collection
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Check final memory
    final_memory = torch.cuda.memory_allocated()
    
    # Verify memory cleanup
    memory_diff = final_memory - initial_memory
    print(f"Memory difference: {memory_diff / 1024**2:.1f} MB")
    assert memory_diff < 100 * 1024**2, "Significant memory leak detected"
    
    print("Memory management test passed!")

def test_error_handling():
    """Test error handling in the queue system."""
    print("\n=== Testing Error Handling ===")
    
    # Test with invalid video
    config = {'debug_output': False}
    extractor = MatteExtractor(config)
    
    with pytest.raises(ValueError):
        extractor.process_video("nonexistent.mp4")
    
    # Test with empty video
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    empty_video = str(test_dir / "empty.mp4")
    
    # Create empty video file
    with open(empty_video, 'wb') as f:
        f.write(b'')
    
    with pytest.raises(ValueError):
        extractor.process_video(empty_video)
    
    print("Error handling tests passed!")

if __name__ == "__main__":
    # Run all tests
    print("Starting integration tests...")
    
    try:
        metrics = test_queue_integration()
        print("\nQueue metrics:", metrics)
        
        test_memory_management()
        test_error_handling()
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        raise 