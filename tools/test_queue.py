#!/usr/bin/env python3
"""
Test script for FrameQueue system with enhanced scene tracking.
"""

import sys
import os
from pathlib import Path
import time
import numpy as np
import torch
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from shottomatte.queue import FrameQueue, DynamicQueueSizer, WorkUnit, SceneInfo, SceneContext

def create_dummy_frame(size=(720, 1280, 3)):
    """Create a dummy frame for testing."""
    return np.random.randint(0, 255, size=size, dtype=np.uint8)

def test_scene_info_structure():
    """Test the enhanced scene info data structure."""
    print("\n=== Testing Scene Info Structure ===")
    
    # Test SceneContext validation
    print("\nTesting SceneContext validation...")
    try:
        context = SceneContext(frame_position=1.5)  # Should fail
        print("ERROR: Invalid frame position accepted")
    except ValueError as e:
        print("Correctly caught invalid frame position")
        
    # Test SceneInfo initialization
    print("\nTesting SceneInfo initialization...")
    scene_info = SceneInfo(
        scene_id=1,
        confidence=0.85,
        boundary_type='start'
    )
    print(f"Created scene info with ID {scene_info.scene_id}")
    
    # Test invalid boundary type
    print("\nTesting boundary type validation...")
    try:
        scene_info.boundary_type = 'invalid'
        print("ERROR: Invalid boundary type accepted")
    except ValueError as e:
        print("Correctly caught invalid boundary type")
        
    # Test feature management
    print("\nTesting feature management...")
    scene_info.update_feature('motion_score', 0.75)
    scene_info.update_feature('color_hist', [0.1, 0.2, 0.3])
    print(f"Added features: {list(scene_info.features.keys())}")
    
    # Test feature timestamps
    time.sleep(0.1)  # Wait to test age
    age = scene_info.get_feature_age('motion_score')
    print(f"Feature age: {age:.3f}s")
    
    # Test completeness check
    print("\nTesting completeness check...")
    print(f"Is complete: {scene_info.is_complete()}")
    
    return scene_info

def test_temporal_context():
    """Test temporal context tracking."""
    print("\n=== Testing Temporal Context ===")
    
    queue = FrameQueue(detection_maxlen=5)
    
    # Create a sequence of frames with temporal relationships
    print("\nCreating frame sequence...")
    frames: List[WorkUnit] = []
    for i in range(5):
        frame = create_dummy_frame()
        queue.add_frame(frame)
        workunit = queue.detection_queue[-1]
        
        # Set up temporal context
        workunit.update_temporal_context(
            prev_scene=i-1 if i > 0 else None,
            next_scene=i+1 if i < 4 else None,
            position=i/4.0  # Position in scene from 0.0 to 1.0
        )
        
        # Set scene info
        workunit.scene_info.scene_id = i // 2  # Change scene every 2 frames
        workunit.scene_info.confidence = 0.8 + (i * 0.05)
        
        frames.append(workunit)
        print(f"Frame {i}: Scene {workunit.scene_info.scene_id}, "
              f"Position {workunit.scene_info.temporal_context.frame_position:.2f}")
    
    # Test temporal relationships
    print("\nVerifying temporal relationships...")
    for i, frame in enumerate(frames):
        context = frame.scene_info.temporal_context
        print(f"Frame {i}: Prev={context.prev_scene_id}, "
              f"Next={context.next_scene_id}, "
              f"Pos={context.frame_position:.2f}")
    
    return frames

def test_feature_management():
    """Test feature management and aging."""
    print("\n=== Testing Feature Management ===")
    
    workunit = WorkUnit(
        frame_idx=0,
        timestamp=time.perf_counter(),
        frame=create_dummy_frame()
    )
    
    # Add features with timestamps
    print("\nAdding features...")
    features = {
        'motion_score': 0.85,
        'scene_confidence': 0.92,
        'color_histogram': np.random.rand(256),
        'edge_density': 0.45
    }
    
    for name, value in features.items():
        workunit.update_scene_feature(name, value)
        print(f"Added feature: {name}")
        time.sleep(0.1)  # Create age difference
    
    # Check feature ages
    print("\nChecking feature ages...")
    for name in features:
        age = workunit.scene_info.get_feature_age(name)
        print(f"Feature {name} age: {age:.3f}s")
    
    # Test feature updates
    print("\nTesting feature updates...")
    workunit.update_scene_feature('motion_score', 0.95)
    new_age = workunit.scene_info.get_feature_age('motion_score')
    print(f"Updated motion_score, new age: {new_age:.3f}s")
    
    return workunit

def test_scene_transitions():
    """Test scene boundary detection and transitions."""
    print("\n=== Testing Scene Transitions ===")
    
    queue = FrameQueue(detection_maxlen=10)
    
    # Simulate a scene transition sequence
    print("\nSimulating scene transition sequence...")
    transitions = [
        ('middle', 0.85, 0.3),  # Middle of scene 1
        ('middle', 0.82, 0.6),  # Later in scene 1
        ('end', 0.78, 1.0),     # End of scene 1
        ('start', 0.88, 0.0),   # Start of scene 2
        ('middle', 0.92, 0.2)   # Early in scene 2
    ]
    
    frames = []
    for i, (boundary_type, conf, pos) in enumerate(transitions):
        frame = create_dummy_frame()
        queue.add_frame(frame)
        workunit = queue.detection_queue[-1]
        
        # Set scene boundary and confidence
        workunit.set_scene_boundary(boundary_type, conf)
        workunit.scene_info.scene_id = 1 if i < 3 else 2
        
        # Set temporal context with position
        workunit.update_temporal_context(
            prev_scene=1 if i >= 3 else None,  # Previous scene for scene 2
            next_scene=2 if i < 3 else None,   # Next scene for scene 1
            position=pos
        )
        
        # Add some features
        workunit.update_scene_feature('motion', 0.5 + i * 0.1)
        workunit.update_scene_feature('edge_density', 0.3 + i * 0.05)
        
        # Set workunit confidence (needed for stitch readiness)
        workunit.confidence = conf * 0.95  # Slightly lower than scene confidence
        
        frames.append(workunit)
        print(f"Frame {i}: {boundary_type} (conf: {conf:.2f}, pos: {pos:.2f}) - Scene {workunit.scene_info.scene_id}")
    
    # Test readiness for different stages
    print("\nTesting stage readiness...")
    for i, workunit in enumerate(frames):
        flow_ready = workunit.is_ready_for_flow()
        stitch_ready = workunit.is_ready_for_stitch()
        print(f"Frame {i}: Flow ready: {flow_ready}, Stitch ready: {stitch_ready}")
        if not stitch_ready:
            print(f"  - Confidence: {workunit.confidence}")
            print(f"  - Scene confidence: {workunit.scene_info.confidence}")
            print(f"  - Position: {workunit.scene_info.temporal_context.frame_position}")
            print(f"  - Complete: {workunit.scene_info.is_complete()}")
    
    # Get scene summaries
    print("\nScene summaries:")
    for i, workunit in enumerate(frames):
        summary = workunit.get_scene_summary()
        print(f"Frame {i}:", summary)
    
    return frames

def main():
    """Run all scene info tests."""
    print("Starting enhanced scene info tests...")
    
    # Test basic scene info structure
    scene_info = test_scene_info_structure()
    
    # Test temporal context
    frames = test_temporal_context()
    
    # Test feature management
    workunit = test_feature_management()
    
    # Test scene transitions
    transition_frames = test_scene_transitions()
    
    print("\nAll tests completed.")
    
if __name__ == "__main__":
    main() 