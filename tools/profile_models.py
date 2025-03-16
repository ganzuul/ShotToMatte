#!/usr/bin/env python3
"""
Profile model initialization and memory usage.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from shottomatte.profiler import PipelineProfiler
from shottomatte.flow import RAFTOpticalFlow
from shottomatte.scene import ContentAwareSceneDetector
import torch
import kornia

def profile_models():
    print("=== Model Profiling Session ===")
    
    # Initialize profiler
    profiler = PipelineProfiler(output_dir='profile_results')
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nInitial CUDA memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
    
    # Profile ResNet18 initialization
    with profiler.profile_stage("resnet18_init"):
        scene_detector = ContentAwareSceneDetector(debug_visualizer=None)
        
    # Profile RAFT initialization
    with profiler.profile_stage("raft_init"):
        flow_estimator = RAFTOpticalFlow()
        
    # Profile LoFTR initialization
    with profiler.profile_stage("loftr_init"):
        matcher = kornia.feature.LoFTR(pretrained='outdoor')
        matcher = matcher.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        matcher.eval()
    
    # Test models with appropriately sized inputs
    resnet_input = torch.zeros(1, 3, 224, 224)  # ResNet standard input size
    raft_input = torch.zeros(1, 3, 256, 256)    # RAFT requires at least 128x128
    loftr_input = torch.zeros(1, 1, 256, 256)   # LoFTR grayscale input
    
    if torch.cuda.is_available():
        resnet_input = resnet_input.cuda()
        raft_input = raft_input.cuda()
        loftr_input = loftr_input.cuda()
    
    with torch.no_grad():
        # Profile ResNet18 inference
        with profiler.profile_stage("resnet18_inference"):
            _ = scene_detector.feature_extractor(resnet_input)
            
        # Profile RAFT inference
        with profiler.profile_stage("raft_inference"):
            _ = flow_estimator(raft_input, raft_input)
            
        # Profile LoFTR inference
        with profiler.profile_stage("loftr_inference"):
            _ = matcher({'image0': loftr_input, 'image1': loftr_input})
    
    # Log final memory state
    profiler.log_metric("final_gpu_memory", 
                       torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)
    
    # Print and save results
    profiler.print_summary()
    profiler.save_session()

if __name__ == "__main__":
    profile_models() 