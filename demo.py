#!/usr/bin/env python3
"""
ShotToMatte Demo Script

This script demonstrates how to use the ShotToMatte project to extract matte paintings
from a video file using a multi-stage pipeline approach.

Usage:
    python demo.py <video_path> [options]

Options:
    --output_dir OUTPUT_DIR       Directory to save output panoramas (default: output)
    --sample_rate SAMPLE_RATE     Process every Nth frame (default: 1)
    --debug                       Enable debug output
    --config_file CONFIG_FILE     Path to JSON configuration file
    --scene_threshold THRESHOLD   Similarity threshold for scene detection (default: 0.7)
    --scan_threshold THRESHOLD    Minimum flow magnitude for scan detection (default: 0.8)
    --flow_consistency THRESHOLD  Threshold for flow direction consistency (default: 0.8)
    --min_inlier_ratio RATIO      Minimum ratio of inliers for homography (default: 0.4)

Example:
    python demo.py my_video.mp4 --output_dir output --sample_rate 2 --debug
"""

import os
import argparse
import time
import json
from shottomatte import MatteExtractor
from tqdm import tqdm
import sys
import torch

class StatusDisplay:
    def __init__(self):
        self.last_status = ""
        
    def update(self, status):
        # Clear the last status line
        if self.last_status:
            sys.stdout.write('\r' + ' ' * len(self.last_status) + '\r')
        # Write the new status
        sys.stdout.write('\r' + status)
        sys.stdout.flush()
        self.last_status = status
        
    def clear(self):
        if self.last_status:
            sys.stdout.write('\r' + ' ' * len(self.last_status) + '\r')
            sys.stdout.flush()
            self.last_status = ""

def main():
    print("\n=== Initializing Demo ===", flush=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract matte paintings from a video file')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('--output_dir', default='output', help='Directory to save output panoramas')
    parser.add_argument('--sample_rate', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    # Scene detection parameters
    parser.add_argument('--scene_threshold', type=float, default=0.7, help='Similarity threshold for scene detection')
    parser.add_argument('--min_scene_length', type=int, default=30, help='Minimum frames in a scene to consider')
    
    # Scan detection parameters
    parser.add_argument('--scan_threshold', type=float, default=0.8, help='Minimum flow magnitude for scan detection')
    parser.add_argument('--flow_consistency', type=float, default=0.8, help='Threshold for flow direction consistency')
    parser.add_argument('--min_scan_duration', type=int, default=10, help='Minimum frames in a scan')
    
    # Panorama creation parameters
    parser.add_argument('--min_inlier_ratio', type=float, default=0.4, help='Minimum ratio of inliers for homography')
    parser.add_argument('--max_frames_for_stitch', type=int, default=15, help='Maximum frames to use for stitching')
    
    # Performance parameters
    parser.add_argument('--sample_rate_scene_detection', type=int, default=3, 
                        help='Process every Nth frame for scene detection')
    parser.add_argument('--sample_rate_flow_analysis', type=int, default=1,
                        help='Process every Nth frame for flow analysis')
    
    # Configuration file
    parser.add_argument('--config_file', help='Path to JSON configuration file')
    
    print("Parsing arguments...", flush=True)
    args = parser.parse_args()
    print("Arguments parsed successfully", flush=True)

    # Check CUDA status
    print("\nChecking CUDA status:", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}", flush=True)
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}", flush=True)
            print(f"  Compute capability: {torch.cuda.get_device_capability(i)}", flush=True)
        print(f"Current device: {torch.cuda.current_device()}", flush=True)
        print(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB", flush=True)
        print(f"Initial CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB", flush=True)
    
    # Verify video file exists and is readable
    if not os.path.exists(args.video_path):
        print(f"\nError: Video file not found: {args.video_path}", flush=True)
        return 1
    
    print(f"\nVerifying video file: {args.video_path}", flush=True)
    import cv2
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error: Could not open video file", flush=True)
        return 1
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames", flush=True)
    cap.release()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare configuration
    config = {
        # Scene detection parameters
        'scene_threshold': args.scene_threshold,
        'min_scene_length': args.min_scene_length,
        
        # Scan detection parameters
        'scan_threshold': args.scan_threshold,
        'flow_consistency_threshold': args.flow_consistency,
        'min_scan_duration': args.min_scan_duration,
        
        # Panorama creation parameters
        'min_inlier_ratio': args.min_inlier_ratio,
        'max_frames_for_stitch': args.max_frames_for_stitch,
        
        # Performance parameters
        'sample_rate_scene_detection': args.sample_rate_scene_detection,
        'sample_rate_flow_analysis': args.sample_rate_flow_analysis,
        
        # Debug output
        'debug_output': args.debug,
        
        # Progress callback
        'progress_callback': StatusDisplay()
    }
    
    # Load config from file if provided
    if args.config_file and os.path.exists(args.config_file):
        try:
            with open(args.config_file, 'r') as f:
                file_config = json.load(f)
                # Don't override progress_callback
                progress_callback = config['progress_callback']
                config.update(file_config)
                config['progress_callback'] = progress_callback
                print(f"Loaded configuration from {args.config_file}")
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    # Initialize the matte extractor with our configuration
    print("\nInitializing MatteExtractor with configuration:")
    for key, value in config.items():
        if key != 'progress_callback':  # Skip printing the callback
            print(f"  {key}: {value}")
    
    print("\nInitializing components...")
    extractor = MatteExtractor(config)
    
    # Process the video
    print(f"\nProcessing video: {args.video_path}")
    start_time = time.time()
    
    try:
        panoramas = extractor.process_video(
            args.video_path,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate
        )
        
        # Clear the status display before printing final results
        config['progress_callback'].clear()
        
        # Print results
        elapsed_time = time.time() - start_time
        print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        print(f"Extracted {len(panoramas)} panoramas")
        print(f"Output saved to {os.path.abspath(args.output_dir)}")
        
        # Save configuration used
        config_path = os.path.join(args.output_dir, 'config_used.json')
        with open(config_path, 'w') as f:
            # Remove progress_callback before saving
            save_config = {k: v for k, v in config.items() if k != 'progress_callback'}
            json.dump(save_config, f, indent=2)
        print(f"Configuration saved to {config_path}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    print("=== Starting ShotToMatte Demo ===", flush=True)
    print(f"Python executable: {sys.executable}", flush=True)
    print(f"Current working directory: {os.getcwd()}", flush=True)
    print(f"Arguments: {sys.argv}", flush=True)
    
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nScript interrupted by user", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnhandled error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1) 