#!/usr/bin/env python3
import argparse
import os
import sys
import torch
from .panorama import MatteExtractor

def main():
    parser = argparse.ArgumentParser(
        description="ShotToMatte - Extract matte paintings from animated content"
    )
    
    parser.add_argument("input", help="Path to input video file or directory")
    parser.add_argument("--output", "-o", default="mattes", help="Output directory for panoramas")
    parser.add_argument("--sample-rate", "-s", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--min-duration", "-d", type=int, default=10, help="Minimum frames in a scan")
    parser.add_argument("--min-movement", "-m", type=float, default=0.2, 
                       help="Minimum movement ratio relative to frame width")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--use-segmentation", action="store_true", help="Use instance segmentation")
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (warning: this will be significantly slower)")
    
    # Configure extractor
    config = {
        'min_scan_duration': args.min_duration,
        'min_movement_ratio': args.min_movement,
        'use_segmentation': args.use_segmentation,
        'batch_size': args.batch_size,
    }
    
    # Create extractor
    extractor = MatteExtractor(config)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process input
    if os.path.isfile(args.input):
        # Single video file
        extractor.process_video(args.input, args.output, args.sample_rate)
    elif os.path.isdir(args.input):
        # Directory of videos
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
        videos = [f for f in os.listdir(args.input) 
                 if os.path.isfile(os.path.join(args.input, f)) and 
                 os.path.splitext(f)[1].lower() in video_extensions]
        
        if not videos:
            print(f"No video files found in {args.input}")
            return
        
        print(f"Found {len(videos)} video files to process")
        
        for video in videos:
            video_path = os.path.join(args.input, video)
            video_output = os.path.join(args.output, os.path.splitext(video)[0])
            os.makedirs(video_output, exist_ok=True)
            
            print(f"\nProcessing: {video}")
            extractor.process_video(video_path, video_output, args.sample_rate)
    else:
        print(f"Input not found: {args.input}")

if __name__ == "__main__":
    main() 