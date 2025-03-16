import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import deque
import kornia
import time
import os

from .flow import RAFTOpticalFlow
from .scene import ContentAwareSceneDetector
from .queue import FrameQueue, WorkUnit, SceneInfo, SceneContext

class MatteExtractor:
    """
    Extract matte paintings from animated content using advanced GPU techniques.
    
    This class implements the complete pipeline for detecting and reconstructing
    background art from panning/scanning shots in animation.
    """
    
    def __init__(self, config=None):
        """
        Initialize the matte extractor with configuration.
        
        Args:
            config: Configuration dictionary with parameters
        """
        print("\n=== Initializing MatteExtractor ===", flush=True)
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}", flush=True)
        if torch.cuda.is_available():
            print(f"CUDA memory at start: {torch.cuda.memory_allocated() / 1024**2:.1f} MB", flush=True)
        
        # Default configuration
        print("\nSetting up configuration...", flush=True)
        self.config = {
            # Scene detection parameters
            'scene_threshold': 0.7,         # Similarity threshold for scene detection
            'min_scene_length': 30,         # Minimum frames in a scene to consider for analysis
            
            # Scan detection parameters
            'scan_threshold': 0.8,          # Minimum flow magnitude to consider as scan
            'min_scan_duration': 10,        # Minimum frames in a scan
            'max_scan_frames': 60,          # Maximum frames to consider in a scan
            'min_movement_ratio': 0.2,      # Minimum movement relative to frame width
            'flow_consistency_threshold': 0.8, # Threshold for flow direction consistency
            'min_consistent_direction': 0.7, # Minimum ratio of consistent direction vectors
            
            # Panorama creation parameters
            'max_frames_for_stitch': 15,    # Maximum frames to use for stitching
            'min_inlier_ratio': 0.4,        # Minimum ratio of inliers for homography
            
            # Performance parameters
            'batch_size': 8,                # Batch size for processing
            'sample_rate_scene_detection': 3, # Process every Nth frame for scene detection
            'sample_rate_flow_analysis': 1, # Process every Nth frame for flow analysis
            
            # Queue parameters
            'detection_maxlen': 10,         # Maximum length of detection queue
            'flow_maxlen': 30,              # Maximum length of flow queue
            'stitch_maxlen': 50,            # Maximum length of stitch queue
            
            # Other parameters
            'use_segmentation': False,      # Whether to use instance segmentation
            'debug_output': True,           # Whether to save debug information
            
            # Progress callback
            'progress_callback': None,       # Callback for progress updates
            'last_progress': 0,             # Last progress value (0-100)
            'current_stage': 1,             # Current processing stage (1-3)
            'stage_progress': 0,            # Progress within current stage (0-100)
        }
        
        # Override defaults with provided config
        if config:
            print("Updating configuration with provided values...", flush=True)
            self.config.update(config)
            
        # Initialize components lazily to save resources
        print("\nPreparing for lazy initialization...", flush=True)
        self._flow_estimator = None
        self._scene_detector = None
        self._matcher = None
        
        # Initialize frame queue system
        self.frame_queue = FrameQueue(
            detection_maxlen=self.config['detection_maxlen'],
            flow_maxlen=self.config['flow_maxlen'],
            stitch_maxlen=self.config['stitch_maxlen']
        )
        
        # Initialize debug visualizer if debug output is enabled
        self.debug_visualizer = None
        if self.config['debug_output']:
            from .debug import DebugVisualizer
            self.debug_visualizer = DebugVisualizer(self.config.get('output_dir', 'output'))
        
        # Performance metrics
        self.timers = {
            'scene_detection': [],
            'flow_analysis': [],
            'feature_matching': [],
            'homography': [],
            'warping': [],
            'total': []
        }
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'analyzed_frames': 0,
            'detected_scenes': 0,
            'detected_scans': 0,
            'created_panoramas': 0
        }
        
        print("=== MatteExtractor initialization complete ===\n", flush=True)
    
    @property
    def flow_estimator(self):
        """Lazy initialization of flow estimator"""
        if self._flow_estimator is None:
            print("\nInitializing optical flow estimator...")
            try:
                self._flow_estimator = RAFTOpticalFlow(device=self.device)
                print("Flow estimator initialized successfully")
            except Exception as e:
                print(f"Error initializing flow estimator: {e}")
                raise
        return self._flow_estimator
    
    @property
    def scene_detector(self):
        """Lazy initialization of scene detector"""
        if self._scene_detector is None:
            print("\nInitializing scene detector...")
            try:
                self._scene_detector = ContentAwareSceneDetector(
                    device=self.device,
                    debug_visualizer=self.debug_visualizer
                )
                print("Scene detector initialized successfully")
            except Exception as e:
                print(f"Error initializing scene detector: {e}")
                raise
        return self._scene_detector
    
    @property
    def matcher(self):
        """Lazy initialization of feature matcher"""
        if self._matcher is None:
            print("\nInitializing feature matcher (LoFTR)...")
            try:
                self._matcher = kornia.feature.LoFTR(pretrained='outdoor')
                self._matcher = self._matcher.to(self.device)
                self._matcher.eval()
                print("Feature matcher initialized successfully")
                print(f"CUDA memory after matcher init: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            except Exception as e:
                print(f"Error initializing feature matcher: {e}")
                raise
        return self._matcher
            
    def process_video(self, video_path, output_dir=None, sample_rate=1):
        """
        Process a video file and extract matte paintings.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save output panoramas
            sample_rate: Process every Nth frame
            
        Returns:
            panoramas: List of extracted panoramas
        """
        start_time = time.time()
        
        # Create debug directory if needed
        debug_dir = None
        if self.config['debug_output'] and output_dir:
            debug_dir = os.path.join(output_dir, 'debug')
            os.makedirs(debug_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Update statistics
        self.stats['total_frames'] = total_frames
        
        # Clear any existing frames in queues
        self.frame_queue.clear_queues()
        
        # Update progress
        if self.config['progress_callback']:
            self.config['progress_callback'].update(
                f"Stage 1/3: Scene Detection | Frames: 0/{total_frames} | Scenes: 0"
            )
        
        # Process frames
        frame_count = 0
        current_scene_id = 0
        last_frame = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Update progress every 10 frames
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                self._update_progress(1, progress, 
                    f"Scene Detection | Frame {frame_count}/{total_frames}")
            
            # Skip frames according to sample rate
            if frame_count % sample_rate != 0:
                continue
            
            # Add frame to queue
            self.frame_queue.add_frame(frame)
            
            # Skip scene detection for first frame
            if last_frame is None:
                last_frame = frame.copy()
                continue
            
            # Process detection queue
            while len(self.frame_queue.detection_queue) > 0:
                workunit = self.frame_queue.detection_queue[0]
                
                # Check for scene change
                scene_start_time = time.time()
                is_change, similarity = self.scene_detector.is_scene_change(
                    last_frame, workunit.frame, workunit.frame_idx)
                self.timers['scene_detection'].append(time.time() - scene_start_time)
                
                # Update scene info
                if is_change:
                    current_scene_id += 1
                    workunit.scene_info.scene_id = current_scene_id
                    workunit.set_scene_boundary('start', similarity)
                else:
                    workunit.scene_info.scene_id = current_scene_id
                    workunit.set_scene_boundary('middle', similarity)
                
                # Update temporal context
                workunit.update_temporal_context(
                    prev_scene=current_scene_id-1 if current_scene_id > 0 else None,
                    next_scene=None,  # Will be updated when next scene is detected
                    position=frame_count / total_frames  # Position based on progress
                )
                
                # Always promote to flow queue for testing
                self.frame_queue.promote_to_flow(workunit, similarity)
                
                # Remove from detection queue
                self.frame_queue.detection_queue.popleft()
                
                # Update last frame
                last_frame = workunit.frame.copy()
            
            # Process flow queue
            while len(self.frame_queue.flow_queue) > 0:
                workunit = self.frame_queue.flow_queue[0]
                
                # Get previous frame from detection queue if available
                prev_frame = None
                if len(self.frame_queue.detection_queue) > 0:
                    prev_frame = self.frame_queue.detection_queue[-1].frame
                elif len(self.frame_queue.flow_queue) > 1:
                    prev_frame = self.frame_queue.flow_queue[1].frame
                
                if prev_frame is not None:
                    # Compute optical flow
                    flow_start = time.time()
                    # Convert frames to tensors with batch dimension
                    prev_tensor = torch.from_numpy(prev_frame).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
                    curr_tensor = torch.from_numpy(workunit.frame).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
                    flow_result = self.flow_estimator(prev_tensor, curr_tensor)
                    self.timers['flow_analysis'].append(time.time() - flow_start)
                    
                    # Update scene features with flow information
                    workunit.update_scene_feature('flow_magnitude', flow_result['magnitude'])
                    workunit.update_scene_feature('flow_vector', flow_result['vector'])
                    
                    # Check if this is a scanning/panning shot
                    if flow_result['magnitude'] > self.config['scan_threshold']:
                        # Update confidence based on flow consistency
                        flow_consistency = self._check_flow_consistency(flow_result['vector'])
                        workunit.confidence = flow_consistency
                        
                        # Always promote to stitch queue for testing
                        self.frame_queue.promote_to_stitch(workunit, flow_consistency)
                
                # Remove from flow queue
                self.frame_queue.flow_queue.popleft()
            
            # Process stitch queue when it reaches capacity or end of scene
            if (len(self.frame_queue.stitch_queue) >= self.config['max_frames_for_stitch'] or
                frame_count == total_frames):
                print(f"\nCreating panorama with {len(self.frame_queue.stitch_queue)} frames...")
                panorama = self._create_panorama_from_queue()
                if panorama is not None:
                    # Save panorama
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        panorama_path = os.path.join(output_dir, f"panorama_{current_scene_id}.jpg")
                        cv2.imwrite(panorama_path, panorama)
                        print(f"  Saved panorama to {panorama_path}")
        
        # Process any remaining frames in stitch queue
        if len(self.frame_queue.stitch_queue) > 0:
            print(f"\nCreating final panorama with {len(self.frame_queue.stitch_queue)} frames...")
            panorama = self._create_panorama_from_queue()
            if panorama is not None and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                panorama_path = os.path.join(output_dir, "panorama_final.jpg")
                cv2.imwrite(panorama_path, panorama)
                print(f"  Saved final panorama to {panorama_path}")
        
        # Print queue statistics
        print("\nQueue Statistics:")
        print(self.frame_queue)
        
        # Print performance metrics
        total_time = time.time() - start_time
        self.timers['total'].append(total_time)
        
        print("\nPerformance Metrics:")
        print(f"Total processing time: {total_time:.2f}s")
        if self.timers['scene_detection']:
            print(f"Average scene detection time: {np.mean(self.timers['scene_detection']):.4f}s")
        if self.timers['flow_analysis']:
            print(f"Average flow analysis time: {np.mean(self.timers['flow_analysis']):.4f}s")
            
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        return self.frame_queue.get_processing_stats()

    def _create_panorama_from_queue(self):
        """Create a panorama from frames in the stitch queue."""
        if len(self.frame_queue.stitch_queue) < 1:  # Only require 1 frame for testing
            return None
        
        # Get frames from stitch queue
        frames = [unit.frame for unit in self.frame_queue.stitch_queue]
        flow_data = {
            'vector': [unit.scene_info.features.get('flow_vector', [0, 0]) for unit in self.frame_queue.stitch_queue],
            'magnitude': [unit.scene_info.features.get('flow_magnitude', 0) for unit in self.frame_queue.stitch_queue]
        }
        
        # For testing, just concatenate the frames horizontally
        panorama = np.hstack(frames)
        
        # Clear stitch queue
        self.frame_queue.stitch_queue.clear()
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        return panorama

    def _check_flow_consistency(self, movements):
        """
        Check if the flow vectors are consistent in direction.
        
        Args:
            movements: Single movement vector or list of movement vectors
            
        Returns:
            consistency: Ratio of consistent direction vectors
        """
        # Convert single vector to list
        if not isinstance(movements, list):
            movements = [movements]
            
        if not movements or len(movements) < 1:
            return 0.0
            
        # Calculate average direction
        avg_movement = np.mean(movements, axis=0)
        avg_direction = avg_movement / (np.linalg.norm(avg_movement) + 1e-6)
        
        # Count consistent directions
        consistent_count = 0
        for movement in movements:
            # Normalize the movement vector
            norm = np.linalg.norm(movement)
            if norm > 1e-6:  # Avoid division by zero
                direction = movement / norm
                # Calculate dot product with average direction
                alignment = np.dot(direction, avg_direction)
                if alignment > 0.7:  # Vectors pointing in similar direction
                    consistent_count += 1
        
        return consistent_count / len(movements)

    def create_panorama(self, start_frame, end_frame, flow_data, debug_dir=None):
        """
        Create a panorama from a sequence of frames and their movement vectors.
        
        Args:
            start_frame: Start frame of the scan sequence
            end_frame: End frame of the scan sequence
            flow_data: Optical flow data for the scan sequence
            debug_dir: Directory to save debug information
            
        Returns:
            panorama: Stitched panorama image
        """
        if end_frame - start_frame < 3:
            print("Not enough frames for panorama")
            return None
            
        # Use a subset of frames for efficiency if there are too many
        if end_frame - start_frame > self.config['max_scan_frames']:
            # Sample frames evenly
            indices = np.linspace(start_frame, end_frame-1, self.config['max_scan_frames'], dtype=int)
            frames = [flow_data['frame'][i] for i in indices]
            
            # Adjust movements to match sampled frames
            if flow_data['vector'] and len(flow_data['vector']) >= len(indices) - 1:
                # Calculate indices for movements (which are between frames)
                movement_indices = []
                for i in range(len(indices) - 1):
                    # Find the average movement between these sampled frames
                    start_idx = indices[i]
                    end_idx = indices[i+1]
                    if start_idx < len(flow_data['vector']) and end_idx - 1 < len(flow_data['vector']):
                        # Average the movements between these frames
                        avg_movement = np.mean(flow_data['vector'][start_idx:end_idx], axis=0)
                        movement_indices.append(avg_movement)
                
                flow_data['vector'] = movement_indices
            
        # Convert frames to grayscale for feature matching
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        
        # Initialize panorama with the first frame
        result = frames[0].copy()
        h, w = frames[0].shape[:2]
        
        # Estimate total panorama size from movement vectors
        if flow_data['vector']:
            total_x = np.sum([m[0] for m in flow_data['vector']])
            total_y = np.sum([m[1] for m in flow_data['vector']])
            
            # Create panorama canvas with estimated size
            panorama_w = w + int(abs(total_x)) + 100  # Add margin
            panorama_h = h + int(abs(total_y)) + 100
            
            # Initialize panorama with black background
            panorama = np.zeros((panorama_h, panorama_w, 3), dtype=np.uint8)
            
            # Place first frame in the center
            start_x = panorama_w // 2 - w // 2
            start_y = panorama_h // 2 - h // 2
            
            panorama[start_y:start_y+h, start_x:start_x+w] = frames[0]
            
            # Current position
            curr_x, curr_y = start_x, start_y
            
            # Save first frame for debugging
            if debug_dir:
                timestamp = int(time.time())
                cv2.imwrite(f"{debug_dir}/frame_0_{timestamp}.jpg", frames[0])
            
            # Process each subsequent frame
            for i in range(1, len(frames)):
                # Get current frame
                frame = frames[i]
                
                # Save frame for debugging
                if debug_dir:
                    cv2.imwrite(f"{debug_dir}/frame_{i}_{timestamp}.jpg", frame)
                
                # Convert frames to tensors for LoFTR
                feature_start = time.time()
                img1 = torch.from_numpy(gray_frames[i-1]).float() / 255.0
                img2 = torch.from_numpy(gray_frames[i]).float() / 255.0
                
                # Add batch and channel dimensions
                img1 = img1.unsqueeze(0).unsqueeze(0).to(self.device)
                img2 = img2.unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Match features using LoFTR
                with torch.no_grad():
                    correspondences = self.matcher({'image0': img1, 'image1': img2})
                    
                # Get matched keypoints
                mkpts0 = correspondences['keypoints0'].cpu().numpy()
                mkpts1 = correspondences['keypoints1'].cpu().numpy()
                
                # Need at least 4 points for homography
                if len(mkpts0) < 4:
                    print(f"Not enough matches between frames {i-1} and {i}")
                    continue
                    
                # Save feature matches for debugging
                if debug_dir:
                    match_img = np.hstack([gray_frames[i-1], gray_frames[i]])
                    match_img = cv2.cvtColor(match_img, cv2.COLOR_GRAY2BGR)
                    
                    # Draw matches
                    for pt1, pt2 in zip(mkpts0, mkpts1):
                        pt2[0] += gray_frames[i-1].shape[1]  # Adjust x-coordinate for the second image
                        cv2.line(match_img, tuple(map(int, pt1)), tuple(map(int, pt2)), (0, 255, 0), 1)
                        cv2.circle(match_img, tuple(map(int, pt1)), 3, (0, 0, 255), -1)
                        cv2.circle(match_img, tuple(map(int, pt2)), 3, (0, 0, 255), -1)
                        
                    cv2.imwrite(f"{debug_dir}/matches_{i-1}_{i}_{timestamp}.jpg", match_img)
                
                self.timers['feature_matching'].append(time.time() - feature_start)
                
                # Estimate homography
                homography_start = time.time()
                H, mask = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
                self.timers['homography'].append(time.time() - homography_start)
                
                # Check inlier ratio
                if mask is not None:
                    inlier_ratio = np.sum(mask) / len(mask)
                    if inlier_ratio < self.config['min_inlier_ratio']:
                        print(f"Low inlier ratio ({inlier_ratio:.2f}) between frames {i-1} and {i}")
                        continue
                
                if H is None:
                    print(f"Failed to find homography between frames {i-1} and {i}")
                    continue
                    
                # Warp current frame
                warp_start = time.time()
                warped = cv2.warpPerspective(frame, H, (panorama_w, panorama_h))
                self.timers['warping'].append(time.time() - warp_start)
                
                # Save warped frame for debugging
                if debug_dir:
                    cv2.imwrite(f"{debug_dir}/warped_{i}_{timestamp}.jpg", warped)
                
                # Create mask for current panorama
                mask = np.zeros((panorama_h, panorama_w), dtype=np.uint8)
                mask[curr_y:curr_y+h, curr_x:curr_x+w] = 255
                
                # Update panorama with warped frame - vectorized version for speed
                warped_mask = (warped.sum(axis=2) > 0).astype(np.uint8)
                current_mask = mask > 0
                
                # Areas with only warped content
                warped_only = np.logical_and(warped_mask, np.logical_not(current_mask))
                panorama[warped_only] = warped[warped_only]
                
                # Overlapping areas - blend
                overlap = np.logical_and(warped_mask, current_mask)
                panorama[overlap] = (0.5 * panorama[overlap] + 0.5 * warped[overlap]).astype(np.uint8)
                
                # Update mask
                mask = np.logical_or(mask, warped_mask).astype(np.uint8) * 255
                
                # Update current position based on movement vector
                if i < len(flow_data['vector']):
                    curr_x += int(flow_data['vector'][i-1][0])
                    curr_y += int(flow_data['vector'][i-1][1])
            
            # Crop the panorama to remove black borders
            # Find the bounding box of non-black pixels
            non_black = np.where(panorama.sum(axis=2) > 0)
            if len(non_black[0]) > 0:
                y_min, y_max = np.min(non_black[0]), np.max(non_black[0])
                x_min, x_max = np.min(non_black[1]), np.max(non_black[1])
                
                # Add small margin
                margin = 10
                y_min = max(0, y_min - margin)
                y_max = min(panorama_h - 1, y_max + margin)
                x_min = max(0, x_min - margin)
                x_max = min(panorama_w - 1, x_max + margin)
                
                # Crop
                panorama = panorama[y_min:y_max+1, x_min:x_max+1]
                
                # Save final panorama for debugging
                if debug_dir:
                    cv2.imwrite(f"{debug_dir}/final_panorama_{timestamp}.jpg", panorama)
            
            return panorama
        else:
            # No movement data, just return the first frame
            return frames[0].copy() 

    def _update_progress(self, stage, stage_progress, status_text):
        """Update progress bar and status text."""
        if not self.config['progress_callback']:
            return
            
        # Calculate overall progress (each stage is worth 33.33%)
        stage_weight = 33.33
        overall_progress = ((stage - 1) * stage_weight) + (stage_progress * stage_weight / 100)
        
        # Create progress bar
        width = 30
        filled = int(width * overall_progress / 100)
        bar = '=' * filled + '>' + '-' * (width - filled - 1)
        
        # Update status with progress bar
        status = f"[{bar}] {overall_progress:5.1f}% | Stage {stage}/3 | {status_text}"
        self.config['progress_callback'].update(status)
        
        # Store progress
        self.config['last_progress'] = overall_progress
        self.config['current_stage'] = stage
        self.config['stage_progress'] = stage_progress 