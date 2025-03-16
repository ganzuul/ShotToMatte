"""
Debug visualization tools for ShotToMatte.

This module provides visualization tools for debugging and analyzing
the scene detection and matte extraction process.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision
from datetime import datetime

class DebugVisualizer:
    def __init__(self, output_dir, enabled=True):
        """
        Initialize debug visualizer.
        
        Args:
            output_dir: Base directory for debug output
            enabled: Whether debug visualization is enabled
        """
        self.enabled = enabled
        if not enabled:
            return
            
        # Create debug directory structure
        self.output_dir = Path(output_dir) / 'debug'
        self.scene_dir = self.output_dir / 'scenes'
        self.feature_dir = self.output_dir / 'features'
        self.similarity_dir = self.output_dir / 'similarity'
        
        # Create directories
        for dir_path in [self.scene_dir, self.feature_dir, self.similarity_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize session timestamp
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize feature visualization parameters
        self.feature_grid_size = (8, 8)  # Show 8x8 grid of feature maps
        
    def save_frame_features(self, frame_idx, features, frame=None):
        """
        Visualize and save feature maps for a frame.
        
        Args:
            frame_idx: Frame index
            features: Feature tensor from ResNet [1, C, H, W]
            frame: Original frame for reference (optional)
        """
        if not self.enabled:
            return
            
        # Get feature maps
        feature_maps = features[0].detach().cpu()  # [C, H, W]
        
        # Calculate grid layout
        n_channels = min(self.feature_grid_size[0] * self.feature_grid_size[1], 
                        feature_maps.size(0))
        
        # Create feature grid
        grid = torchvision.utils.make_grid(
            feature_maps[:n_channels].unsqueeze(1),  # Add channel dim for grayscale
            nrow=self.feature_grid_size[0],
            normalize=True,
            padding=2
        )
        
        # Convert to numpy and scale to 0-255
        grid_np = (grid.numpy() * 255).astype(np.uint8)
        
        # Create visualization
        if frame is not None:
            # Resize grid to match frame height
            h, w = frame.shape[:2]
            grid_h = h
            grid_w = int(grid_np.shape[2] * (h / grid_np.shape[1]))
            grid_resized = cv2.resize(grid_np[0], (grid_w, grid_h))
            
            # Create side-by-side visualization
            vis = np.zeros((h, w + grid_w, 3), dtype=np.uint8)
            vis[:, :w] = frame
            vis[:, w:, 0] = grid_resized  # Use blue channel for feature visualization
            vis[:, w:, 1] = grid_resized
            vis[:, w:, 2] = grid_resized
            
            # Add text
            cv2.putText(vis, f'Frame {frame_idx}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(vis, f'Features ({n_channels} channels)', (w + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Just show the feature grid
            vis = cv2.cvtColor(grid_np[0], cv2.COLOR_GRAY2BGR)
            cv2.putText(vis, f'Frame {frame_idx} Features', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualization
        output_path = self.feature_dir / f'features_{self.session_id}_frame_{frame_idx:06d}.jpg'
        cv2.imwrite(str(output_path), vis)
        
    def save_similarity_matrix(self, frame_indices, similarity_matrix):
        """
        Visualize and save frame similarity matrix.
        
        Args:
            frame_indices: List of frame indices
            similarity_matrix: NxN matrix of frame similarities
        """
        if not self.enabled:
            return
            
        # Convert similarity matrix to visualization
        vis_size = 800
        matrix_vis = cv2.resize((similarity_matrix * 255).astype(np.uint8), 
                              (vis_size, vis_size))
        matrix_vis = cv2.applyColorMap(matrix_vis, cv2.COLORMAP_VIRIDIS)
        
        # Add frame indices
        n_frames = len(frame_indices)
        tick_step = max(1, n_frames // 10)  # Show at most 10 ticks
        
        # Add grid lines
        for i in range(0, vis_size, vis_size // 10):
            cv2.line(matrix_vis, (i, 0), (i, vis_size), (255, 255, 255), 1)
            cv2.line(matrix_vis, (0, i), (vis_size, i), (255, 255, 255), 1)
            
            # Add frame numbers
            frame_idx = frame_indices[int(i * n_frames / vis_size)]
            cv2.putText(matrix_vis, str(frame_idx), (i + 5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(matrix_vis, str(frame_idx), (5, i + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add title
        cv2.putText(matrix_vis, 'Frame Similarity Matrix', (vis_size//4, vis_size-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualization
        output_path = self.similarity_dir / f'similarity_{self.session_id}.jpg'
        cv2.imwrite(str(output_path), matrix_vis)
        
    def save_scene_boundary(self, frame_idx, frame, is_boundary, similarity=None):
        """
        Save frame with scene boundary information.
        
        Args:
            frame_idx: Frame index
            frame: Frame image
            is_boundary: Whether this frame is a scene boundary
            similarity: Similarity score with previous frame (optional)
        """
        if not self.enabled:
            return
            
        # Create visualization
        vis = frame.copy()
        
        # Add red border if this is a boundary
        if is_boundary:
            cv2.rectangle(vis, (0, 0), (vis.shape[1]-1, vis.shape[0]-1), 
                         (0, 0, 255), 5)
        
        # Add text
        text = f'Frame {frame_idx}'
        if similarity is not None:
            text += f' (Similarity: {similarity:.3f})'
        if is_boundary:
            text += ' - SCENE BOUNDARY'
            
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        
        # Save visualization
        output_path = self.scene_dir / f'frame_{self.session_id}_{frame_idx:06d}.jpg'
        cv2.imwrite(str(output_path), vis)
        
    def save_scene_summary(self, scene_frames, scene_idx):
        """
        Save summary visualization of a detected scene.
        
        Args:
            scene_frames: List of frames in the scene
            scene_idx: Scene index
        """
        if not self.enabled or not scene_frames:
            return
            
        # Calculate grid layout
        n_frames = len(scene_frames)
        grid_size = min(5, n_frames)  # Show at most 5x5 grid
        
        # Calculate cell size to maintain aspect ratio
        frame_h, frame_w = scene_frames[0].shape[:2]
        cell_h = 200  # Fixed cell height
        cell_w = int(cell_h * frame_w / frame_h)
        
        # Create grid
        grid_h = grid_size * cell_h
        grid_w = grid_size * cell_w
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # Sample frames evenly
        sample_indices = np.linspace(0, n_frames-1, grid_size*grid_size, dtype=int)
        
        # Place frames in grid
        for idx, frame_idx in enumerate(sample_indices):
            if frame_idx >= len(scene_frames):
                break
                
            frame = scene_frames[frame_idx]
            
            # Calculate grid position
            row = idx // grid_size
            col = idx % grid_size
            
            # Resize frame
            frame_resized = cv2.resize(frame, (cell_w, cell_h))
            
            # Place in grid
            y1 = row * cell_h
            y2 = y1 + cell_h
            x1 = col * cell_w
            x2 = x1 + cell_w
            grid[y1:y2, x1:x2] = frame_resized
            
            # Add frame number
            cv2.putText(grid, f'Frame {frame_idx}', (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add title
        cv2.putText(grid, f'Scene {scene_idx} Summary ({n_frames} frames)',
                   (10, grid_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualization
        output_path = self.scene_dir / f'scene_{self.session_id}_{scene_idx:03d}_summary.jpg'
        cv2.imwrite(str(output_path), grid) 