import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import functional as TF

class ContentAwareSceneDetector:
    """
    Scene change detection with content awareness for animation.
    
    Uses a combination of perceptual features and instance segmentation
    to accurately detect scene changes while ignoring character movement.
    """
    
    def __init__(self, segmentation_model=None, device=None, debug_visualizer=None):
        """
        Initialize the scene detector.
        
        Args:
            segmentation_model: Pre-trained instance segmentation model
            device: torch device
            debug_visualizer: DebugVisualizer instance for debugging
        """
        print("\n=== Initializing Scene Detector ===", flush=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}", flush=True)
        if torch.cuda.is_available():
            print(f"CUDA memory before ResNet: {torch.cuda.memory_allocated() / 1024**2:.1f} MB", flush=True)
        
        # Store segmentation model and debug visualizer
        self.segmentation_model = segmentation_model
        self.debug_visualizer = debug_visualizer
        
        # Initialize similarity tracking
        self.frame_features = []
        self.frame_indices = []
        self.similarity_matrix = None
        
        # Scene detection parameters
        self.params = {
            'candidate_threshold': 0.8,  # Lower threshold for candidate changes
            'confirm_threshold': 0.7,    # Higher threshold for confirmed changes
            'temporal_window': 3,        # Number of frames to analyze for temporal consistency
            'min_scene_length': 24,      # Minimum scene length (1 second at 24fps)
            'max_scene_length': 7200,    # Maximum scene length (5 minutes at 24fps)
        }
        
        # Load feature extractor
        try:
            import torchvision.models as models
            print("\nStep 1: Downloading ResNet18 weights...", flush=True)
            self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT, progress=True)
            print("ResNet18 weights downloaded successfully", flush=True)
            
            print("\nStep 2: Modifying network architecture...", flush=True)
            # Remove classification layer and modify for feature extraction
            modules = list(self.feature_extractor.children())[:-2]
            self.feature_extractor = torch.nn.Sequential(*modules)
            print("Network architecture modified", flush=True)
            
            print("\nStep 3: Moving model to device and testing...", flush=True)
            print(f"Moving model to {self.device}", flush=True)
            self.feature_extractor = self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
            
            if torch.cuda.is_available():
                print(f"CUDA memory after model to device: {torch.cuda.memory_allocated() / 1024**2:.1f} MB", flush=True)
            
            # Test the feature extractor
            print("\nStep 4: Testing feature extractor...", flush=True)
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)
                _ = self.feature_extractor(dummy_input)
            print("Feature extractor test successful", flush=True)
            
            if torch.cuda.is_available():
                print(f"Final CUDA memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB", flush=True)
            
        except ImportError:
            print("\nError: torchvision is required for feature extraction", flush=True)
            raise ImportError("torchvision is required for feature extraction")
        except Exception as e:
            print(f"\nError initializing feature extractor: {e}", flush=True)
            raise
            
        # Load segmentation model if provided
        if self.segmentation_model is not None:
            print("\nStep 5: Initializing segmentation model...", flush=True)
            try:
                self.segmentation_model.to(self.device)
                self.segmentation_model.eval()
                print("Segmentation model loaded successfully", flush=True)
            except Exception as e:
                print(f"\nError initializing segmentation model: {e}", flush=True)
                raise
                
        print("=== Scene Detector initialization complete ===\n", flush=True)
            
    def extract_features(self, frame):
        """
        Extract deep features from a frame.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            features: Deep features from ResNet
        """
        # Convert to tensor
        if isinstance(frame, np.ndarray):
            # Convert BGR to RGB if needed
            if frame.shape[2] == 3:
                frame = frame[..., ::-1].copy()
            
            # Convert to tensor and normalize
            frame_tensor = TF.to_tensor(frame)
            frame_tensor = TF.normalize(frame_tensor, 
                                       mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
            frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(frame_tensor)
            
            # Save feature visualization if debug visualizer is available
            if self.debug_visualizer:
                self.debug_visualizer.save_frame_features(len(self.frame_indices), features, frame)
            
        return features
    
    def remove_foreground(self, frame):
        """
        Remove foreground objects (characters) from a frame.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            bg_frame: Frame with foreground objects removed
        """
        if self.segmentation_model is None:
            return frame
        
        # Convert to tensor
        if isinstance(frame, np.ndarray):
            # BGR to RGB if needed
            if frame.shape[2] == 3:
                frame_rgb = frame[..., ::-1].copy()
            
            # Convert to tensor
            frame_tensor = TF.to_tensor(frame_rgb).unsqueeze(0).to(self.device)
        
        # Run segmentation
        with torch.no_grad():
            predictions = self.segmentation_model(frame_tensor)
            
        # Get masks for character classes (person, animal, etc.)
        masks = self._get_character_masks(predictions)
        
        # Create background-only frame
        bg_frame = frame.copy()
        if len(masks) > 0:
            # Use simple inpainting for removed regions
            for mask in masks:
                mask_np = mask.cpu().numpy().astype(np.uint8)
                bg_frame = self._simple_inpaint(bg_frame, mask_np)
        
        return bg_frame
    
    def _get_character_masks(self, predictions):
        """Extract masks for character classes from segmentation predictions."""
        # This implementation will depend on the specific segmentation model used
        # For Detectron2 or Mask R-CNN, we would extract masks for relevant classes
        # Placeholder implementation
        return []
    
    def _simple_inpaint(self, frame, mask):
        """Simple inpainting by averaging nearby pixels."""
        # Placeholder implementation
        # In a real implementation, we'd use more sophisticated inpainting
        return frame
    
    def compute_similarity(self, features1, features2):
        """
        Compute similarity between two feature sets.
        
        Args:
            features1: First feature tensor
            features2: Second feature tensor
            
        Returns:
            similarity: Cosine similarity score
        """
        # Flatten features
        f1 = features1.flatten()
        f2 = features2.flatten()
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(f1, f2, dim=0)
        
        return similarity.item()
    
    def is_scene_change(self, frame1, frame2, frame_idx=None):
        """
        Detect if there's a scene change between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            frame_idx: Current frame index for debugging
            
        Returns:
            is_change: True if scene change detected
            similarity: Similarity score between frames
        """
        # Extract features
        features1 = self.extract_features(frame1)
        features2 = self.extract_features(frame2)
        
        # Store features for similarity matrix
        if frame_idx is not None:
            self.frame_features.append(features2.cpu())
            self.frame_indices.append(frame_idx)
            
            # Update similarity matrix if we have enough frames
            if len(self.frame_features) >= self.params['temporal_window']:
                self._update_similarity_matrix()
                
                # Clear old features to save memory
                self.frame_features = self.frame_features[-self.params['temporal_window']:]
                self.frame_indices = self.frame_indices[-self.params['temporal_window']:]
                
                # Clear GPU memory
                torch.cuda.empty_cache()
        
        # Compute similarity
        similarity = self.compute_similarity(features1, features2)
        
        # Check for scene change using dual threshold approach
        is_candidate = similarity < self.params['candidate_threshold']
        is_confirmed = similarity < self.params['confirm_threshold']
        
        # Use temporal window if available
        if len(self.frame_features) >= self.params['temporal_window']:
            is_change = self._check_temporal_consistency(is_candidate, is_confirmed)
        else:
            is_change = is_confirmed
        
        # Save debug visualization
        if self.debug_visualizer:
            self.debug_visualizer.save_scene_boundary(
                frame_idx if frame_idx is not None else -1,
                frame2,
                is_change,
                similarity
            )
        
        return is_change, similarity
    
    def _update_similarity_matrix(self):
        """Update the similarity matrix with current frame features."""
        n_frames = len(self.frame_features)
        
        # Create or resize similarity matrix
        if self.similarity_matrix is None:
            self.similarity_matrix = torch.zeros((n_frames, n_frames))
        elif self.similarity_matrix.size(0) < n_frames:
            # Create new larger matrix
            new_matrix = torch.zeros((n_frames, n_frames))
            # Copy old values
            old_size = self.similarity_matrix.size(0)
            new_matrix[:old_size, :old_size] = self.similarity_matrix
            self.similarity_matrix = new_matrix
        
        # Compute similarities for the new frame
        latest_features = self.frame_features[-1]
        for i in range(n_frames-1):
            sim = self.compute_similarity(self.frame_features[i], latest_features)
            self.similarity_matrix[i, -1] = sim
            self.similarity_matrix[-1, i] = sim
        
        self.similarity_matrix[-1, -1] = 1.0
        
        # Save visualization
        if self.debug_visualizer and n_frames % 10 == 0:  # Update every 10 frames
            self.debug_visualizer.save_similarity_matrix(
                self.frame_indices,
                self.similarity_matrix.cpu().numpy()
            )
    
    def _check_temporal_consistency(self, is_candidate, is_confirmed):
        """
        Check temporal consistency of scene change detection.
        
        Args:
            is_candidate: Whether current frame is a scene change candidate
            is_confirmed: Whether current frame meets confirmed threshold
            
        Returns:
            is_change: Whether a scene change is confirmed
        """
        if not is_candidate:
            return False
            
        # Get recent similarity scores
        recent_sims = self.similarity_matrix[-self.params['temporal_window']:, -1]
        
        # Check if we have a clear drop in similarity
        pre_change = torch.mean(recent_sims[:-1])
        post_change = recent_sims[-1]
        
        # Confirm scene change if:
        # 1. Current frame meets confirmed threshold
        # 2. Previous frames were relatively similar to each other
        # 3. There's a significant drop in similarity
        return (is_confirmed and 
                pre_change > self.params['candidate_threshold'] and
                post_change < self.params['confirm_threshold'])
    
    def finalize_debug(self):
        """Save final debug visualizations."""
        if self.debug_visualizer and self.similarity_matrix is not None:
            self.debug_visualizer.save_similarity_matrix(
                self.frame_indices,
                self.similarity_matrix.cpu().numpy()
            ) 