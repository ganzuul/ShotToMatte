import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models.optical_flow as tv_flow

class RAFTOpticalFlow:
    """
    Optical flow estimation using the RAFT neural network model.
    
    RAFT: Recurrent All-Pairs Field Transforms for Optical Flow
    https://arxiv.org/abs/2003.12039
    """
    
    def __init__(self, device=None):
        """
        Initialize the RAFT optical flow model.
        
        Args:
            device: torch device (will use CUDA if available)
        """
        print("\n=== Initializing RAFT Optical Flow ===", flush=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}", flush=True)
        if torch.cuda.is_available():
            print(f"CUDA memory before RAFT: {torch.cuda.memory_allocated() / 1024**2:.1f} MB", flush=True)
        
        try:
            # Load model
            print("\nStep 1: Downloading RAFT model weights...", flush=True)
            self.model = tv_flow.raft_large(weights=tv_flow.Raft_Large_Weights.DEFAULT, progress=True)
            print("RAFT weights downloaded successfully", flush=True)
            
            print("\nStep 3: Moving model to device and testing...", flush=True)
            print(f"Moving model to {self.device}", flush=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            if torch.cuda.is_available():
                print(f"CUDA memory after model to device: {torch.cuda.memory_allocated() / 1024**2:.1f} MB", flush=True)
            
            # Test the model
            print("\nStep 4: Testing model with dummy input...", flush=True)
            with torch.no_grad():
                # RAFT requires input size to be at least 128x128 due to 8x downsampling
                dummy_input1 = torch.zeros(1, 3, 256, 256).to(self.device)
                dummy_input2 = torch.zeros(1, 3, 256, 256).to(self.device)
                _ = self.model(dummy_input1, dummy_input2)
            print("Model test successful", flush=True)
            
            if torch.cuda.is_available():
                print(f"Final CUDA memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB", flush=True)
            
        except Exception as e:
            print(f"\nError initializing RAFT model: {e}", flush=True)
            raise
        
        print("=== RAFT initialization complete ===\n", flush=True)
        
    def __call__(self, frame1, frame2):
        """
        Compute optical flow between two frames.
        
        Args:
            frame1: First frame tensor [1, 3, H, W]
            frame2: Second frame tensor [1, 3, H, W]
            
        Returns:
            dict containing flow results
        """
        with torch.no_grad():
            # Ensure frames are on correct device
            frame1 = frame1.to(self.device)
            frame2 = frame2.to(self.device)
            
            # Compute flow
            flow_predictions = self.model(frame1, frame2)
            
            # Get final flow prediction (last element in the list)
            flow_pred = flow_predictions[-1]
            
            # Compute flow magnitude
            flow_magnitude = torch.norm(flow_pred, dim=1)
            
            return {
                'flow': flow_pred,
                'magnitude': flow_magnitude.mean().item(),
                'vector': [
                    flow_pred[0, 0].mean().item(),  # x component
                    flow_pred[0, 1].mean().item()   # y component
                ]
            }
        
    def batch_flow(self, frames, batch_size=8):
        """
        Compute optical flow for a sequence of frames in batches.
        
        Args:
            frames: List of frames (numpy arrays)
            batch_size: Number of frame pairs to process in parallel
            
        Returns:
            flows: List of flow results dictionaries
        """
        results = []
        
        # Process frames in batches for efficiency
        for i in range(len(frames) - 1):
            # Determine batch end
            end_idx = min(i + batch_size, len(frames) - 1)
            
            # Create batch of frame pairs
            batch_frames1 = []
            batch_frames2 = []
            
            for j in range(i, end_idx):
                batch_frames1.append(frames[j])
                batch_frames2.append(frames[j+1])
                
            # Convert frames to tensors
            if isinstance(batch_frames1[0], np.ndarray):
                # Convert BGR to RGB if necessary
                batch_tensors1 = []
                batch_tensors2 = []
                
                for frame1, frame2 in zip(batch_frames1, batch_frames2):
                    if frame1.shape[2] == 3:
                        frame1_rgb = frame1[..., ::-1].copy()
                        frame2_rgb = frame2[..., ::-1].copy()
                    else:
                        frame1_rgb = frame1
                        frame2_rgb = frame2
                        
                    # Convert to torch tensors and normalize
                    tensor1 = torch.from_numpy(frame1_rgb).permute(2, 0, 1).float() / 255.0
                    tensor2 = torch.from_numpy(frame2_rgb).permute(2, 0, 1).float() / 255.0
                    
                    batch_tensors1.append(tensor1)
                    batch_tensors2.append(tensor2)
                    
                # Stack tensors into batch
                batch1 = torch.stack(batch_tensors1).to(self.device)
                batch2 = torch.stack(batch_tensors2).to(self.device)
            else:
                # Already tensors
                batch1 = torch.stack(batch_frames1).to(self.device)
                batch2 = torch.stack(batch_frames2).to(self.device)
            
            # Process batch
            with torch.no_grad():
                flow_predictions = self.model(batch1, batch2)
                
            # TorchVision's RAFT returns a list of flow predictions at different scales
            # We use the last one which is the final prediction
            final_flow = flow_predictions[-1]
                
            # Extract individual results
            for b in range(final_flow.size(0)):
                flow = final_flow[b]
                flow_magnitude = torch.sqrt(flow[0, :, :]**2 + flow[1, :, :]**2)
                flow_mean = torch.mean(flow_magnitude)
                flow_vector = torch.mean(flow, dim=(1, 2))  # Average across height, width
                
                results.append({
                    'flow': flow.permute(1, 2, 0).cpu().numpy(),
                    'magnitude': flow_mean.cpu().item(),
                    'vector': flow_vector.cpu().numpy()
                })
                
                # Break early if we have enough results
                if len(results) >= len(frames) - 1:
                    break
                    
        return results 