"""
Lightweight profiling tools for performance optimization.
"""

import time
from contextlib import contextmanager
from collections import defaultdict
import torch
import os
from pathlib import Path
import json
from datetime import datetime

class PipelineProfiler:
    def __init__(self, output_dir=None, enabled=True):
        self.enabled = enabled
        if not enabled:
            return
            
        self.stage_times = defaultdict(list)
        self.memory_stats = defaultdict(list)
        self.gpu_stats = defaultdict(list)
        self.metrics = defaultdict(list)
        
        # Setup output directory
        self.output_dir = Path(output_dir) if output_dir else Path('profiler_output')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session identifier
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initial system state
        self.initial_state = {
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A',
            'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        }
        
    @contextmanager
    def profile_stage(self, stage_name):
        """Profile a specific pipeline stage."""
        if not self.enabled:
            yield
            return
            
        # Ensure GPU operations are finished
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Record initial state
        start = time.perf_counter()
        gpu_mem_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        gpu_reserved_start = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        
        try:
            yield
        finally:
            # Ensure GPU operations are finished
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            # Record final state
            end = time.perf_counter()
            gpu_mem_end = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            gpu_reserved_end = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            
            # Store metrics
            duration = end - start
            self.stage_times[stage_name].append(duration)
            
            # Store memory stats
            if torch.cuda.is_available():
                self.memory_stats[stage_name].append({
                    'gpu_mem_delta': gpu_mem_end - gpu_mem_start,
                    'gpu_reserved_delta': gpu_reserved_end - gpu_reserved_start
                })
                
                # Log peak memory for this stage
                self.gpu_stats[stage_name].append({
                    'peak_memory': torch.cuda.max_memory_allocated(),
                    'peak_reserved': torch.cuda.max_memory_reserved(),
                    'current_memory': gpu_mem_end,
                    'current_reserved': gpu_reserved_end
                })
                
    def log_metric(self, name, value):
        """Log a custom metric."""
        if self.enabled:
            self.metrics[name].append(value)
            
    def get_stage_summary(self, stage_name):
        """Get summary statistics for a stage."""
        if not self.enabled or stage_name not in self.stage_times:
            return None
            
        times = self.stage_times[stage_name]
        return {
            'count': len(times),
            'mean_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'total_time': sum(times)
        }
        
    def get_memory_summary(self):
        """Get memory usage summary."""
        if not self.enabled:
            return None
            
        if torch.cuda.is_available():
            return {
                'current_gpu': torch.cuda.memory_allocated(),
                'peak_gpu': torch.cuda.max_memory_allocated(),
                'current_reserved': torch.cuda.memory_reserved(),
                'peak_reserved': torch.cuda.max_memory_reserved()
            }
        else:
            return {}
            
    def save_session(self):
        """Save profiling data to disk."""
        if not self.enabled:
            return
            
        data = {
            'session_id': self.session_id,
            'initial_state': self.initial_state,
            'stage_times': {k: v for k, v in self.stage_times.items()},
            'memory_stats': {k: v for k, v in self.memory_stats.items()},
            'gpu_stats': {k: v for k, v in self.gpu_stats.items()},
            'metrics': {k: v for k, v in self.metrics.items()}
        }
        
        output_file = self.output_dir / f'profile_{self.session_id}.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def print_summary(self):
        """Print a summary of profiling data."""
        if not self.enabled:
            return
            
        print("\n=== Pipeline Profile Summary ===")
        print(f"Session ID: {self.session_id}")
        
        print("\nStage Timing Summary:")
        for stage in sorted(self.stage_times.keys()):
            summary = self.get_stage_summary(stage)
            print(f"\n{stage}:")
            print(f"  Count: {summary['count']}")
            print(f"  Mean Time: {summary['mean_time']:.4f}s")
            print(f"  Min Time: {summary['min_time']:.4f}s")
            print(f"  Max Time: {summary['max_time']:.4f}s")
            print(f"  Total Time: {summary['total_time']:.4f}s")
            
        print("\nMemory Usage Summary:")
        mem_summary = self.get_memory_summary()
        if torch.cuda.is_available():
            print(f"  Current GPU Memory: {mem_summary['current_gpu'] / 1024**2:.1f}MB")
            print(f"  Peak GPU Memory: {mem_summary['peak_gpu'] / 1024**2:.1f}MB")
            print(f"  Current Reserved Memory: {mem_summary['current_reserved'] / 1024**2:.1f}MB")
            print(f"  Peak Reserved Memory: {mem_summary['peak_reserved'] / 1024**2:.1f}MB")
        
        print("\nCustom Metrics:")
        for name, values in self.metrics.items():
            if values:
                mean_val = sum(values) / len(values)
                print(f"  {name}: {mean_val:.4f} (mean of {len(values)} samples)")
                
        print("\n=== End Summary ===\n") 