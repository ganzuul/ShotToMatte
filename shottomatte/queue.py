"""
Frame queue management with starvation detection and monitoring.
"""

from collections import deque
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from .profiler import PipelineProfiler

@dataclass
class DropReason:
    """Detailed information about why a frame was dropped."""
    reason_type: str  # e.g., 'queue_full', 'low_confidence', 'scene_mismatch'
    details: Dict[str, Any] = None  # Additional context about the drop
    threshold: Optional[float] = None  # Relevant threshold that caused the drop
    queue_state: Optional[Dict] = None  # State of the queue at drop time
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

@dataclass
class QueueMetrics:
    """Metrics for queue health monitoring."""
    last_input_time: float
    processed_count: int
    dropped_count: int
    starvation_events: int
    avg_processing_time: float
    queue_latency: float
    drop_reasons: Dict[str, int] = None  # Count of drops by reason
    
    def __post_init__(self):
        if self.drop_reasons is None:
            self.drop_reasons = {}
    
    def record_drop(self, reason: DropReason):
        """Record a drop with its detailed reason."""
        self.drop_reasons[reason.reason_type] = self.drop_reasons.get(reason.reason_type, 0) + 1

@dataclass
class SceneContext:
    """Temporal context for a frame within a scene."""
    prev_scene_id: Optional[int] = None
    next_scene_id: Optional[int] = None
    frame_position: float = 0.0  # 0.0 to 1.0 in scene
    
    def __post_init__(self):
        if not 0.0 <= self.frame_position <= 1.0:
            raise ValueError("frame_position must be between 0.0 and 1.0")

@dataclass
class SceneInfo:
    """Detailed scene information for a frame."""
    scene_id: Optional[int] = None
    confidence: float = 0.0
    features: Dict[str, Any] = None
    temporal_context: Optional[SceneContext] = None
    feature_timestamps: Dict[str, float] = None  # When each feature was computed
    
    def __init__(self, 
                 scene_id: Optional[int] = None,
                 confidence: float = 0.0,
                 features: Dict[str, Any] = None,
                 boundary_type: Optional[str] = None,
                 temporal_context: Optional[SceneContext] = None,
                 feature_timestamps: Dict[str, float] = None):
        self.scene_id = scene_id
        self.confidence = confidence
        self.features = features if features is not None else {}
        self.temporal_context = temporal_context if temporal_context is not None else SceneContext()
        self.feature_timestamps = feature_timestamps if feature_timestamps is not None else {}
        self._boundary_type = None
        if boundary_type is not None:
            self.boundary_type = boundary_type  # Use property setter for validation
            
    def update_feature(self, name: str, value: Any):
        """Add or update a feature with timestamp."""
        self.features[name] = value
        self.feature_timestamps[name] = time.perf_counter()
        
    def get_feature_age(self, name: str) -> Optional[float]:
        """Get age of a feature in seconds."""
        timestamp = self.feature_timestamps.get(name)
        if timestamp is None:
            return None
        return time.perf_counter() - timestamp
        
    def is_boundary(self) -> bool:
        """Check if frame is a scene boundary."""
        return self._boundary_type in ['start', 'end']
        
    def is_complete(self) -> bool:
        """Check if scene info is complete enough for processing."""
        return (
            self.scene_id is not None
            and self.confidence > 0.0
            and len(self.features) > 0
            and self._boundary_type is not None
        )
        
    @property
    def boundary_type(self) -> Optional[str]:
        """Get boundary type."""
        return self._boundary_type
        
    @boundary_type.setter
    def boundary_type(self, value: Optional[str]):
        """Set boundary type with validation."""
        if value is not None and value not in ['start', 'end', 'middle']:
            raise ValueError("boundary_type must be one of: 'start', 'end', 'middle'")
        self._boundary_type = value

@dataclass
class WorkUnit:
    """A unit of work moving through the pipeline."""
    frame_idx: int
    timestamp: float
    frame: np.ndarray
    features: Optional[Dict] = None
    confidence: float = 0.0
    stage_history: List[Tuple[str, float]] = None
    dropped_reason: Optional[DropReason] = None
    scene_info: Optional[SceneInfo] = None
    
    def __post_init__(self):
        if self.stage_history is None:
            self.stage_history = []
        if self.scene_info is None:
            self.scene_info = SceneInfo()
            
    def add_stage_time(self, stage_name: str, duration: float):
        """Record time spent in a processing stage."""
        self.stage_history.append((stage_name, duration))
        
    def total_processing_time(self) -> float:
        """Get total time spent processing this work unit."""
        return sum(duration for _, duration in self.stage_history)
    
    def mark_as_dropped(self, reason_type: str, details: Dict[str, Any] = None, 
                       threshold: float = None, queue_state: Dict = None):
        """Mark this work unit as dropped with detailed reason."""
        self.dropped_reason = DropReason(
            reason_type=reason_type,
            details=details,
            threshold=threshold,
            queue_state=queue_state
        )
        
    def update_scene_feature(self, name: str, value: Any):
        """Update a scene-specific feature."""
        self.scene_info.update_feature(name, value)
        
    def set_scene_boundary(self, boundary_type: str, confidence: float):
        """Mark frame as a scene boundary."""
        if boundary_type not in ['start', 'end', 'middle']:
            raise ValueError("boundary_type must be one of: 'start', 'end', 'middle'")
        self.scene_info.boundary_type = boundary_type
        self.scene_info.confidence = confidence
        
    def update_temporal_context(self, 
                              prev_scene: Optional[int] = None,
                              next_scene: Optional[int] = None,
                              position: Optional[float] = None):
        """Update temporal context information."""
        if prev_scene is not None:
            self.scene_info.temporal_context.prev_scene_id = prev_scene
        if next_scene is not None:
            self.scene_info.temporal_context.next_scene_id = next_scene
        if position is not None:
            self.scene_info.temporal_context.frame_position = position
            
    def is_ready_for_flow(self) -> bool:
        """Check if work unit is ready for flow processing."""
        return (
            self.scene_info.is_complete()
            and self.scene_info.confidence >= 0.5  # Minimum confidence threshold
            and not self.dropped_reason
        )
        
    def is_ready_for_stitch(self) -> bool:
        """Check if work unit is ready for stitch processing."""
        return (
            self.is_ready_for_flow()
            and self.confidence >= 0.7  # Higher confidence for stitching
            and self.scene_info.temporal_context.frame_position > 0.0
        )
        
    def get_scene_summary(self) -> Dict[str, Any]:
        """Get a summary of scene-related information."""
        return {
            'frame_idx': self.frame_idx,
            'scene_id': self.scene_info.scene_id,
            'boundary_type': self.scene_info.boundary_type,
            'confidence': self.scene_info.confidence,
            'position': self.scene_info.temporal_context.frame_position,
            'feature_count': len(self.scene_info.features),
            'processing_time': self.total_processing_time()
        }

class DynamicQueueSizer:
    """Handles dynamic adjustment of queue sizes based on performance metrics."""
    
    def __init__(self, 
                 min_size: int = 5,
                 max_size: int = 100,
                 adjustment_interval: float = 5.0,
                 drop_threshold: float = 0.1):
        self.min_size = min_size
        self.max_size = max_size
        self.adjustment_interval = adjustment_interval
        self.drop_threshold = drop_threshold
        self.last_adjustment = time.perf_counter()
        self.queue_sizes = {}
        self.metrics_history = []
        
    def should_adjust(self) -> bool:
        """Check if it's time to adjust queue sizes."""
        return time.perf_counter() - self.last_adjustment > self.adjustment_interval
        
    def compute_new_size(self, queue_name: str, current_size: int, 
                        metrics: QueueMetrics) -> int:
        """
        Compute new queue size based on performance metrics.
        TODO: Implement sophisticated sizing logic based on:
        - Drop rates
        - Processing times
        - Memory pressure
        - Upstream/downstream queue states
        """
        # Placeholder for now - just maintains current size
        return current_size
        
    def record_metrics(self, queue_metrics: Dict[str, QueueMetrics]):
        """Record metrics for future size adjustments."""
        self.metrics_history.append((time.perf_counter(), queue_metrics))
        # Trim old history
        while len(self.metrics_history) > 100:  # Keep last 100 samples
            self.metrics_history.pop(0)
            
    def get_size_recommendations(self, 
                               queue_metrics: Dict[str, QueueMetrics]
                               ) -> Dict[str, int]:
        """
        Get recommended queue sizes based on metrics.
        TODO: Implement adaptive sizing based on:
        - Scene detection patterns
        - Memory availability
        - Processing bottlenecks
        - Drop patterns
        """
        if not self.should_adjust():
            return {}
            
        self.record_metrics(queue_metrics)
        recommendations = {}
        
        for queue_name, metrics in queue_metrics.items():
            current_size = self.queue_sizes.get(queue_name, self.min_size)
            new_size = self.compute_new_size(queue_name, current_size, metrics)
            if new_size != current_size:
                recommendations[queue_name] = new_size
                
        self.last_adjustment = time.perf_counter()
        return recommendations

class QueueMonitor:
    """Monitors queue health and detects starvation conditions."""
    
    def __init__(self, starvation_threshold: float = 1.0):
        self.starvation_threshold = starvation_threshold
        self.metrics: Dict[str, QueueMetrics] = {}
        self.start_time = time.perf_counter()
        self.dropped_frames: Dict[str, List[Tuple[int, str]]] = {
            'detection': [],
            'flow': [],
            'stitch': []
        }
        
    def initialize_queue(self, queue_name: str):
        """Initialize monitoring for a new queue."""
        self.metrics[queue_name] = QueueMetrics(
            last_input_time=time.perf_counter(),
            processed_count=0,
            dropped_count=0,
            starvation_events=0,
            avg_processing_time=0.0,
            queue_latency=0.0
        )
        
    def update_metrics(self, queue_name: str, workunit: WorkUnit, is_processed: bool = True):
        """Update metrics for a queue based on workunit processing."""
        if queue_name not in self.metrics:
            self.initialize_queue(queue_name)
            
        metrics = self.metrics[queue_name]
        current_time = time.perf_counter()
        
        # Update last input time
        metrics.last_input_time = current_time
        
        if is_processed:
            metrics.processed_count += 1
            # Update average processing time
            new_time = workunit.total_processing_time()
            metrics.avg_processing_time = (
                (metrics.avg_processing_time * (metrics.processed_count - 1) + new_time)
                / metrics.processed_count
            )
        else:
            metrics.dropped_count += 1
            if workunit.dropped_reason:
                self.dropped_frames[queue_name].append((workunit.frame_idx, workunit.dropped_reason.reason_type))
            
        # Calculate queue latency
        metrics.queue_latency = current_time - workunit.timestamp
        
        # Check for starvation
        if current_time - metrics.last_input_time > self.starvation_threshold:
            metrics.starvation_events += 1
            print(f"WARNING: Queue {queue_name} starved for {current_time - metrics.last_input_time:.2f}s")
            
    def get_queue_health(self, queue_name: str) -> Dict:
        """Get health metrics for a specific queue."""
        if queue_name not in self.metrics:
            return {}
            
        metrics = self.metrics[queue_name]
        current_time = time.perf_counter()
        
        return {
            'processed_rate': metrics.processed_count / (current_time - self.start_time),
            'drop_rate': metrics.dropped_count / (current_time - self.start_time),
            'starvation_rate': metrics.starvation_events / (current_time - self.start_time),
            'avg_processing_time': metrics.avg_processing_time,
            'current_latency': metrics.queue_latency,
            'time_since_last_input': current_time - metrics.last_input_time,
            'total_dropped': metrics.dropped_count,
            'total_processed': metrics.processed_count
        }
    
    def get_dropped_frames(self, queue_name: str = None) -> Dict[str, List[Tuple[int, str]]]:
        """Get information about dropped frames."""
        if queue_name:
            return {queue_name: self.dropped_frames[queue_name]}
        return self.dropped_frames

class FrameQueue:
    """
    Three-tier queue system for frame processing with starvation detection.
    
    Attributes:
        detection_queue: Fast path for scene detection
        flow_queue: Medium path for flow analysis
        stitch_queue: Slow path for panorama stitching
    """
    
    def __init__(self, 
                 detection_maxlen: int = 10,
                 flow_maxlen: int = 30,
                 stitch_maxlen: int = 50,
                 profiler: Optional[PipelineProfiler] = None):
        # Initialize queues
        self.detection_queue = deque(maxlen=detection_maxlen)
        self.flow_queue = deque(maxlen=flow_maxlen)
        self.stitch_queue = deque(maxlen=stitch_maxlen)
        
        # Initialize monitoring
        self.monitor = QueueMonitor()
        self.profiler = profiler
        
        # Queue state tracking
        self.frame_count = 0
        self.last_processed_frame = -1
        self.sequence_boundaries = []
        
        # Performance metrics
        self.metrics = {
            'detection': {'processed': 0, 'dropped': 0},
            'flow': {'processed': 0, 'dropped': 0},
            'stitch': {'processed': 0, 'dropped': 0}
        }
        
    def add_frame(self, frame: np.ndarray) -> None:
        """Add a new frame to the detection queue."""
        timestamp = time.perf_counter()
        workunit = WorkUnit(
            frame_idx=self.frame_count,
            timestamp=timestamp,
            frame=frame
        )
        
        # Profile memory before adding
        if self.profiler:
            self.profiler.log_metric('queue_memory_pre_add', 
                                   torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)
        
        # Check if queue is full
        if len(self.detection_queue) >= self.detection_queue.maxlen:
            workunit.mark_as_dropped("detection_queue_full", queue_state={'queue_size': len(self.detection_queue)})
            self.metrics['detection']['dropped'] += 1
            self.monitor.update_metrics('detection', workunit, is_processed=False)
        else:
            # Add to detection queue
            self.detection_queue.append(workunit)
            self.metrics['detection']['processed'] += 1
            self.monitor.update_metrics('detection', workunit)
            
        self.frame_count += 1
        
        # Profile memory after adding
        if self.profiler:
            self.profiler.log_metric('queue_memory_post_add',
                                   torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)
            
    def promote_to_flow(self, workunit: WorkUnit, confidence: float) -> None:
        """Promote a workunit from detection to flow queue."""
        workunit.confidence = confidence
        workunit.add_stage_time('detection', time.perf_counter() - workunit.timestamp)
        
        if len(self.flow_queue) >= self.flow_queue.maxlen:
            workunit.mark_as_dropped("flow_queue_full", queue_state={'queue_size': len(self.flow_queue)})
            self.metrics['flow']['dropped'] += 1
            self.monitor.update_metrics('flow', workunit, is_processed=False)
        else:
            self.flow_queue.append(workunit)
            self.metrics['flow']['processed'] += 1
            self.monitor.update_metrics('flow', workunit)
            
    def promote_to_stitch(self, workunit: WorkUnit, confidence: float) -> None:
        """Promote a workunit from flow to stitch queue."""
        workunit.confidence = confidence
        workunit.add_stage_time('flow', time.perf_counter() - workunit.timestamp)
        
        if len(self.stitch_queue) >= self.stitch_queue.maxlen:
            workunit.mark_as_dropped("stitch_queue_full", queue_state={'queue_size': len(self.stitch_queue)})
            self.metrics['stitch']['dropped'] += 1
            self.monitor.update_metrics('stitch', workunit, is_processed=False)
        else:
            self.stitch_queue.append(workunit)
            self.metrics['stitch']['processed'] += 1
            self.monitor.update_metrics('stitch', workunit)
            
    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current sizes of all queues."""
        return {
            'detection': len(self.detection_queue),
            'flow': len(self.flow_queue),
            'stitch': len(self.stitch_queue)
        }
        
    def get_health_metrics(self) -> Dict[str, Dict]:
        """Get health metrics for all queues."""
        return {
            'detection': self.monitor.get_queue_health('detection'),
            'flow': self.monitor.get_queue_health('flow'),
            'stitch': self.monitor.get_queue_health('stitch')
        }
        
    def check_starvation(self) -> Dict[str, bool]:
        """Check starvation status of all queues."""
        current_time = time.perf_counter()
        return {
            queue_name: (current_time - metrics.last_input_time > self.monitor.starvation_threshold)
            for queue_name, metrics in self.monitor.metrics.items()
        }
        
    def get_processing_stats(self) -> Dict[str, Dict]:
        """Get processing statistics for all queues."""
        return {
            queue_name: {
                'processed': self.metrics[queue_name]['processed'],
                'dropped': self.metrics[queue_name]['dropped'],
                'total': self.metrics[queue_name]['processed'] + self.metrics[queue_name]['dropped']
            }
            for queue_name in ['detection', 'flow', 'stitch']
        }
        
    def get_dropped_frames(self) -> Dict[str, List[Tuple[int, str]]]:
        """Get detailed information about dropped frames."""
        return self.monitor.get_dropped_frames()
        
    def clear_queues(self) -> None:
        """Clear all queues and reset metrics."""
        self.detection_queue.clear()
        self.flow_queue.clear()
        self.stitch_queue.clear()
        self.frame_count = 0
        self.last_processed_frame = -1
        self.sequence_boundaries = []
        
        # Reset metrics
        for queue in self.metrics.values():
            queue['processed'] = 0
            queue['dropped'] = 0
            
    def __str__(self) -> str:
        """String representation with queue sizes and health metrics."""
        sizes = self.get_queue_sizes()
        health = self.get_health_metrics()
        drops = self.get_dropped_frames()
        
        status = [
            "FrameQueue Status:",
            f"  Detection Queue: {sizes['detection']} frames "
            f"(Latency: {health['detection'].get('current_latency', 0):.2f}s, "
            f"Dropped: {len(drops['detection'])})",
            f"  Flow Queue: {sizes['flow']} frames "
            f"(Latency: {health['flow'].get('current_latency', 0):.2f}s, "
            f"Dropped: {len(drops['flow'])})",
            f"  Stitch Queue: {sizes['stitch']} frames "
            f"(Latency: {health['stitch'].get('current_latency', 0):.2f}s, "
            f"Dropped: {len(drops['stitch'])})"
        ]
        
        return "\n".join(status) 