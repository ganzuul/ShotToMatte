# FrameQueue Design and Pipeline Strategy

## Core Requirements

1. Scene Sequence Detection (Fast Path)
   - Quick detection of scene boundaries
   - Minimum sequence length threshold (10 frames)
   - Capture first and last frame of each sequence
   - Early termination for short sequences

2. Flow Analysis (Medium Path)
   - Triggered only for sequences > threshold length
   - Backtracking capability for confirmed sequences
   - Translation detection at any angle
   - Steady pace validation
   - Confidence metrics for flow consistency

3. Stitching Phase (Slow Path)
   - Activated only after positive scene + flow confirmation
   - Mask application when required
   - Final panorama assembly

## FrameQueue Architecture

### 1. Queue Structure
```python
class FrameQueue:
    def __init__(self):
        self.detection_queue = deque(maxlen=10)  # Fast path
        self.flow_queue = deque()               # Medium path
        self.stitch_queue = deque()             # Slow path
        self.confidence_metrics = {}
```

### 2. Processing Stages

#### Stage 1: Scene Detection (Fast)
- Maintain rolling window of recent frames
- Quick scene boundary detection
- Confidence metrics:
  - Scene change probability
  - Sequence stability score
  - Frame similarity metrics

#### Stage 2: Flow Analysis (Medium)
- Triggered conditions:
  - Sequence length > 10 frames
  - Scene stability confirmed
- Flow metrics:
  - Translation vector
  - Flow consistency score
  - Pace stability measurement

#### Stage 3: Stitching (Slow)
- Entry requirements:
  - Positive scene detection
  - Confirmed steady flow
  - Sufficient sequence length

## Instrumentation Strategy

### 1. Scene-Flow Cooperation Metrics
```python
class SceneFlowMetrics:
    def __init__(self):
        self.scene_confidence = []
        self.flow_confidence = []
        self.correlation_score = []
```

### 2. Key Measurements
- Scene boundary detection accuracy
- Flow vector stability
- Translation pace consistency
- Processing time per stage
- Memory usage per queue

### 3. Error Analysis
- False positive rate for scene detection
- Flow estimation error margins
- Sequence boundary precision
- Stage transition accuracy

## Optimization Rules

1. Fast Path (Scene Detection)
   - Run continuously on all frames
   - Early termination for non-qualifying sequences
   - Minimal memory footprint
   - Real-time performance target

2. Medium Path (Flow Analysis)
   - Triggered only for promising sequences
   - Backtracking limited to sequence boundaries
   - Adaptive sampling rate based on confidence
   - Resource-aware scheduling

3. Slow Path (Stitching)
   - Activated only with high confidence
   - Batch processing for efficiency
   - Quality-driven resource allocation
   - Background processing when possible

## Implementation Priorities

1. Queue Management
   - [ ] Implement three-tier queue structure
   - [ ] Add confidence metric tracking
   - [ ] Create stage transition logic
   - [ ] Implement backtracking capability

2. Scene Detection Enhancement
   - [ ] Add early termination logic
   - [ ] Implement boundary detection
   - [ ] Create confidence scoring
   - [ ] Optimize memory usage

3. Flow Analysis Integration
   - [ ] Add translation detection
   - [ ] Implement pace validation
   - [ ] Create stability metrics
   - [ ] Add backtracking logic

4. Performance Monitoring
   - [ ] Add stage timing metrics
   - [ ] Implement memory tracking
   - [ ] Create confidence logging
   - [ ] Add error margin analysis

## Success Metrics

1. Performance Targets
   - Scene detection: < 5ms per frame
   - Flow analysis: < 50ms per frame
   - Stitching: < 200ms per frame
   - Total latency: < 1s for qualifying sequences

2. Accuracy Targets
   - Scene boundary precision: > 95%
   - Flow detection accuracy: > 90%
   - False positive rate: < 5%
   - Sequence completion rate: > 85%

3. Resource Usage
   - Peak memory: < 2GB
   - GPU utilization: > 70%
   - CPU usage: < 50%
   - Disk I/O: < 100MB/s

## Notes and Constraints

1. Processing Strategy
   - Prioritize scene detection speed
   - Use flow analysis selectively
   - Optimize for steady translation
   - Maintain quality metrics

2. Resource Management
   - Clear queues aggressively
   - Implement smart batching
   - Use adaptive sampling
   - Monitor memory pressure

3. Quality Control
   - Track confidence scores
   - Maintain error margins
   - Log decision metrics
   - Enable debug visualization

4. Special Considerations
   - Handle sequence boundaries carefully
   - Preserve first/last frames
   - Track confidence intervals
   - Monitor resource usage 