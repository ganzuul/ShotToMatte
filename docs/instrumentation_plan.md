# Pipeline Instrumentation Plan

## Core Metrics to Track

### 1. GPU Metrics
- VRAM Usage per Model
  - Peak allocation
  - Steady-state usage
  - Fragmentation
- Compute Utilization
  - Kernel occupancy
  - Memory bandwidth
  - PCIe transfer overhead
- Batch Processing Efficiency
  - Throughput vs batch size
  - Optimal batch size determination
  - Memory pressure points

### 2. Pipeline Timing
- Stage-by-stage latency
  - Model initialization
  - Frame preprocessing
  - Scene detection
  - Flow computation
  - Feature matching
  - Debug visualization
- Inter-stage delays
  - Data transfer times
  - Queue waiting times
- Frame processing timeline
  - End-to-end latency
  - Pipeline stall points

### 3. Memory Flow Analysis
- Frame buffer usage
  - Input queue depth
  - Processing queue depth
  - Output buffer state
- CPU-GPU transfers
  - Transfer frequency
  - Buffer sizes
  - Bottleneck identification

## Implementation Strategy

### 1. Lightweight Profiling Wrapper
```python
class PipelineProfiler:
    def __init__(self):
        self.stage_times = defaultdict(list)
        self.memory_stats = defaultdict(list)
        self.gpu_stats = defaultdict(list)
        
    @contextmanager
    def profile_stage(self, stage_name):
        torch.cuda.synchronize()
        start = time.perf_counter()
        start_mem = torch.cuda.memory_allocated()
        yield
        torch.cuda.synchronize()
        end = time.perf_counter()
        end_mem = torch.cuda.memory_allocated()
        
        self.stage_times[stage_name].append(end - start)
        self.memory_stats[stage_name].append(end_mem - start_mem)
```

### 2. Critical Points for Instrumentation
- Model Loading
  ```python
  with profiler.profile_stage("model_init"):
      model.load_weights()
  ```
- Frame Processing
  ```python
  with profiler.profile_stage("frame_processing"):
      frames = process_batch(batch)
  ```
- Memory Transfers
  ```python
  with profiler.profile_stage("gpu_transfer"):
      data = data.to(device)
  ```

### 3. Performance Metrics Collection
- Throughput metrics
  - Frames per second
  - Batch processing rate
  - Model inference time
- Resource utilization
  - GPU memory high watermark
  - CPU memory usage
  - Disk I/O for debug output
- Quality metrics
  - Scene detection accuracy
  - Flow estimation quality
  - Feature matching success rate

## Data Collection Strategy

### 1. Continuous Monitoring
- Rolling window statistics
- Moving averages
- Anomaly detection
- Resource pressure alerts

### 2. Periodic Sampling
- Frame capture at key points
- Memory snapshots
- Pipeline state dumps
- Performance counters

### 3. Debug Artifacts
- Timeline visualizations
- Memory usage graphs
- Stage timing distributions
- Bottleneck identification

## Analysis Tools

### 1. Built-in Tools
- `torch.cuda.memory_summary()`
- `torch.autograd.profiler`
- `nvidia-smi` metrics
- System resource monitors

### 2. Custom Analytics
- Pipeline efficiency score
- Resource utilization index
- Bottleneck detection
- Performance regression tracking

## Implementation Priority

1. Basic stage timing
2. Memory usage tracking
3. GPU utilization monitoring
4. Batch processing metrics
5. Quality assessments
6. Full pipeline analysis

## Success Criteria

- Complete visibility into pipeline performance
- Early warning system for resource constraints
- Clear identification of optimization targets
- Quantifiable improvement metrics
- Minimal overhead from instrumentation

## Notes
- Keep instrumentation overhead below 5%
- Focus on actionable metrics
- Enable/disable granular monitoring
- Store historical data for trend analysis 