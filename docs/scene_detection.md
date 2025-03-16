# Scene Detection Strategy

## Overview
The goal is to reliably detect scene changes in anime content to identify the first and last frame of each scene. This is crucial for the subsequent steps of matte extraction.

## Current Implementation Analysis
- Using ResNet18 for feature extraction
- Cosine similarity comparison between frames
- Optional instance segmentation for foreground removal
- Basic threshold-based scene change detection

## Challenges in Anime Scene Detection
1. Hard cuts vs. transitions/fades
2. Fast action sequences
3. Character movement vs. actual scene changes
4. Art style variations
5. Varying frame rates and encoding

## Proposed Strategy

### 1. Feature Extraction
- Continue using ResNet18 but focus on background features
- Add spatial pyramid pooling to handle different scales
- Consider edge detection for layout changes
- Weight features from different regions differently (center vs edges)

### 2. Scene Change Detection
- Implement dual-threshold approach:
  - Lower threshold for candidate changes
  - Higher threshold for confirmed changes
- Use temporal window analysis (3-5 frames)
- Track sudden changes in color histograms
- Detect fade transitions separately

### 3. False Positive Reduction
- Implement motion estimation check
- Add scene length validation
- Check color palette consistency
- Verify spatial layout persistence

### 4. Performance Optimization
- Batch process frames
- Use frame sampling for initial pass
- Cache feature vectors
- Implement early stopping for clear changes

## Implementation Checklist

### Phase 1: Basic Detection
- [ ] Add debug visualization for frame features
- [ ] Implement frame similarity matrix
- [ ] Add basic scene boundary detection
- [ ] Create debug output directory structure
- [ ] Add progress reporting for scene detection
- [ ] Test with sample video segments

### Phase 2: Enhanced Detection
- [ ] Implement dual-threshold logic
- [ ] Add temporal window analysis
- [ ] Add color histogram analysis
- [ ] Implement fade detection
- [ ] Test with different anime styles

### Phase 3: False Positive Reduction
- [ ] Add motion estimation check
- [ ] Implement scene length validation
- [ ] Add color palette tracking
- [ ] Test with action sequences

### Phase 4: Optimization
- [ ] Implement batch processing
- [ ] Add frame sampling
- [ ] Optimize feature extraction
- [ ] Add caching mechanism
- [ ] Profile and optimize bottlenecks

### Phase 5: Validation
- [ ] Create test suite with different scenes
- [ ] Add metrics for detection accuracy
- [ ] Create visualization tools
- [ ] Document edge cases and solutions

## Success Metrics
1. Accuracy: >95% correct scene detection
2. Speed: Process 24fps video at >12fps
3. Memory: Peak usage <4GB GPU RAM
4. Robustness: Handle different anime styles

## Debug Tools Needed
1. Frame similarity visualizer
2. Scene boundary marker
3. Feature vector inspector
4. Performance profiler
5. Memory usage tracker

## Next Steps
1. Implement Phase 1 checklist
2. Create debug visualization tools
3. Test with sample video segments
4. Iterate based on results 