# Shottomatte

Shottomatte is a sophisticated video processing tool designed to extract high-quality matte paintings and background art from animated content. It uses advanced computer vision techniques to detect and reconstruct background scenes from panning and scanning shots.

## Features

- Content-aware scene detection optimized for animation
- Optical flow analysis for motion tracking
- Advanced feature matching using LoFTR
- Panorama reconstruction from panning/scanning shots
- GPU-accelerated processing with PyTorch
- Memory-efficient frame queue system
- Debug visualization tools

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Conda package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/shottomatte.git
cd shottomatte
```

2. Create and activate a Conda environment:
```bash
conda create -n shottomatte python=3.8
conda activate shottomatte
```

3. Install dependencies:
```bash
conda install pytorch torchvision cudatoolkit -c pytorch
conda install opencv -c conda-forge
conda install kornia -c conda-forge
```

## Usage

Basic usage example:

```python
from shottomatte import MatteExtractor

# Initialize the extractor
extractor = MatteExtractor()

# Process a video file
extractor.process_video(
    video_path="input.mp4",
    output_dir="output"
)
```

## Configuration

The `MatteExtractor` accepts various configuration parameters:

```python
config = {
    'scene_threshold': 0.7,         # Similarity threshold for scene detection
    'scan_threshold': 0.8,          # Minimum flow magnitude for scan detection
    'min_scan_duration': 10,        # Minimum frames in a scan
    'flow_consistency_threshold': 0.8, # Threshold for flow direction consistency
    'debug_output': True            # Enable debug visualizations
}
```

## Testing

Run the integration tests:

```bash
python tests/test_integration.py
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 