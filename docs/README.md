# VidNCI Documentation

Welcome to the VidNCI documentation! This library provides a robust, open-source Python implementation for detecting AI-generated videos using the Noise-Coded Illumination (NCI) technique.

## Quick Start

### Installation

```bash
pip install vidnci
```

### Basic Usage

```python
from vidnci import VidNCIAnalyzer

# Initialize the analyzer
analyzer = VidNCIAnalyzer()

# Analyze a video
results = analyzer.analyze_video(
    video_path="path/to/video.mp4",
    analysis_type="ai_detection"
)

# Check results
ai_probability = results['analysis_results']['ai_detection']['ai_probability']
print(f"AI Generation Probability: {ai_probability:.3f}")
```

## Documentation Sections

- [Installation Guide](installation.md)
- [User Guide](user_guide.md)
- [API Reference](api_reference.md)
- [Examples](examples.md)
- [Advanced Usage](advanced_usage.md)
- [Contributing](contributing.md)

## What is VidNCI?

VidNCI (Video Noise-Coded Illumination) is a Python library that implements the NCI technique for forensic video analysis. This technique embeds unique noise codes into video illumination, allowing for the detection of manipulations and AI-generated content.

## Key Features

- **Code Generation**: Generate various types of noise codes
- **Video Embedding**: Simulate coded illumination in videos
- **Code Extraction**: Recover embedded codes using the NCI algorithm
- **AI Detection**: Analyze recovered codes for AI-generated content
- **High Performance**: Cython-optimized core algorithms
- **Comprehensive Analysis**: Multiple analysis types and metrics

## Research Foundation

This library is based on the research paper:
**"Noise-Coded Illumination For Forensic And Photometric Video Analysis"** (arXiv:2507.23002)

## License

MIT License - see [LICENSE](../LICENSE) file for details.
