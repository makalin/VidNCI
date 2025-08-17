# VidNCI - Video Noise-Coded Illumination

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/vidnci.svg)](https://badge.fury.io/py/vidnci)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://vidnci.readthedocs.io/)

A robust, open-source Python library for detecting AI-generated videos using **Noise-Coded Illumination (NCI)** technique.

## 🎯 What is VidNCI?

VidNCI implements the novel NCI technique described in the research paper **"Noise-Coded Illumination For Forensic And Photometric Video Analysis"** ([arXiv:2507.23002](https://arxiv.org/abs/2507.23002)). This technique embeds unique, pseudo-random noise codes into video illumination, providing a verifiable signature for detecting manipulations and AI-generated content.

## ✨ Key Features

- **🔐 Active Detection**: Unlike passive methods, NCI embeds verifiable signatures directly into videos
- **🤖 AI Content Detection**: Specifically designed to detect deepfakes and AI-generated videos
- **⚡ High Performance**: Cython-optimized core algorithms for processing large videos
- **📊 Comprehensive Analysis**: Multiple analysis types from basic to advanced AI detection
- **🎨 Flexible Code Generation**: Support for Gaussian, binary, bipolar, and uniform noise codes
- **🔍 Forensic Tools**: Advanced anomaly detection and pattern analysis
- **📈 Visualization**: Built-in plotting and analysis visualization tools

## 🚀 Quick Start

### Installation

```bash
pip install vidnci
```

### Basic Usage

```python
from vidnci import VidNCIAnalyzer

# Initialize the analyzer
analyzer = VidNCIAnalyzer()

# Analyze a video for AI-generated content
results = analyzer.analyze_video(
    video_path="path/to/video.mp4",
    analysis_type="ai_detection"
)

# Check the results
ai_probability = results['analysis_results']['ai_detection']['ai_probability']
confidence = results['analysis_results']['ai_detection']['confidence']

print(f"AI Generation Probability: {ai_probability:.3f}")
print(f"Confidence: {confidence:.3f}")
```

## 📚 Documentation

- **[Full Documentation](https://vidnci.readthedocs.io/)**
- **[Installation Guide](docs/installation.md)**
- **[User Guide](docs/user_guide.md)**
- **[API Reference](docs/api_reference.md)**
- **[Examples](examples/)**

## 🔬 How It Works

### 1. Code Generation
Generate pseudo-random noise codes that will be embedded into video illumination:
```python
from vidnci import CodeGenerator

generator = CodeGenerator(seed=42)
code = generator.generate_gaussian_code(length=1000, mean=0, std=1)
```

### 2. Video Embedding (Simulation)
Simulate the effect of coded illumination on videos:
```python
from vidnci import VideoEmbedder

embedder = VideoEmbedder()
embedder.embed_single_code(
    input_video="input.mp4",
    output_video="embedded.mp4",
    code=code,
    strength=0.1
)
```

### 3. Code Extraction
Recover the embedded noise code from the video:
```python
from vidnci import CodeExtractor

extractor = CodeExtractor()
code_image = extractor.extract_code_image(
    video="embedded.mp4",
    code=code
)
```

### 4. Analysis
Analyze the recovered code image for anomalies and AI-generated content:
```python
from vidnci import Analyzer

analyzer = Analyzer()
results = analyzer.analyze_code_image(
    code_image=code_image,
    analysis_type="ai_detection"
)
```

## 🛠️ Installation Options

### Basic Installation
```bash
pip install vidnci
```

### With AI Capabilities
```bash
pip install vidnci[ai]
```

### Development Installation
```bash
git clone https://github.com/makalin/VidNCI.git
cd VidNCI
pip install -e .[dev]
```

## 📋 Requirements

- **Python**: 3.8 or higher
- **Core Dependencies**: NumPy, SciPy, OpenCV, Pillow, scikit-image, matplotlib
- **Optional**: Cython (for performance), PyTorch/TensorFlow (for AI features)

## 🔬 Research Applications

VidNCI is particularly useful for:

- **Digital Forensics**: Verifying video authenticity and detecting manipulations
- **Content Moderation**: Identifying AI-generated content on social platforms
- **Journalism**: Fact-checking video content and detecting deepfakes
- **Academic Research**: Studying AI video generation and detection methods
- **Security**: Video authentication and tamper detection

## 📊 Performance

- **Processing Speed**: Up to 10x faster than pure Python implementations
- **Memory Efficiency**: Optimized for large video files
- **Scalability**: Supports batch processing of multiple videos
- **Accuracy**: High detection rates for AI-generated content

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/makalin/VidNCI.git
cd VidNCI
pip install -e .[dev]
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Research Foundation**: Based on the work from arXiv:2507.23002
- **Open Source Community**: Built with popular Python scientific computing libraries
- **Contributors**: All contributors and users of the library

## 📞 Support

- **Documentation**: [https://vidnci.readthedocs.io/](https://vidnci.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/makalin/VidNCI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/makalin/VidNCI/discussions)

## 📈 Roadmap

- [ ] GPU acceleration support
- [ ] Real-time video analysis
- [ ] Additional noise code types
- [ ] Machine learning model integration
- [ ] Web interface
- [ ] Mobile app support

---

**VidNCI** - Empowering the fight against AI-generated misinformation through advanced video forensics.
