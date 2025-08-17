"""
VidNCI - A Python Library for Defining AI-Generated Videos

A robust, open-source Python library for the forensic analysis of videos,
with a primary focus on detecting and defining artificial videos created by AI
using Noise-Coded Illumination (NCI) technique.

For more information, visit: https://github.com/makalin/VidNCI
"""

__version__ = "0.1.0"
__author__ = "VidNCI Team"
__email__ = "contact@vidnci.org"
__license__ = "MIT"
__url__ = "https://github.com/makalin/VidNCI"

# Core modules
from .core.code_generator import CodeGenerator
from .core.video_embedder import VideoEmbedder
from .core.code_extractor import CodeExtractor
from .core.analyzer import Analyzer

# High-level API
from .api import VidNCIAnalyzer

# Version info
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
    "CodeGenerator",
    "VideoEmbedder", 
    "CodeExtractor",
    "Analyzer",
    "VidNCIAnalyzer",
]
