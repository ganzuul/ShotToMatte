"""
ShotToMatte: Advanced GPU-accelerated matte painting extraction from animated content.
"""

__version__ = "0.1.0"

from .panorama import MatteExtractor
from .flow import RAFTOpticalFlow
from .scene import ContentAwareSceneDetector 