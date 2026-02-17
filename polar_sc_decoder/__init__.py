"""
polar-sc-decoder: A NumPy-based research toolkit for Polar Codes.

This package provides tools for Polar Codes under Successive Cancellation (SC) decoding,
featuring decoder-aware BLER analysis and SC-optimized code construction.
"""

__version__ = "0.1.0"
__author__ = "Hassan Noghrei Kalateh Sha Mohammad"

from .decoder import SCDecoder
from .encoder import PolarEncoder
from .code_construction import polar_code_construct

__all__ = [
    "SCDecoder",
    "PolarEncoder",
    "polar_code_construct",
]
