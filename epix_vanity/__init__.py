"""
EpixVanity - High-performance vanity address generator for Epix blockchain.

This package provides CPU and GPU-accelerated vanity address generation
for the Epix blockchain, which is built on the Cosmos SDK with Ethereum
compatibility features.
"""

__version__ = "1.0.0"
__author__ = "Mud"

from .core.generator import VanityGenerator
from .core.crypto import EpixCrypto
from .core.config import EpixConfig

__all__ = [
    "VanityGenerator",
    "EpixCrypto", 
    "EpixConfig",
]
