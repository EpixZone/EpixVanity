"""GPU acceleration modules for EpixVanity."""

try:
    from .cuda_generator import CudaVanityGenerator
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    CudaVanityGenerator = None

try:
    from .opencl_generator import OpenCLVanityGenerator
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    OpenCLVanityGenerator = None

__all__ = ["CudaVanityGenerator", "OpenCLVanityGenerator", "CUDA_AVAILABLE", "OPENCL_AVAILABLE"]
