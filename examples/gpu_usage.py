#!/usr/bin/env python3
"""GPU acceleration example for EpixVanity."""

import sys
import os

# Add the parent directory to the path so we can import epix_vanity
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from epix_vanity.core.config import EpixConfig
from epix_vanity.utils.patterns import Pattern, PatternType, PatternValidator
from epix_vanity.gpu import CUDA_AVAILABLE, OPENCL_AVAILABLE

if CUDA_AVAILABLE:
    from epix_vanity.gpu.cuda_generator import CudaVanityGenerator

if OPENCL_AVAILABLE:
    from epix_vanity.gpu.opencl_generator import OpenCLVanityGenerator


def test_cuda():
    """Test CUDA acceleration."""
    if not CUDA_AVAILABLE:
        print("CUDA is not available")
        return
    
    print("Testing CUDA acceleration...")
    
    try:
        # Create configuration and pattern
        config = EpixConfig.default_testnet()
        pattern_validator = PatternValidator(config.get_address_prefix())
        pattern = pattern_validator.create_pattern("test", PatternType.PREFIX)
        
        # Create CUDA generator
        generator = CudaVanityGenerator(
            config=config,
            device_id=0,
            batch_size=1024 * 256  # Smaller batch for demo
        )
        
        # Get device info
        device_info = generator.get_device_info()
        print(f"CUDA Device: {device_info.get('device_name', 'Unknown')}")
        print(f"Compute Capability: {device_info.get('compute_capability', 'Unknown')}")
        print(f"Total Memory: {device_info.get('total_memory', 0) / (1024**3):.1f} GB")
        
        # Generate vanity address
        print("\nStarting CUDA generation...")
        result = generator.generate_vanity_address(
            pattern=pattern,
            max_attempts=1000000,
            timeout=10.0
        )
        
        if result.success:
            print("🎉 CUDA SUCCESS!")
            print(f"Address: {result.keypair.address}")
            print(f"Attempts: {result.attempts:,}")
            print(f"Time: {result.elapsed_time:.2f}s")
            print(f"Rate: {result.attempts / result.elapsed_time:.0f} attempts/s")
        else:
            print(f"CUDA generation failed: {result.error}")
    
    except Exception as e:
        print(f"CUDA error: {e}")


def test_opencl():
    """Test OpenCL acceleration."""
    if not OPENCL_AVAILABLE:
        print("OpenCL is not available")
        return
    
    print("\nTesting OpenCL acceleration...")
    
    try:
        # List available devices
        devices = OpenCLVanityGenerator.list_devices()
        print("Available OpenCL devices:")
        for device in devices:
            if "error" not in device:
                print(f"  Platform {device['platform_id']}, Device {device['device_id']}: "
                      f"{device['device_name']} ({device['device_type']})")
        
        if not devices or "error" in devices[0]:
            print("No OpenCL devices available")
            return
        
        # Create configuration and pattern
        config = EpixConfig.default_testnet()
        pattern_validator = PatternValidator(config.get_address_prefix())
        pattern = pattern_validator.create_pattern("test", PatternType.PREFIX)
        
        # Create OpenCL generator
        generator = OpenCLVanityGenerator(
            config=config,
            platform_id=0,
            device_id=0,
            batch_size=1024 * 256  # Smaller batch for demo
        )
        
        # Get device info
        device_info = generator.get_device_info()
        print(f"\nUsing OpenCL Device: {device_info.get('device_name', 'Unknown')}")
        print(f"Device Type: {device_info.get('device_type', 'Unknown')}")
        print(f"Compute Units: {device_info.get('max_compute_units', 'Unknown')}")
        
        # Generate vanity address
        print("\nStarting OpenCL generation...")
        result = generator.generate_vanity_address(
            pattern=pattern,
            max_attempts=1000000,
            timeout=10.0
        )
        
        if result.success:
            print("🎉 OpenCL SUCCESS!")
            print(f"Address: {result.keypair.address}")
            print(f"Attempts: {result.attempts:,}")
            print(f"Time: {result.elapsed_time:.2f}s")
            print(f"Rate: {result.attempts / result.elapsed_time:.0f} attempts/s")
        else:
            print(f"OpenCL generation failed: {result.error}")
    
    except Exception as e:
        print(f"OpenCL error: {e}")


def main():
    """GPU acceleration example."""
    
    print("EpixVanity GPU Acceleration Example")
    print("=" * 40)
    
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    print(f"OpenCL Available: {OPENCL_AVAILABLE}")
    
    if not CUDA_AVAILABLE and not OPENCL_AVAILABLE:
        print("\nNo GPU acceleration libraries available.")
        print("Install PyCUDA or PyOpenCL to use GPU acceleration:")
        print("  pip install pycuda")
        print("  pip install pyopencl")
        return
    
    try:
        if CUDA_AVAILABLE:
            test_cuda()
        
        if OPENCL_AVAILABLE:
            test_opencl()
    
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
