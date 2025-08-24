#!/usr/bin/env python3
"""Installation script for EpixVanity."""

import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install core dependencies
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        print("✅ Core dependencies installed")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def install_gpu_support():
    """Optionally install GPU support."""
    print("\n🚀 GPU Support Installation")
    print("=" * 30)

    gpu_installed = False

    # Check for NVIDIA GPU
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("🎮 NVIDIA GPU detected")

            response = input("Install CUDA support? (y/N): ").lower().strip()
            if response in ['y', 'yes']:
                print("📦 Installing CUDA support...")
                try:
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", "pycuda"
                    ], check=True)
                    print("✅ CUDA support installed")
                    gpu_installed = True
                except subprocess.CalledProcessError:
                    print("❌ Failed to install CUDA support")
                    print("💡 You may need to install CUDA toolkit first")
                    print("💡 See: https://developer.nvidia.com/cuda-downloads")
        else:
            print("ℹ️  No NVIDIA GPU detected")

    except FileNotFoundError:
        print("ℹ️  nvidia-smi not found, skipping CUDA detection")

    # OpenCL support
    print("\n🔧 OpenCL Support")
    response = input("Install OpenCL support? (y/N): ").lower().strip()
    if response in ['y', 'yes']:
        print("📦 Installing OpenCL support...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "pyopencl"
            ], check=True)
            print("✅ OpenCL support installed")
            gpu_installed = True
        except subprocess.CalledProcessError:
            print("❌ Failed to install OpenCL support")
            print("💡 You may need to install OpenCL drivers first")
            print("💡 For Intel/AMD: Install your GPU drivers")
            print("💡 For NVIDIA: CUDA toolkit includes OpenCL")

    if not gpu_installed:
        print("\nℹ️  No GPU acceleration installed - CPU-only mode available")
        print("💡 You can install GPU support later with:")
        print("   pip install pycuda      # For CUDA")
        print("   pip install pyopencl    # For OpenCL")


def install_package():
    """Install the EpixVanity package."""
    print("\n📦 Installing EpixVanity...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], check=True)
        
        print("✅ EpixVanity installed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install EpixVanity: {e}")
        return False


def test_installation():
    """Test the installation."""
    print("\n🧪 Testing installation...")

    try:
        # Test basic import
        import epix_vanity
        print("✅ Basic import successful")

        # Test CLI
        result = subprocess.run([
            "epix-vanity", "--help"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ CLI working")
        else:
            print("❌ CLI test failed, trying alternative...")
            # Try alternative CLI access
            result = subprocess.run([
                sys.executable, "-c", "from epix_vanity.cli import main; main()"
            ], input="--help\n", capture_output=True, text=True)
            if "EpixVanity" in result.stdout or result.returncode == 0:
                print("✅ CLI accessible via Python")
            else:
                print("❌ CLI test failed")
                return False

        # Test core functionality
        from epix_vanity.core.config import EpixConfig
        from epix_vanity.core.crypto import EpixCrypto

        config = EpixConfig.default_testnet()
        crypto = EpixCrypto(config)
        keypair = crypto.generate_keypair()

        print("✅ Core functionality working")
        print(f"Sample address: {keypair.address}")

        # Test GPU availability
        from epix_vanity.gpu import CUDA_AVAILABLE, OPENCL_AVAILABLE
        if CUDA_AVAILABLE:
            print("✅ CUDA support available")
        if OPENCL_AVAILABLE:
            print("✅ OpenCL support available")
        if not CUDA_AVAILABLE and not OPENCL_AVAILABLE:
            print("ℹ️  GPU acceleration not available (CPU-only mode)")

        return True

    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def main():
    """Main installation function."""
    print("EpixVanity Installation Script")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        print("Please run this script from the EpixVanity root directory")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Install package
    if not install_package():
        sys.exit(1)
    
    # GPU support (optional)
    install_gpu_support()
    
    # Test installation
    if test_installation():
        print("\n🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Try: epix-vanity info")
        print("2. Generate a vanity address: epix-vanity generate abc")
        print("3. See examples: python examples/basic_usage.py")
    else:
        print("\n❌ Installation completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
