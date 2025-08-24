#!/usr/bin/env python3
"""CPU-only installation script for EpixVanity (no GPU dependencies)."""

import sys
import subprocess
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
    """Install required dependencies (CPU-only)."""
    print("\n📦 Installing CPU-only dependencies...")

    try:
        # Try regular pip install first
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Core dependencies installed")
            return True

        # If that fails due to externally managed environment, try with --user
        if "externally-managed-environment" in result.stderr:
            print("🔄 Externally managed environment detected, trying --user installation...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--user", "-r", "requirements.txt"
            ], check=True)
            print("✅ Core dependencies installed (user mode)")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("\n💡 Try creating a virtual environment:")
        print("   python3 -m venv epix_venv")
        print("   source epix_venv/bin/activate")
        print("   python install_cpu_only.py")
        return False


def install_package():
    """Install the EpixVanity package."""
    print("\n📦 Installing EpixVanity...")

    try:
        # Try regular installation first
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ EpixVanity installed successfully")
            return True

        # If that fails, try with --user
        if "externally-managed-environment" in result.stderr:
            print("🔄 Trying user installation...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--user", "-e", "."
            ], check=True)
            print("✅ EpixVanity installed successfully (user mode)")
            return True
        else:
            print(f"❌ Failed to install EpixVanity: {result.stderr}")
            return False

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
        
        # Test core functionality
        from epix_vanity.core.config import EpixConfig
        from epix_vanity.core.crypto import EpixCrypto
        from epix_vanity.core.generator import VanityGenerator
        
        config = EpixConfig.default_testnet()
        crypto = EpixCrypto(config)
        keypair = crypto.generate_keypair()
        
        print("✅ Core functionality working")
        print(f"Sample address: {keypair.address}")
        
        # Test generator
        generator = VanityGenerator(config=config, num_threads=1)
        print(f"✅ CPU generator ready ({generator.num_threads} threads)")
        
        # Check GPU status
        from epix_vanity.gpu import CUDA_AVAILABLE, OPENCL_AVAILABLE
        print(f"ℹ️  CUDA available: {CUDA_AVAILABLE}")
        print(f"ℹ️  OpenCL available: {OPENCL_AVAILABLE}")
        
        return True
    
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def main():
    """Main installation function."""
    print("EpixVanity CPU-Only Installation")
    print("=" * 40)
    print("This installer skips GPU dependencies for systems without CUDA/OpenCL")
    print()
    
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
    
    # Test installation
    if test_installation():
        print("\n🎉 CPU-only installation completed successfully!")
        print("\nNext steps:")
        print("1. Try: python -c \"from epix_vanity.cli import main; main()\" info")
        print("2. Generate a vanity address: python -c \"from epix_vanity.cli import main; main()\" generate abc")
        print("3. See examples: python examples/basic_usage.py")
        print("\nNote: GPU acceleration is not available in this installation.")
        print("To add GPU support later, run: python install.py")
    else:
        print("\n❌ Installation completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
