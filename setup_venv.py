#!/usr/bin/env python3
"""Setup virtual environment and install EpixVanity."""

import sys
import subprocess
import os
from pathlib import Path


def create_virtual_environment():
    """Create a virtual environment for EpixVanity."""
    print("🔧 Creating virtual environment...")
    
    venv_path = Path("epix_venv")
    
    if venv_path.exists():
        print("ℹ️  Virtual environment already exists")
        return True
    
    try:
        subprocess.run([
            sys.executable, "-m", "venv", str(venv_path)
        ], check=True)
        
        print("✅ Virtual environment created")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False


def get_venv_python():
    """Get the Python executable from the virtual environment."""
    venv_path = Path("epix_venv")
    
    if os.name == 'nt':  # Windows
        return venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux/macOS
        return venv_path / "bin" / "python"


def install_in_venv():
    """Install EpixVanity in the virtual environment."""
    print("\n📦 Installing EpixVanity in virtual environment...")
    
    venv_python = get_venv_python()
    
    if not venv_python.exists():
        print("❌ Virtual environment Python not found")
        return False
    
    try:
        # Upgrade pip first
        subprocess.run([
            str(venv_python), "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)
        
        # Install dependencies
        subprocess.run([
            str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        # Install EpixVanity
        subprocess.run([
            str(venv_python), "-m", "pip", "install", "-e", "."
        ], check=True)
        
        print("✅ EpixVanity installed successfully in virtual environment")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install in virtual environment: {e}")
        return False


def test_installation():
    """Test the installation in virtual environment."""
    print("\n🧪 Testing installation...")
    
    venv_python = get_venv_python()
    
    try:
        # Test basic functionality
        result = subprocess.run([
            str(venv_python), "-c", 
            "from epix_vanity.core.config import EpixConfig; "
            "from epix_vanity.core.crypto import EpixCrypto; "
            "config = EpixConfig.default_testnet(); "
            "crypto = EpixCrypto(config); "
            "keypair = crypto.generate_keypair(); "
            "print(f'✅ Test successful! Sample address: {keypair.address}')"
        ], capture_output=True, text=True, check=True)
        
        print(result.stdout.strip())
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation test failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def print_usage_instructions():
    """Print instructions for using the virtual environment."""
    venv_path = Path("epix_venv")
    
    if os.name == 'nt':  # Windows
        activate_script = venv_path / "Scripts" / "activate.bat"
        python_exe = venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux/macOS
        activate_script = venv_path / "bin" / "activate"
        python_exe = venv_path / "bin" / "python"
    
    print(f"\n🎉 Installation completed successfully!")
    print(f"\nTo use EpixVanity, activate the virtual environment:")
    
    if os.name == 'nt':  # Windows
        print(f"   {activate_script}")
    else:  # Unix/Linux/macOS
        print(f"   source {activate_script}")
    
    print(f"\nThen you can use EpixVanity:")
    print(f"   {python_exe} -m epix_vanity.cli generate abc")
    print(f"   {python_exe} -m epix_vanity.cli info")
    print(f"   {python_exe} examples/basic_usage.py")
    
    print(f"\nOr run directly without activation:")
    print(f"   {python_exe} -m epix_vanity.cli generate abc")


def main():
    """Main setup function."""
    print("EpixVanity Virtual Environment Setup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        print("Please run this script from the EpixVanity root directory")
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install in virtual environment
    if not install_in_venv():
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        sys.exit(1)
    
    # Print usage instructions
    print_usage_instructions()


if __name__ == "__main__":
    main()
