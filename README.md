# EpixVanity

🚀 **High-performance vanity address generator for Epix blockchain**

EpixVanity is a powerful, GPU-accelerated vanity address generator specifically designed for the Epix blockchain. It supports both CPU and GPU acceleration (CUDA/OpenCL) to generate custom addresses that match your desired patterns.

## ✨ Features

- **🔥 GPU Acceleration**: CUDA and OpenCL support for massive parallel processing
- **⚡ Multi-threaded CPU**: Optimized CPU generation with configurable thread count
- **🎯 Pattern Matching**: Support for prefix, suffix, contains, and regex patterns
- **📊 Real-time Monitoring**: Live performance statistics and progress tracking
- **🛡️ Secure**: Cryptographically secure key generation using secp256k1
- **🔧 Configurable**: Flexible configuration for different Epix networks
- **📱 CLI Interface**: Beautiful command-line interface with Rich formatting
- **💾 Export Options**: Save results in JSON, CSV, or text formats

## 🏗️ Architecture

EpixVanity is built specifically for the Epix blockchain, which uses:

- **Cosmos SDK** foundation with Ethereum compatibility
- **secp256k1** elliptic curve cryptography
- **Keccak-256** hashing for Ethereum-style addresses
- **Bech32** encoding with `epix` prefix for user-friendly addresses

## 📋 Requirements

- Python 3.8 or higher
- For GPU acceleration:
  - CUDA-capable GPU + PyCUDA (for CUDA support)
  - OpenCL-compatible device + PyOpenCL (for OpenCL support)

## 🚀 Quick Start

### Installation

#### Option 1: Virtual Environment Setup (Recommended)

For systems with externally managed Python environments:

```bash
# Clone the repository
git clone https://github.com/EpixZone/EpixVanity.git
cd EpixVanity

# Set up virtual environment and install
python3 setup_venv.py

# Activate the virtual environment
source epix_venv/bin/activate

# Now you can use EpixVanity
epix-vanity info
```

#### Option 2: CPU-Only Installation

If you don't have a GPU or want to skip GPU dependencies:

```bash
# Clone the repository
git clone https://github.com/EpixZone/EpixVanity.git
cd EpixVanity

# Run CPU-only installation (handles externally managed environments)
python3 install_cpu_only.py
```

#### Option 3: Manual Installation

```bash
# Clone the repository
git clone https://github.com/EpixZone/EpixVanity.git
cd EpixVanity

# Install core dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Optional: Install GPU support
pip install pycuda      # For CUDA
pip install pyopencl    # For OpenCL
```

### Basic Usage

```bash
# Generate an address starting with "test" (valid bech32 pattern)
epix-vanity generate test

# Generate with GPU acceleration (CUDA)
epix-vanity generate test --gpu cuda

# Generate with custom parameters
epix-vanity generate qpz --type prefix --threads 8 --timeout 60

# Get pattern suggestions for a word
epix-vanity suggest bitcoin

# Check valid characters and system info
epix-vanity info
```

### Python API

```python
from epix_vanity import VanityGenerator, EpixConfig
from epix_vanity.utils.patterns import Pattern, PatternType

# Create configuration
config = EpixConfig.default_testnet()

# Create generator
generator = VanityGenerator(config=config)

# Create pattern
pattern = Pattern("abc", PatternType.PREFIX)

# Generate vanity address
result = generator.generate_vanity_address(pattern)

if result.success:
    print(f"Address: {result.keypair.address}")
    print(f"Private Key: {result.keypair.private_key.hex()}")
```

## 📖 Detailed Usage

### Command Line Interface

#### Generate Command

```bash
epix-vanity generate [PATTERN] [OPTIONS]
```

**Options:**

- `--type`: Pattern type (`prefix`, `suffix`, `contains`, `regex`)
- `--case-sensitive`: Enable case-sensitive matching
- `--max-attempts`: Maximum number of attempts
- `--timeout`: Timeout in seconds
- `--threads`: Number of CPU threads
- `--gpu`: GPU acceleration (`cuda`, `opencl`)
- `--gpu-device`: GPU device ID
- `--batch-size`: GPU batch size
- `--output`: Output file path
- `--format`: Output format (`json`, `text`, `csv`)

#### Examples

```bash
# Basic prefix search
epix-vanity generate abc

# Case-sensitive suffix search
epix-vanity generate XYZ --type suffix --case-sensitive

# Regex pattern with timeout
epix-vanity generate "^[0-9]{4}" --type regex --timeout 300

# GPU acceleration with custom batch size
epix-vanity generate test --gpu cuda --batch-size 2097152

# Save results to file
epix-vanity generate lucky --output results.json --format json
```

#### Info Command

```bash
# Display system and device information
epix-vanity info
```

#### Suggest Command

```bash
# Get pattern suggestions for a word
epix-vanity suggest bitcoin
```

### Pattern Types

#### 1. Prefix Patterns

Addresses that start with the specified pattern:

```bash
epix-vanity generate abc --type prefix
# Generates: epixabc1234567890...
```

#### 2. Suffix Patterns

Addresses that end with the specified pattern:

```bash
epix-vanity generate xyz --type suffix
# Generates: epix1234567890...xyz
```

#### 3. Contains Patterns

Addresses that contain the pattern anywhere:

```bash
epix-vanity generate test --type contains
# Generates: epix123test456789...
```

#### 4. Regex Patterns

Advanced pattern matching using regular expressions:

```bash
epix-vanity generate "^[0-9]{4}" --type regex
# Generates: epix1234abcdef...
```

### GPU Acceleration

#### CUDA Setup

1. Install CUDA toolkit
2. Install PyCUDA:

   ```bash
   pip install pycuda
   ```

3. Use CUDA acceleration:

   ```bash
   epix-vanity generate abc --gpu cuda
   ```

#### OpenCL Setup

1. Install OpenCL drivers
2. Install PyOpenCL:

   ```bash
   pip install pyopencl
   ```

3. Use OpenCL acceleration:

   ```bash
   epix-vanity generate abc --gpu opencl
   ```

## 📊 Performance

### Difficulty Estimates

Pattern difficulty depends on length and character set:

| Pattern Length | Prefix/Suffix | Contains | Estimated Time* |
|----------------|---------------|----------|-----------------|
| 3 characters   | ~32,000       | ~16,000  | < 1 minute      |
| 4 characters   | ~1,000,000    | ~500,000 | 1-10 minutes    |
| 5 characters   | ~32,000,000   | ~16M     | 30-60 minutes   |
| 6 characters   | ~1,000,000,000| ~500M    | Hours to days   |

*Estimates based on modern CPU/GPU performance

### Performance Benchmarks

Typical performance on different hardware:

| Hardware | Performance |
|----------|-------------|
| CPU (8 cores) | 50,000 - 200,000 attempts/s |
| GTX 1060 | 1,000,000 - 5,000,000 attempts/s |
| RTX 3080 | 10,000,000 - 50,000,000 attempts/s |
| RTX 4090 | 20,000,000 - 100,000,000 attempts/s |

### Optimization Tips

1. **Use GPU acceleration** for patterns requiring > 1M attempts
2. **Start with shorter patterns** to test your setup
3. **Use prefix patterns** - they're typically faster than suffix/contains
4. **Increase batch size** for GPU generation (if you have enough VRAM)
5. **Monitor temperature** during intensive GPU generation

## 🔧 Configuration

### Custom Chain Configuration

Create a custom configuration file:

```json
{
  "chainId": "epix_1917-1",
  "chainName": "Epix",
  "rpc": "https://rpc.testnet.epix.zone",
  "rest": "https://api.testnet.epix.zone",
  "bech32Config": {
    "bech32PrefixAccAddr": "epix"
  }
}
```

Use with CLI:

```bash
epix-vanity --config custom_config.json generate abc
```

### Environment Variables

- `EPIX_VANITY_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `EPIX_VANITY_GPU_DEVICE`: Default GPU device ID
- `EPIX_VANITY_THREADS`: Default number of CPU threads

## 🛡️ Security

### Important Security Notes

1. **Private Key Security**: Generated private keys are cryptographically secure
2. **Random Number Generation**: Uses OS-provided secure random number generation
3. **Key Storage**: Never store private keys in plain text
4. **Network Security**: Generated addresses are for the specified Epix network only

### Best Practices

- Generate addresses on an offline machine for maximum security
- Immediately transfer generated private keys to secure storage
- Verify generated addresses before use
- Use hardware wallets for storing significant amounts

## 🐛 Troubleshooting

### Common Issues

#### Installation Issues

**PyCUDA/PyOpenCL Installation Fails:**

```bash
# Use CPU-only installation instead
python install_cpu_only.py

# Or install manually without GPU dependencies
pip install -r requirements.txt
pip install -e .
```

**Missing CUDA Toolkit:**

```bash
# Install CUDA toolkit first (NVIDIA GPUs)
# Download from: https://developer.nvidia.com/cuda-downloads
# Then install PyCUDA:
pip install pycuda
```

**Missing OpenCL Drivers:**

```bash
# For Intel/AMD GPUs: Install latest GPU drivers
# For NVIDIA GPUs: CUDA toolkit includes OpenCL
# Then install PyOpenCL:
pip install pyopencl
```

#### Runtime Issues

**CUDA Not Working:**

```bash
# Check CUDA installation
nvidia-smi

# Verify PyCUDA installation
python -c "import pycuda; print('CUDA OK')"
```

**OpenCL Not Working:**

```bash
# List OpenCL devices
epix-vanity info

# Test OpenCL installation
python -c "import pyopencl; print('OpenCL OK')"
```

**Low Performance:**

- Check GPU temperature and throttling
- Increase batch size for GPU generation
- Ensure adequate cooling
- Close other GPU-intensive applications

**Memory Issues:**

- Reduce batch size for GPU generation
- Close other applications
- Check available GPU memory with `nvidia-smi`

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=epix_vanity

# Run specific test
pytest tests/test_crypto.py
```

## 📚 Examples

See the `examples/` directory for detailed usage examples:

- `basic_usage.py` - Basic CPU generation
- `gpu_usage.py` - GPU acceleration examples
- `pattern_examples.py` - Pattern types and difficulty

Run examples:

```bash
python examples/basic_usage.py
python examples/gpu_usage.py
python examples/pattern_examples.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/EpixZone/EpixVanity.git
cd EpixVanity

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

- Use Black for code formatting
- Follow PEP 8 guidelines
- Add type hints where appropriate
- Write tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Epix Team](https://epix.zone) for the Epix blockchain
- [Cosmos SDK](https://cosmos.network) for the underlying framework
- [PyCUDA](https://mathema.tician.de/software/pycuda/) and [PyOpenCL](https://mathema.tician.de/software/pyopencl/) for GPU acceleration
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output

## 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/EpixZone/EpixVanity/wiki)
- **Issues**: [GitHub Issues](https://github.com/EpixZone/EpixVanity/issues)
- **Discord**: [Epix Community](https://discord.gg/epix)
- **Website**: [epix.zone](https://epix.zone)

---

**⚠️ Disclaimer**: This software is provided as-is. Always verify generated addresses and keep private keys secure. The authors are not responsible for any loss of funds due to misuse of this software.
