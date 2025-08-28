"""OpenCL-accelerated vanity address generator for Epix blockchain."""

import time
import numpy as np
from typing import Optional, Dict, Any, Callable, List

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    cl = None

from ..core.crypto import EpixCrypto, KeyPair
from ..core.config import EpixConfig
from ..core.generator import GenerationResult
from ..utils.patterns import Pattern, PatternValidator
from ..utils.performance import PerformanceMonitor
from ..utils.logging import ProgressLogger, get_logger


# OpenCL kernel for vanity address generation
OPENCL_KERNEL = """
// Simplified secp256k1 and keccak256 for demonstration
// In production, use proper cryptographic implementations

void secp256k1_scalar_mult(__global uchar *result, __global const uchar *scalar) {
    // Placeholder implementation
    for (int i = 0; i < 32; i++) {
        result[i] = scalar[i];
    }
}

void keccak256(__global uchar *output, __global const uchar *input, int len) {
    // Placeholder implementation
    for (int i = 0; i < 32; i++) {
        output[i] = input[i % len] ^ (i * 7);
    }
}

bool matches_pattern(__global const uchar *address, __global const char *pattern, int pattern_len) {
    for (int i = 0; i < pattern_len; i++) {
        if (address[i] != pattern[i]) {
            return false;
        }
    }
    return true;
}

__kernel void generate_vanity_addresses(
    __global uchar *private_keys,
    __global uchar *addresses,
    __global bool *matches,
    __global const char *pattern,
    int pattern_len,
    ulong offset
) {
    int idx = get_global_id(0);
    
    // Generate private key based on thread index and offset
    uchar private_key[32];
    ulong seed = offset + idx;
    
    // Simple PRNG (use proper cryptographic RNG in production)
    for (int i = 0; i < 32; i++) {
        private_key[i] = (seed >> (i % 8)) ^ (seed * 1103515245 + 12345);
        seed = seed * 1664525 + 1013904223;
    }
    
    // Generate public key (placeholder)
    uchar public_key[64];
    secp256k1_scalar_mult(public_key, private_key);
    
    // Generate Ethereum address
    uchar eth_address[32];
    keccak256(eth_address, public_key, 64);
    
    // Take last 20 bytes as address
    uchar address[20];
    for (int i = 0; i < 20; i++) {
        address[i] = eth_address[i + 12];
    }
    
    // Check pattern match
    matches[idx] = matches_pattern(address, pattern, pattern_len);
    
    // Store results
    for (int i = 0; i < 32; i++) {
        private_keys[idx * 32 + i] = private_key[i];
    }
    for (int i = 0; i < 20; i++) {
        addresses[idx * 20 + i] = address[i];
    }
}
"""


class OpenCLVanityGenerator:
    """OpenCL-accelerated vanity address generator."""
    
    def __init__(
        self,
        config: Optional[EpixConfig] = None,
        platform_id: int = 0,
        device_id: int = 0,
        batch_size: int = 1024 * 1024,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize OpenCL vanity generator."""
        if not OPENCL_AVAILABLE:
            raise ImportError("PyOpenCL is not available. Install with: pip install pyopencl")
        
        self.config = config or EpixConfig.default_testnet()
        self.crypto = EpixCrypto(self.config)
        self.pattern_validator = PatternValidator(self.config.get_address_prefix())
        self.performance_monitor = PerformanceMonitor()
        self.progress_logger = ProgressLogger()
        self.logger = get_logger()
        
        # OpenCL configuration
        self.platform_id = platform_id
        self.device_id = device_id
        self.batch_size = batch_size
        self.progress_callback = progress_callback
        
        # Initialize OpenCL
        self._init_opencl()
        
        # Control flags
        self._stop_generation = False
    
    def _init_opencl(self) -> None:
        """Initialize OpenCL context and compile kernel."""
        try:
            # Get platform and device
            platforms = cl.get_platforms()
            if self.platform_id >= len(platforms):
                raise ValueError(f"Platform {self.platform_id} not available")
            
            platform = platforms[self.platform_id]
            devices = platform.get_devices()
            if self.device_id >= len(devices):
                raise ValueError(f"Device {self.device_id} not available on platform {self.platform_id}")
            
            self.device = devices[self.device_id]
            
            # Create context and queue
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            
            # Compile kernel
            self.program = cl.Program(self.context, OPENCL_KERNEL).build()
            self.kernel = self.program.generate_vanity_addresses
            
            # Get device info
            self.device_name = self.device.name
            self.max_work_group_size = self.device.max_work_group_size
            self.max_compute_units = self.device.max_compute_units
            
            self.logger.info(f"Initialized OpenCL device: {self.device_name}")
            self.logger.info(f"Max work group size: {self.max_work_group_size}")
            self.logger.info(f"Max compute units: {self.max_compute_units}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenCL: {e}")
            raise
    
    def generate_vanity_address(
        self,
        pattern: Pattern,
        max_attempts: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> GenerationResult:
        """Generate vanity address using OpenCL acceleration."""
        
        self.logger.info(f"Starting OpenCL vanity generation for pattern: {pattern.pattern}")
        self.logger.info(f"Batch size: {self.batch_size:,}")
        
        # Validate pattern
        if not self.pattern_validator.validate_pattern(pattern.pattern, pattern.pattern_type):
            return GenerationResult(
                success=False,
                error=f"Invalid pattern: {pattern.pattern}"
            )
        
        # Currently only support prefix patterns for GPU
        if pattern.pattern_type.value != "prefix":
            return GenerationResult(
                success=False,
                error="GPU acceleration currently only supports prefix patterns"
            )
        
        # Reset monitoring
        self.performance_monitor.reset()
        self._stop_generation = False
        
        start_time = time.time()
        total_attempts = 0
        offset = 0
        
        try:
            # Create buffers
            private_keys_buffer = cl.Buffer(
                self.context, 
                cl.mem_flags.WRITE_ONLY, 
                self.batch_size * 32
            )
            addresses_buffer = cl.Buffer(
                self.context, 
                cl.mem_flags.WRITE_ONLY, 
                self.batch_size * 20
            )
            matches_buffer = cl.Buffer(
                self.context, 
                cl.mem_flags.WRITE_ONLY, 
                self.batch_size
            )
            
            # Pattern buffer
            pattern_bytes = pattern.pattern.encode('ascii')
            pattern_buffer = cl.Buffer(
                self.context,
                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=pattern_bytes
            )
            
            # Calculate work group size
            local_size = min(self.max_work_group_size, 256)
            global_size = ((self.batch_size + local_size - 1) // local_size) * local_size
            
            while not self._stop_generation:
                # Check timeout
                if timeout and (time.time() - start_time) >= timeout:
                    break
                
                # Check max attempts
                if max_attempts and total_attempts >= max_attempts:
                    break
                
                # Execute kernel
                self.kernel(
                    self.queue,
                    (global_size,),
                    (local_size,),
                    private_keys_buffer,
                    addresses_buffer,
                    matches_buffer,
                    pattern_buffer,
                    np.int32(len(pattern_bytes)),
                    np.uint64(offset)
                )
                
                # Read results
                matches = np.zeros(self.batch_size, dtype=np.bool_)
                cl.enqueue_copy(self.queue, matches, matches_buffer)
                
                # Check for matches
                match_indices = np.where(matches)[0]
                if len(match_indices) > 0:
                    # Found a match! Get the first one
                    match_idx = match_indices[0]
                    
                    # Read private keys and addresses
                    private_keys = np.zeros(self.batch_size * 32, dtype=np.uint8)
                    addresses = np.zeros(self.batch_size * 20, dtype=np.uint8)
                    
                    cl.enqueue_copy(self.queue, private_keys, private_keys_buffer)
                    cl.enqueue_copy(self.queue, addresses, addresses_buffer)
                    
                    # Extract the matching keypair
                    private_key_bytes = private_keys[match_idx * 32:(match_idx + 1) * 32].tobytes()
                    address_bytes = addresses[match_idx * 20:(match_idx + 1) * 20].tobytes()

                    # Convert to proper format
                    eth_address = "0x" + address_bytes.hex()
                    bech32_address = self.crypto.eth_address_to_bech32(eth_address)
                    public_key = self.crypto.private_key_to_public_key(private_key_bytes)
                    
                    keypair = KeyPair(
                        private_key=private_key_bytes,
                        public_key=public_key,
                        address=bech32_address,
                        eth_address=eth_address
                    )
                    
                    elapsed_time = time.time() - start_time
                    total_attempts += match_idx + 1
                    
                    self.progress_logger.log_success(
                        keypair.address,
                        self.crypto.private_key_to_hex(keypair.private_key),
                        total_attempts,
                        elapsed_time,
                        pattern.pattern
                    )
                    
                    return GenerationResult(
                        success=True,
                        keypair=keypair,
                        attempts=total_attempts,
                        elapsed_time=elapsed_time
                    )
                
                # Update counters
                total_attempts += self.batch_size
                offset += self.batch_size
                
                # Update performance monitoring
                self.performance_monitor.update_attempts(self.batch_size)
                
                # Log progress
                stats = self.performance_monitor.get_stats()
                self.progress_logger.log_progress(
                    stats.total_attempts,
                    stats.current_rate,
                    stats.elapsed_time,
                    pattern.pattern
                )
                
                # Call progress callback
                if self.progress_callback:
                    try:
                        self.progress_callback(self.performance_monitor.get_formatted_stats())
                    except Exception as e:
                        self.logger.error(f"Error in progress callback: {e}")
        
        except Exception as e:
            self.logger.error(f"OpenCL generation error: {e}")
            return GenerationResult(
                success=False,
                error=f"OpenCL error: {e}"
            )
        
        # No match found
        elapsed_time = time.time() - start_time
        return GenerationResult(
            success=False,
            attempts=total_attempts,
            elapsed_time=elapsed_time,
            error="No match found within limits"
        )
    
    def stop_generation(self) -> None:
        """Stop the current generation process."""
        self._stop_generation = True
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get OpenCL device information."""
        if not OPENCL_AVAILABLE:
            return {"error": "OpenCL not available"}
        
        try:
            return {
                "device_name": self.device.name,
                "device_type": cl.device_type.to_string(self.device.type),
                "vendor": self.device.vendor,
                "version": self.device.version,
                "driver_version": self.device.driver_version,
                "max_compute_units": self.device.max_compute_units,
                "max_work_group_size": self.device.max_work_group_size,
                "max_work_item_dimensions": self.device.max_work_item_dimensions,
                "global_mem_size": self.device.global_mem_size,
                "local_mem_size": self.device.local_mem_size,
            }
        except Exception as e:
            return {"error": f"Failed to get device info: {e}"}
    
    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        """List all available OpenCL devices."""
        if not OPENCL_AVAILABLE:
            return [{"error": "OpenCL not available"}]
        
        devices = []
        try:
            for platform_id, platform in enumerate(cl.get_platforms()):
                for device_id, device in enumerate(platform.get_devices()):
                    devices.append({
                        "platform_id": platform_id,
                        "device_id": device_id,
                        "platform_name": platform.name,
                        "device_name": device.name,
                        "device_type": cl.device_type.to_string(device.type),
                        "vendor": device.vendor,
                        "max_compute_units": device.max_compute_units,
                    })
        except Exception as e:
            devices.append({"error": f"Failed to list devices: {e}"})
        
        return devices
