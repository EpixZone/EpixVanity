"""Main vanity address generator for Epix blockchain."""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass

from .crypto import EpixCrypto, KeyPair
from .config import EpixConfig
from ..utils.patterns import Pattern, PatternValidator
from ..utils.performance import PerformanceMonitor
from ..utils.logging import ProgressLogger, get_logger


@dataclass
class GenerationResult:
    """Result of vanity address generation."""
    success: bool
    keypair: Optional[KeyPair] = None
    attempts: int = 0
    elapsed_time: float = 0.0
    error: Optional[str] = None


class VanityGenerator:
    """CPU-based vanity address generator for Epix blockchain."""
    
    def __init__(
        self,
        config: Optional[EpixConfig] = None,
        num_threads: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize vanity generator."""
        self.config = config or EpixConfig.default_testnet()
        self.crypto = EpixCrypto(self.config)
        self.pattern_validator = PatternValidator(self.config.get_address_prefix())
        self.performance_monitor = PerformanceMonitor()
        self.progress_logger = ProgressLogger()
        self.logger = get_logger()
        
        # Threading configuration
        self.num_threads = num_threads or self._get_optimal_thread_count()
        self.progress_callback = progress_callback
        
        # Control flags
        self._stop_generation = threading.Event()
        self._generation_active = False
    
    def _get_optimal_thread_count(self) -> int:
        """Determine optimal number of threads for CPU generation."""
        import os
        cpu_count = os.cpu_count() or 4
        # Use CPU count for CPU-bound tasks
        return cpu_count
    
    def generate_vanity_address(
        self,
        pattern: Pattern,
        max_attempts: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> GenerationResult:
        """Generate a vanity address matching the given pattern."""
        
        self.logger.info(f"Starting vanity generation for pattern: {pattern.pattern}")
        self.logger.info(f"Pattern type: {pattern.pattern_type.value}")
        self.logger.info(f"Using {self.num_threads} threads")
        
        # Validate pattern
        if not self.pattern_validator.validate_pattern(pattern.pattern, pattern.pattern_type):
            return GenerationResult(
                success=False,
                error=f"Invalid pattern: {pattern.pattern}"
            )
        
        # Get pattern difficulty estimate
        pattern_info = self.pattern_validator.get_pattern_info(pattern)
        self.logger.info(f"Estimated difficulty: {pattern_info['difficulty_description']} "
                        f"({pattern_info['estimated_attempts']:,} attempts)")
        
        # Reset monitoring
        self.performance_monitor.reset()
        self._stop_generation.clear()
        self._generation_active = True
        
        start_time = time.time()
        result = None
        
        try:
            # Start generation with thread pool
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit worker tasks
                futures = []
                for thread_id in range(self.num_threads):
                    future = executor.submit(
                        self._worker_thread,
                        pattern,
                        thread_id,
                        max_attempts,
                        timeout,
                        start_time
                    )
                    futures.append(future)
                
                # Wait for first successful result
                for future in as_completed(futures):
                    try:
                        worker_result = future.result()
                        if worker_result and worker_result.success:
                            result = worker_result
                            self._stop_generation.set()
                            break
                    except Exception as e:
                        self.logger.error(f"Worker thread error: {e}")
                
                # Cancel remaining futures
                for future in futures:
                    future.cancel()
        
        except KeyboardInterrupt:
            self.logger.info("Generation interrupted by user")
            self._stop_generation.set()
        
        finally:
            self._generation_active = False
        
        # Return result or timeout/failure
        if result:
            elapsed_time = time.time() - start_time
            self.progress_logger.log_success(
                result.keypair.address,
                self.crypto.private_key_to_hex(result.keypair.private_key),
                result.attempts,
                elapsed_time,
                pattern.pattern
            )
            return result
        else:
            elapsed_time = time.time() - start_time
            stats = self.performance_monitor.get_stats()
            
            if timeout and elapsed_time >= timeout:
                error_msg = f"Timeout reached after {elapsed_time:.1f}s ({stats.total_attempts:,} attempts)"
            elif max_attempts and stats.total_attempts >= max_attempts:
                error_msg = f"Max attempts reached: {max_attempts:,}"
            else:
                error_msg = "Generation stopped without finding a match"
            
            return GenerationResult(
                success=False,
                attempts=stats.total_attempts,
                elapsed_time=elapsed_time,
                error=error_msg
            )

    def _worker_thread(
        self,
        pattern: Pattern,
        thread_id: int,
        max_attempts: Optional[int],
        timeout: Optional[float],
        start_time: float
    ) -> Optional[GenerationResult]:
        """Worker thread for vanity address generation."""

        attempts = 0
        batch_size = 1000  # Process in batches for better performance monitoring

        while not self._stop_generation.is_set():
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                break

            # Check max attempts
            if max_attempts and self.performance_monitor.get_stats().total_attempts >= max_attempts:
                break

            # Generate batch of addresses
            batch_attempts = 0
            for _ in range(batch_size):
                if self._stop_generation.is_set():
                    break

                try:
                    # Generate keypair
                    keypair = self.crypto.generate_keypair()
                    attempts += 1
                    batch_attempts += 1

                    # Check if address matches pattern
                    if self.pattern_validator.matches_pattern(keypair.address, pattern):
                        # Found a match!
                        self.performance_monitor.update_attempts(batch_attempts, 1)
                        return GenerationResult(
                            success=True,
                            keypair=keypair,
                            attempts=attempts,
                            elapsed_time=time.time() - start_time
                        )

                except Exception as e:
                    self.logger.error(f"Error in worker thread {thread_id}: {e}")
                    continue

            # Update performance monitoring
            self.performance_monitor.update_attempts(batch_attempts)

            # Log progress periodically
            if thread_id == 0:  # Only log from main thread
                stats = self.performance_monitor.get_stats()
                self.progress_logger.log_progress(
                    stats.total_attempts,
                    stats.current_rate,
                    stats.elapsed_time,
                    pattern.pattern
                )

                # Call progress callback if provided
                if self.progress_callback:
                    try:
                        self.progress_callback(self.performance_monitor.get_formatted_stats())
                    except Exception as e:
                        self.logger.error(f"Error in progress callback: {e}")

        return None

    def stop_generation(self) -> None:
        """Stop the current generation process."""
        self._stop_generation.set()

    def is_generating(self) -> bool:
        """Check if generation is currently active."""
        return self._generation_active

    def get_performance_stats(self) -> Dict[str, str]:
        """Get current performance statistics."""
        return self.performance_monitor.get_formatted_stats()

    def estimate_time_remaining(self, pattern: Pattern) -> Optional[float]:
        """Estimate time remaining for pattern generation."""
        pattern_info = self.pattern_validator.get_pattern_info(pattern)
        return self.performance_monitor.estimate_time_remaining(
            pattern_info['estimated_attempts']
        )
