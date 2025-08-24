"""Performance monitoring and statistics for vanity generation."""

import time
import psutil
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class PerformanceStats:
    """Performance statistics for vanity generation."""
    start_time: float = field(default_factory=time.time)
    total_attempts: int = 0
    successful_attempts: int = 0
    current_rate: float = 0.0  # attempts per second
    average_rate: float = 0.0  # average attempts per second
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: Optional[float] = None
    gpu_memory_usage: Optional[float] = None
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Get success rate as a percentage."""
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_attempts / self.total_attempts) * 100


class PerformanceMonitor:
    """Monitors and tracks performance metrics during vanity generation."""
    
    def __init__(self, update_interval: float = 1.0):
        """Initialize performance monitor."""
        self.update_interval = update_interval
        self.stats = PerformanceStats()
        self.last_update = time.time()
        self.last_attempts = 0
        self.lock = Lock()
        self.rate_history: List[float] = []
        self.max_history_size = 60  # Keep 60 seconds of history
        
    def update_attempts(self, attempts: int, successful: int = 0) -> None:
        """Update attempt counters."""
        with self.lock:
            self.stats.total_attempts += attempts
            self.stats.successful_attempts += successful
            
            current_time = time.time()
            if current_time - self.last_update >= self.update_interval:
                self._update_rates(current_time)
                self._update_system_stats()
                self.last_update = current_time
    
    def _update_rates(self, current_time: float) -> None:
        """Update rate calculations."""
        time_diff = current_time - self.last_update
        attempts_diff = self.stats.total_attempts - self.last_attempts
        
        if time_diff > 0:
            self.stats.current_rate = attempts_diff / time_diff
            self.rate_history.append(self.stats.current_rate)
            
            # Trim history
            if len(self.rate_history) > self.max_history_size:
                self.rate_history.pop(0)
            
            # Calculate average rate
            if self.rate_history:
                self.stats.average_rate = sum(self.rate_history) / len(self.rate_history)
        
        self.last_attempts = self.stats.total_attempts
    
    def _update_system_stats(self) -> None:
        """Update system resource usage statistics."""
        try:
            # CPU usage
            self.stats.cpu_usage = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.stats.memory_usage = memory.percent
            
            # GPU stats (if available)
            self._update_gpu_stats()
            
        except Exception:
            # Ignore errors in system monitoring
            pass
    
    def _update_gpu_stats(self) -> None:
        """Update GPU statistics if available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.stats.gpu_usage = utilization.gpu
                
                # GPU memory
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.stats.gpu_memory_usage = (memory_info.used / memory_info.total) * 100
                
        except ImportError:
            # pynvml not available
            pass
        except Exception:
            # Other GPU monitoring errors
            pass
    
    def get_stats(self) -> PerformanceStats:
        """Get current performance statistics."""
        with self.lock:
            return PerformanceStats(
                start_time=self.stats.start_time,
                total_attempts=self.stats.total_attempts,
                successful_attempts=self.stats.successful_attempts,
                current_rate=self.stats.current_rate,
                average_rate=self.stats.average_rate,
                cpu_usage=self.stats.cpu_usage,
                memory_usage=self.stats.memory_usage,
                gpu_usage=self.stats.gpu_usage,
                gpu_memory_usage=self.stats.gpu_memory_usage
            )
    
    def get_formatted_stats(self) -> Dict[str, str]:
        """Get formatted statistics for display."""
        stats = self.get_stats()
        
        return {
            "Elapsed Time": f"{stats.elapsed_time:.1f}s",
            "Total Attempts": f"{stats.total_attempts:,}",
            "Successful": f"{stats.successful_attempts}",
            "Current Rate": f"{stats.current_rate:.0f} attempts/s",
            "Average Rate": f"{stats.average_rate:.0f} attempts/s",
            "Success Rate": f"{stats.success_rate:.6f}%",
            "CPU Usage": f"{stats.cpu_usage:.1f}%",
            "Memory Usage": f"{stats.memory_usage:.1f}%",
            "GPU Usage": f"{stats.gpu_usage:.1f}%" if stats.gpu_usage is not None else "N/A",
            "GPU Memory": f"{stats.gpu_memory_usage:.1f}%" if stats.gpu_memory_usage is not None else "N/A"
        }
    
    def estimate_time_remaining(self, target_difficulty: int) -> Optional[float]:
        """Estimate time remaining to find a match based on current rate."""
        if self.stats.average_rate <= 0:
            return None
        
        remaining_attempts = target_difficulty - self.stats.total_attempts
        if remaining_attempts <= 0:
            return 0.0
        
        return remaining_attempts / self.stats.average_rate
    
    def reset(self) -> None:
        """Reset all statistics."""
        with self.lock:
            self.stats = PerformanceStats()
            self.last_update = time.time()
            self.last_attempts = 0
            self.rate_history.clear()
    
    def export_stats(self) -> Dict:
        """Export statistics for logging or analysis."""
        stats = self.get_stats()
        return {
            "timestamp": time.time(),
            "elapsed_time": stats.elapsed_time,
            "total_attempts": stats.total_attempts,
            "successful_attempts": stats.successful_attempts,
            "current_rate": stats.current_rate,
            "average_rate": stats.average_rate,
            "success_rate": stats.success_rate,
            "cpu_usage": stats.cpu_usage,
            "memory_usage": stats.memory_usage,
            "gpu_usage": stats.gpu_usage,
            "gpu_memory_usage": stats.gpu_memory_usage,
            "rate_history": self.rate_history.copy()
        }
