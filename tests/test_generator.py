"""Tests for vanity address generation."""

import pytest
from unittest.mock import Mock, patch
from epix_vanity.core.generator import VanityGenerator, GenerationResult
from epix_vanity.core.config import EpixConfig
from epix_vanity.utils.patterns import Pattern, PatternType


class TestVanityGenerator:
    """Test cases for VanityGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EpixConfig.default_testnet()
        self.generator = VanityGenerator(config=self.config, num_threads=2)
    
    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.config == self.config
        assert self.generator.num_threads == 2
        assert not self.generator.is_generating()
    
    def test_optimal_thread_count(self):
        """Test optimal thread count calculation."""
        generator = VanityGenerator()
        assert generator.num_threads > 0
        assert generator.num_threads <= 64  # Reasonable upper bound
    
    def test_invalid_pattern(self):
        """Test generation with invalid pattern."""
        # Create invalid pattern (contains invalid characters)
        pattern = Pattern("BIO", PatternType.PREFIX)  # Contains invalid chars
        
        result = self.generator.generate_vanity_address(pattern, max_attempts=10)
        
        assert not result.success
        assert "Invalid pattern" in result.error
    
    def test_generation_timeout(self):
        """Test generation with timeout."""
        pattern = Pattern("zzzzz", PatternType.PREFIX)  # Very difficult pattern
        
        result = self.generator.generate_vanity_address(
            pattern, 
            timeout=0.1  # Very short timeout
        )
        
        assert not result.success
        assert "Timeout" in result.error or "stopped" in result.error
    
    def test_generation_max_attempts(self):
        """Test generation with max attempts limit."""
        pattern = Pattern("zzzzz", PatternType.PREFIX)  # Very difficult pattern
        
        result = self.generator.generate_vanity_address(
            pattern,
            max_attempts=100  # Very low limit
        )
        
        assert not result.success
        assert ("Max attempts" in result.error or 
                "stopped" in result.error or 
                "Timeout" in result.error)
    
    @patch('epix_vanity.core.generator.VanityGenerator._worker_thread')
    def test_successful_generation(self, mock_worker):
        """Test successful generation (mocked)."""
        # Mock a successful result
        mock_keypair = Mock()
        mock_keypair.address = "epixabc123def456"
        mock_keypair.private_key = b"x" * 32
        
        mock_result = GenerationResult(
            success=True,
            keypair=mock_keypair,
            attempts=1000,
            elapsed_time=1.0
        )
        
        mock_worker.return_value = mock_result
        
        pattern = Pattern("abc", PatternType.PREFIX)
        result = self.generator.generate_vanity_address(pattern, max_attempts=10000)
        
        assert result.success
        assert result.keypair == mock_keypair
        assert result.attempts == 1000
    
    def test_stop_generation(self):
        """Test stopping generation."""
        assert not self.generator.is_generating()
        
        # Start generation in background (would need threading for real test)
        self.generator.stop_generation()
        
        # Should be able to call stop even when not generating
        assert not self.generator.is_generating()
    
    def test_performance_stats(self):
        """Test performance statistics retrieval."""
        stats = self.generator.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert "Total Attempts" in stats
        assert "Current Rate" in stats
        assert "CPU Usage" in stats
    
    def test_estimate_time_remaining(self):
        """Test time remaining estimation."""
        pattern = Pattern("abc", PatternType.PREFIX)
        
        # Before any generation, should return None
        time_remaining = self.generator.estimate_time_remaining(pattern)
        assert time_remaining is None or isinstance(time_remaining, float)
    
    def test_worker_thread_stop_condition(self):
        """Test worker thread respects stop condition."""
        pattern = Pattern("test", PatternType.PREFIX)
        
        # Set stop flag before starting worker
        self.generator._stop_generation.set()
        
        result = self.generator._worker_thread(
            pattern=pattern,
            thread_id=0,
            max_attempts=1000,
            timeout=10.0,
            start_time=0.0
        )
        
        # Should return None (stopped)
        assert result is None
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        callback_calls = []
        
        def progress_callback(stats):
            callback_calls.append(stats)
        
        generator = VanityGenerator(
            config=self.config,
            num_threads=1,
            progress_callback=progress_callback
        )
        
        # The callback should be stored
        assert generator.progress_callback == progress_callback


class TestGenerationResult:
    """Test cases for GenerationResult class."""
    
    def test_successful_result(self):
        """Test successful generation result."""
        mock_keypair = Mock()
        result = GenerationResult(
            success=True,
            keypair=mock_keypair,
            attempts=1000,
            elapsed_time=5.0
        )
        
        assert result.success
        assert result.keypair == mock_keypair
        assert result.attempts == 1000
        assert result.elapsed_time == 5.0
        assert result.error is None
    
    def test_failed_result(self):
        """Test failed generation result."""
        result = GenerationResult(
            success=False,
            attempts=5000,
            elapsed_time=10.0,
            error="Timeout reached"
        )
        
        assert not result.success
        assert result.keypair is None
        assert result.attempts == 5000
        assert result.elapsed_time == 10.0
        assert result.error == "Timeout reached"
    
    def test_default_values(self):
        """Test default values in GenerationResult."""
        result = GenerationResult(success=False)
        
        assert not result.success
        assert result.keypair is None
        assert result.attempts == 0
        assert result.elapsed_time == 0.0
        assert result.error is None
