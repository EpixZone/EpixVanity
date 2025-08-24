"""Tests for pattern validation and matching."""

import pytest
from epix_vanity.utils.patterns import Pattern, PatternType, PatternValidator


class TestPatternValidator:
    """Test cases for PatternValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = PatternValidator("epix")
    
    def test_validate_pattern_prefix(self):
        """Test prefix pattern validation."""
        assert self.validator.validate_pattern("abc", PatternType.PREFIX)
        assert self.validator.validate_pattern("123", PatternType.PREFIX)
        assert self.validator.validate_pattern("test", PatternType.PREFIX)
        
        # Invalid characters
        assert not self.validator.validate_pattern("ABC", PatternType.PREFIX)  # Uppercase
        assert not self.validator.validate_pattern("1bi", PatternType.PREFIX)  # Contains 'b', 'i'
        assert not self.validator.validate_pattern("o0", PatternType.PREFIX)   # Contains 'o'
    
    def test_validate_pattern_suffix(self):
        """Test suffix pattern validation."""
        assert self.validator.validate_pattern("xyz", PatternType.SUFFIX)
        assert self.validator.validate_pattern("789", PatternType.SUFFIX)
        
        # Invalid characters
        assert not self.validator.validate_pattern("XYZ", PatternType.SUFFIX)
        assert not self.validator.validate_pattern("1o", PatternType.SUFFIX)
    
    def test_validate_pattern_contains(self):
        """Test contains pattern validation."""
        assert self.validator.validate_pattern("test", PatternType.CONTAINS)
        assert self.validator.validate_pattern("456", PatternType.CONTAINS)
        
        # Invalid characters
        assert not self.validator.validate_pattern("TEST", PatternType.CONTAINS)
        assert not self.validator.validate_pattern("bi", PatternType.CONTAINS)
    
    def test_validate_pattern_regex(self):
        """Test regex pattern validation."""
        assert self.validator.validate_pattern("^[0-9]{3}", PatternType.REGEX)
        assert self.validator.validate_pattern("test.*", PatternType.REGEX)
        assert self.validator.validate_pattern("[a-z]+", PatternType.REGEX)
        
        # Invalid regex
        assert not self.validator.validate_pattern("[", PatternType.REGEX)
        assert not self.validator.validate_pattern("*", PatternType.REGEX)
    
    def test_create_pattern(self):
        """Test pattern creation."""
        pattern = self.validator.create_pattern("abc", PatternType.PREFIX)
        
        assert isinstance(pattern, Pattern)
        assert pattern.pattern == "abc"
        assert pattern.pattern_type == PatternType.PREFIX
        assert not pattern.case_sensitive
        
        # Case sensitive pattern
        pattern_cs = self.validator.create_pattern("test", PatternType.CONTAINS, case_sensitive=True)
        assert pattern_cs.case_sensitive
    
    def test_create_pattern_string_type(self):
        """Test pattern creation with string type."""
        pattern = self.validator.create_pattern("xyz", "suffix")
        assert pattern.pattern_type == PatternType.SUFFIX
        
        # Invalid type
        with pytest.raises(ValueError):
            self.validator.create_pattern("test", "invalid_type")
    
    def test_create_pattern_invalid(self):
        """Test pattern creation with invalid patterns."""
        with pytest.raises(ValueError):
            self.validator.create_pattern("ABC", PatternType.PREFIX)  # Invalid chars
        
        with pytest.raises(ValueError):
            self.validator.create_pattern("", PatternType.PREFIX)  # Empty pattern
        
        with pytest.raises(ValueError):
            self.validator.create_pattern("[", PatternType.REGEX)  # Invalid regex
    
    def test_matches_pattern_prefix(self):
        """Test prefix pattern matching."""
        pattern = Pattern("abc", PatternType.PREFIX)
        
        assert self.validator.matches_pattern("epixabc123", pattern)
        assert self.validator.matches_pattern("epixabcdef", pattern)
        assert not self.validator.matches_pattern("epix123abc", pattern)
        assert not self.validator.matches_pattern("epixdef123", pattern)
    
    def test_matches_pattern_suffix(self):
        """Test suffix pattern matching."""
        pattern = Pattern("xyz", PatternType.SUFFIX)
        
        assert self.validator.matches_pattern("epix123xyz", pattern)
        assert self.validator.matches_pattern("epixabcxyz", pattern)
        assert not self.validator.matches_pattern("epixxyza123", pattern)
        assert not self.validator.matches_pattern("epix123abc", pattern)
    
    def test_matches_pattern_contains(self):
        """Test contains pattern matching."""
        pattern = Pattern("test", PatternType.CONTAINS)
        
        assert self.validator.matches_pattern("epixtest123", pattern)
        assert self.validator.matches_pattern("epix123test", pattern)
        assert self.validator.matches_pattern("epix1test23", pattern)
        assert not self.validator.matches_pattern("epix123456", pattern)
    
    def test_matches_pattern_regex(self):
        """Test regex pattern matching."""
        pattern = Pattern("^[0-9]{3}", PatternType.REGEX)
        
        assert self.validator.matches_pattern("epix123abc", pattern)
        assert self.validator.matches_pattern("epix456def", pattern)
        assert not self.validator.matches_pattern("epixabc123", pattern)
        assert not self.validator.matches_pattern("epix12abc", pattern)  # Only 2 digits
    
    def test_matches_pattern_case_sensitive(self):
        """Test case-sensitive pattern matching."""
        pattern_cs = Pattern("Test", PatternType.CONTAINS, case_sensitive=True)
        pattern_ci = Pattern("Test", PatternType.CONTAINS, case_sensitive=False)
        
        # Case sensitive - exact match required
        assert self.validator.matches_pattern("epixTest123", pattern_cs)
        assert not self.validator.matches_pattern("epixtest123", pattern_cs)
        assert not self.validator.matches_pattern("epixTEST123", pattern_cs)
        
        # Case insensitive - any case matches
        assert self.validator.matches_pattern("epixTest123", pattern_ci)
        assert self.validator.matches_pattern("epixtest123", pattern_ci)
        assert self.validator.matches_pattern("epixTEST123", pattern_ci)
    
    def test_estimate_difficulty(self):
        """Test difficulty estimation."""
        # Prefix patterns
        diff_3 = self.validator.estimate_difficulty("abc", PatternType.PREFIX)
        diff_4 = self.validator.estimate_difficulty("abcd", PatternType.PREFIX)
        assert diff_4 > diff_3
        
        # Contains should be easier than prefix/suffix
        diff_contains = self.validator.estimate_difficulty("abc", PatternType.CONTAINS)
        diff_prefix = self.validator.estimate_difficulty("abc", PatternType.PREFIX)
        assert diff_contains < diff_prefix
    
    def test_get_pattern_info(self):
        """Test pattern information retrieval."""
        pattern = Pattern("abc", PatternType.PREFIX)
        info = self.validator.get_pattern_info(pattern)
        
        assert "pattern" in info
        assert "type" in info
        assert "case_sensitive" in info
        assert "estimated_attempts" in info
        assert "difficulty_description" in info
        
        assert info["pattern"] == "abc"
        assert info["type"] == "prefix"
        assert not info["case_sensitive"]
        assert isinstance(info["estimated_attempts"], int)
        assert isinstance(info["difficulty_description"], str)
    
    def test_suggest_patterns(self):
        """Test pattern suggestions."""
        # Valid word with all valid characters
        suggestions = PatternValidator.suggest_patterns("test")
        assert "test" in suggestions
        
        # Word with invalid characters
        suggestions = PatternValidator.suggest_patterns("bitcoin")
        # Should filter out invalid characters and suggest valid parts
        assert len(suggestions) > 0
        for suggestion in suggestions:
            for char in suggestion:
                assert char in PatternValidator.BECH32_CHARSET
        
        # Long word should have partial suggestions
        suggestions = PatternValidator.suggest_patterns("verylongword")
        assert len(suggestions) > 1  # Should have multiple suggestions
        
        # Word with no valid characters
        suggestions = PatternValidator.suggest_patterns("BIO")
        assert len(suggestions) == 0 or all(len(s) == 0 for s in suggestions)
    
    def test_bech32_charset(self):
        """Test that BECH32_CHARSET is correct."""
        charset = PatternValidator.BECH32_CHARSET
        
        # Should not contain excluded characters
        excluded = "1bio"
        for char in excluded:
            assert char not in charset
        
        # Should be lowercase
        assert charset == charset.lower()
        
        # Should have expected length (32 characters total - 4 excluded = 28)
        assert len(charset) == 28
