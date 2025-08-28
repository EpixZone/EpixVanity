"""Pattern validation and matching utilities for vanity address generation."""

import re
from typing import List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    """Types of patterns supported."""
    PREFIX = "prefix"
    SUFFIX = "suffix"
    CONTAINS = "contains"
    REGEX = "regex"


@dataclass
class Pattern:
    """Represents a vanity address pattern."""
    pattern: str
    pattern_type: PatternType
    case_sensitive: bool = False
    
    def __post_init__(self):
        """Validate pattern after initialization."""
        if not self.pattern:
            raise ValueError("Pattern cannot be empty")
        
        # Validate pattern based on type
        if self.pattern_type == PatternType.REGEX:
            try:
                re.compile(self.pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")


class PatternValidator:
    """Validates and manages vanity address patterns."""
    
    # Valid characters for bech32 addresses (excluding '1', 'b', 'i', 'o')
    BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
    
    def __init__(self, address_prefix: str = "epix"):
        """Initialize with address prefix."""
        self.address_prefix = address_prefix
    
    def validate_pattern(self, pattern: str, pattern_type: PatternType) -> bool:
        """Validate a pattern for the given type."""
        if not pattern:
            return False
        
        # Convert to lowercase for validation (bech32 is case-insensitive)
        pattern_lower = pattern.lower()
        
        if pattern_type == PatternType.REGEX:
            try:
                re.compile(pattern)
                return True
            except re.error:
                return False
        
        # For non-regex patterns, check character validity
        for char in pattern_lower:
            if char not in self.BECH32_CHARSET:
                return False
        
        return True
    
    def estimate_difficulty(self, pattern: str, pattern_type: PatternType) -> int:
        """Estimate the difficulty (expected attempts) for finding a pattern."""
        if pattern_type == PatternType.REGEX:
            # For regex, return a conservative estimate
            return 32 ** len(pattern)  # Rough approximation
        
        charset_size = len(self.BECH32_CHARSET)
        pattern_length = len(pattern)
        
        if pattern_type == PatternType.PREFIX:
            return charset_size ** pattern_length
        elif pattern_type == PatternType.SUFFIX:
            return charset_size ** pattern_length
        elif pattern_type == PatternType.CONTAINS:
            # More complex calculation for contains
            # This is a rough approximation
            return charset_size ** pattern_length // 2
        
        return charset_size ** pattern_length
    
    def create_pattern(
        self, 
        pattern_str: str, 
        pattern_type: Union[str, PatternType],
        case_sensitive: bool = False
    ) -> Pattern:
        """Create and validate a pattern object."""
        if isinstance(pattern_type, str):
            try:
                pattern_type = PatternType(pattern_type.lower())
            except ValueError:
                raise ValueError(f"Invalid pattern type: {pattern_type}")
        
        if not self.validate_pattern(pattern_str, pattern_type):
            raise ValueError(f"Invalid pattern '{pattern_str}' for type {pattern_type.value}")
        
        return Pattern(
            pattern=pattern_str,
            pattern_type=pattern_type,
            case_sensitive=case_sensitive
        )
    
    def matches_pattern(self, address: str, pattern: Pattern) -> bool:
        """Check if an address matches the given pattern."""
        # Remove prefix and bech32 separator for pattern matching
        if address.startswith(self.address_prefix):
            # Remove both the prefix (e.g., "epix") and the bech32 separator "1"
            prefix_with_separator = self.address_prefix + "1"
            if address.startswith(prefix_with_separator):
                address_body = address[len(prefix_with_separator):]
            else:
                # Fallback to old behavior if separator not found
                address_body = address[len(self.address_prefix):]
        else:
            address_body = address
        
        pattern_str = pattern.pattern
        if not pattern.case_sensitive:
            address_body = address_body.lower()
            pattern_str = pattern_str.lower()
        
        if pattern.pattern_type == PatternType.PREFIX:
            return address_body.startswith(pattern_str)
        elif pattern.pattern_type == PatternType.SUFFIX:
            return address_body.endswith(pattern_str)
        elif pattern.pattern_type == PatternType.CONTAINS:
            return pattern_str in address_body
        elif pattern.pattern_type == PatternType.REGEX:
            try:
                flags = 0 if pattern.case_sensitive else re.IGNORECASE
                return bool(re.search(pattern_str, address_body, flags))
            except re.error:
                return False
        
        return False
    
    def get_pattern_info(self, pattern: Pattern) -> dict:
        """Get detailed information about a pattern."""
        difficulty = self.estimate_difficulty(pattern.pattern, pattern.pattern_type)
        
        return {
            "pattern": pattern.pattern,
            "type": pattern.pattern_type.value,
            "case_sensitive": pattern.case_sensitive,
            "estimated_attempts": difficulty,
            "difficulty_description": self._get_difficulty_description(difficulty)
        }
    
    def _get_difficulty_description(self, attempts: int) -> str:
        """Get a human-readable difficulty description."""
        if attempts < 1000:
            return "Very Easy"
        elif attempts < 10000:
            return "Easy"
        elif attempts < 100000:
            return "Medium"
        elif attempts < 1000000:
            return "Hard"
        elif attempts < 10000000:
            return "Very Hard"
        else:
            return "Extremely Hard"
    
    @classmethod
    def suggest_patterns(cls, desired_word: str) -> List[str]:
        """Suggest valid patterns based on a desired word."""
        suggestions = []
        
        # Convert to lowercase and filter valid characters
        valid_chars = ""
        for char in desired_word.lower():
            if char in cls.BECH32_CHARSET:
                valid_chars += char
        
        if valid_chars:
            suggestions.append(valid_chars)
            
            # Suggest partial matches if the word is long
            if len(valid_chars) > 4:
                suggestions.append(valid_chars[:4])
                suggestions.append(valid_chars[-4:])
        
        return suggestions
