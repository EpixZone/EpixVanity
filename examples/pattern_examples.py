#!/usr/bin/env python3
"""Pattern examples for EpixVanity."""

import sys
import os

# Add the parent directory to the path so we can import epix_vanity
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from epix_vanity.core.config import EpixConfig
from epix_vanity.utils.patterns import Pattern, PatternType, PatternValidator


def demonstrate_patterns():
    """Demonstrate different pattern types and their difficulty."""
    
    print("EpixVanity Pattern Examples")
    print("=" * 40)
    
    # Create configuration
    config = EpixConfig.default_testnet()
    pattern_validator = PatternValidator(config.get_address_prefix())
    
    print(f"Address prefix: {config.get_address_prefix()}")
    print(f"Valid characters: {pattern_validator.BECH32_CHARSET}")
    print()
    
    # Example patterns
    examples = [
        ("abc", PatternType.PREFIX, "Simple 3-character prefix"),
        ("xyz", PatternType.SUFFIX, "3-character suffix"),
        ("test", PatternType.CONTAINS, "Contains 'test' anywhere"),
        ("epix", PatternType.PREFIX, "Starts with 'epix'"),
        ("000", PatternType.PREFIX, "Starts with three zeros"),
        ("123", PatternType.SUFFIX, "Ends with '123'"),
        ("a.*z", PatternType.REGEX, "Regex: starts with 'a', ends with 'z'"),
        ("^[0-9]{4}", PatternType.REGEX, "Regex: starts with 4 digits"),
    ]
    
    print("Pattern Examples and Difficulty Estimates:")
    print("-" * 60)
    
    for pattern_str, pattern_type, description in examples:
        try:
            # Validate pattern
            if pattern_validator.validate_pattern(pattern_str, pattern_type):
                pattern = pattern_validator.create_pattern(pattern_str, pattern_type)
                pattern_info = pattern_validator.get_pattern_info(pattern)
                
                print(f"Pattern: '{pattern_str}' ({pattern_type.value})")
                print(f"  Description: {description}")
                print(f"  Difficulty: {pattern_info['difficulty_description']}")
                print(f"  Expected attempts: {pattern_info['estimated_attempts']:,}")
                print()
            else:
                print(f"Pattern: '{pattern_str}' ({pattern_type.value})")
                print(f"  Description: {description}")
                print(f"  Status: INVALID - contains unsupported characters")
                print()
        
        except Exception as e:
            print(f"Pattern: '{pattern_str}' ({pattern_type.value})")
            print(f"  Description: {description}")
            print(f"  Error: {e}")
            print()


def suggest_patterns_for_words():
    """Suggest valid patterns for common words."""
    
    print("\nPattern Suggestions for Common Words:")
    print("-" * 40)
    
    config = EpixConfig.default_testnet()
    pattern_validator = PatternValidator(config.get_address_prefix())
    
    words = ["bitcoin", "ethereum", "cosmos", "epix", "crypto", "defi", "nft", "dao"]
    
    for word in words:
        suggestions = pattern_validator.suggest_patterns(word)
        print(f"Word: '{word}'")
        
        if suggestions:
            for suggestion in suggestions:
                try:
                    pattern = pattern_validator.create_pattern(suggestion, PatternType.PREFIX)
                    pattern_info = pattern_validator.get_pattern_info(pattern)
                    print(f"  Suggestion: '{suggestion}' - {pattern_info['difficulty_description']} "
                          f"({pattern_info['estimated_attempts']:,} attempts)")
                except Exception:
                    print(f"  Suggestion: '{suggestion}' - Error calculating difficulty")
        else:
            print("  No valid suggestions (word contains invalid characters)")
        print()


def test_pattern_matching():
    """Test pattern matching functionality."""
    
    print("\nPattern Matching Tests:")
    print("-" * 30)
    
    config = EpixConfig.default_testnet()
    pattern_validator = PatternValidator(config.get_address_prefix())
    
    # Test addresses (these are examples, not real addresses)
    test_addresses = [
        "epixabc123def456ghi789",
        "epix000111222333444",
        "epixtest987654321",
        "epix123456789abcdef",
        "epixzyx987654321"
    ]
    
    # Test patterns
    test_patterns = [
        ("abc", PatternType.PREFIX),
        ("000", PatternType.PREFIX),
        ("321", PatternType.SUFFIX),
        ("test", PatternType.CONTAINS),
        ("^[0-9]{3}", PatternType.REGEX)
    ]
    
    print("Testing pattern matching:")
    print()
    
    for pattern_str, pattern_type in test_patterns:
        try:
            pattern = pattern_validator.create_pattern(pattern_str, pattern_type)
            print(f"Pattern: '{pattern_str}' ({pattern_type.value})")
            
            for address in test_addresses:
                matches = pattern_validator.matches_pattern(address, pattern)
                status = "✓" if matches else "✗"
                print(f"  {status} {address}")
            print()
        
        except Exception as e:
            print(f"Pattern: '{pattern_str}' - Error: {e}")
            print()


def main():
    """Main function."""
    
    try:
        demonstrate_patterns()
        suggest_patterns_for_words()
        test_pattern_matching()
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
