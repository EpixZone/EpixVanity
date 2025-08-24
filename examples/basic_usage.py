#!/usr/bin/env python3
"""Basic usage example for EpixVanity."""

import sys
import os

# Add the parent directory to the path so we can import epix_vanity
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from epix_vanity.core.generator import VanityGenerator
from epix_vanity.core.config import EpixConfig
from epix_vanity.utils.patterns import Pattern, PatternType, PatternValidator


def main():
    """Basic vanity address generation example."""
    
    print("EpixVanity Basic Usage Example")
    print("=" * 40)
    
    # Create configuration
    config = EpixConfig.default_testnet()
    print(f"Using chain: {config.chain_name} ({config.chain_id})")
    print(f"Address prefix: {config.get_address_prefix()}")
    
    # Create pattern validator
    pattern_validator = PatternValidator(config.get_address_prefix())
    
    # Define a simple pattern
    pattern_str = "abc"
    pattern = pattern_validator.create_pattern(pattern_str, PatternType.PREFIX)
    
    # Get pattern information
    pattern_info = pattern_validator.get_pattern_info(pattern)
    print(f"\nPattern: {pattern.pattern}")
    print(f"Type: {pattern.pattern_type.value}")
    print(f"Estimated difficulty: {pattern_info['difficulty_description']}")
    print(f"Expected attempts: {pattern_info['estimated_attempts']:,}")
    
    # Create generator
    generator = VanityGenerator(config=config, num_threads=2)
    print(f"\nUsing {generator.num_threads} CPU threads")
    
    # Generate vanity address
    print("\nStarting generation...")
    print("Press Ctrl+C to stop")
    
    try:
        result = generator.generate_vanity_address(
            pattern=pattern,
            max_attempts=100000,  # Limit attempts for demo
            timeout=30.0  # 30 second timeout
        )
        
        if result.success:
            print("\n🎉 SUCCESS!")
            print(f"Address: {result.keypair.address}")
            print(f"Ethereum Address: {result.keypair.eth_address}")
            print(f"Private Key: {result.keypair.private_key.hex()}")
            print(f"Attempts: {result.attempts:,}")
            print(f"Time: {result.elapsed_time:.2f}s")
            
            if result.attempts > 0:
                rate = result.attempts / result.elapsed_time
                print(f"Rate: {rate:.0f} attempts/s")
        else:
            print(f"\n❌ Failed: {result.error}")
            print(f"Attempts: {result.attempts:,}")
            print(f"Time: {result.elapsed_time:.2f}s")
    
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
