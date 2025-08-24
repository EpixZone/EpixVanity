#!/usr/bin/env python3
"""Simple test runner for EpixVanity."""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run the test suite."""
    
    print("EpixVanity Test Runner")
    print("=" * 30)
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest is not installed. Install with: pip install pytest")
        return False
    
    # Run tests
    test_dir = Path(__file__).parent / "tests"
    
    if not test_dir.exists():
        print("❌ Tests directory not found")
        return False
    
    print(f"Running tests from: {test_dir}")
    print()
    
    # Run pytest
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        str(test_dir),
        "-v",
        "--tb=short"
    ])
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False


def run_examples():
    """Run example scripts."""
    
    print("\nRunning Examples")
    print("=" * 20)
    
    examples_dir = Path(__file__).parent / "examples"
    
    if not examples_dir.exists():
        print("❌ Examples directory not found")
        return False
    
    examples = [
        "pattern_examples.py",
        "basic_usage.py"
    ]
    
    for example in examples:
        example_path = examples_dir / example
        if example_path.exists():
            print(f"\n🔄 Running {example}...")
            try:
                result = subprocess.run([
                    sys.executable, str(example_path)
                ], timeout=30, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {example} completed successfully")
                else:
                    print(f"❌ {example} failed:")
                    print(result.stderr)
            except subprocess.TimeoutExpired:
                print(f"⏰ {example} timed out")
            except Exception as e:
                print(f"❌ Error running {example}: {e}")
        else:
            print(f"⚠️  {example} not found")
    
    return True


def main():
    """Main function."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        run_examples()
    elif len(sys.argv) > 1 and sys.argv[1] == "--all":
        success = run_tests()
        if success:
            run_examples()
    else:
        run_tests()


if __name__ == "__main__":
    main()
