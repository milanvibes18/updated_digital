"""
Digital Twin System Test Suite

This test suite covers:
- AI module functionality
- API endpoint testing
- WebSocket functionality
- Security features
- Data management
- Report generation
- Integration testing
- Performance benchmarks

Run tests with: pytest TESTS/
"""

__version__ = "2.0.0"
__author__ = "Digital Twin Team"

if __name__ == "__main__":
    import pytest
    pytest.main([__file__.replace('__init__.py', ''), '-v'])
