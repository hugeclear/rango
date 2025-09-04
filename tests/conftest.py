"""
pytest configuration for Chameleon regression tests.
"""

import pytest
import torch
import logging

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests

def pytest_configure(config):
    """Configure pytest settings."""
    # Set torch to use CPU by default in tests unless explicitly testing GPU
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available - some tests may be skipped", allow_module_level=True)

@pytest.fixture(scope="session")
def device():
    """Provide consistent device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Ensure reproducible results
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Set deterministic mode for testing
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False