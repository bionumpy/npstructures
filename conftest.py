import pytest

def pytest_addoption(parser):
    parser.addoption(
            "--cupy",
            action="store",
            type=bool,
            default=False,
            help="Run cupy tests.")

def pytest_configure(config):
    config.addinivalue_line(
            "markers", "
