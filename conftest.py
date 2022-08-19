import sys
import pytest
import npstructures
#import numpy as np
import tests.npbackend as npb

def pytest_addoption(parser):
    parser.addoption("--cupy", action="store_true", default=False,
            help="Run cupy tests.")

def pytest_configure(config):
    if "cupy" in config.getoption("-m"):
        print("CUPY!!")
        import cupy as cp
        npstructures.set_backend(cp)
        npb.np = cp

