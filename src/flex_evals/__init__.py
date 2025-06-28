"""
Flexible Evaluation Protocol (FEP) - Python Implementation.

A vendor-neutral, schema-driven standard for measuring the quality of any system
that produces complex or variable outputs.
"""

from .engine import evaluate
from . import schemas

__version__ = "0.1.0"
__all__ = ["evaluate", "schemas"]
