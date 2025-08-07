"""
Standard check implementations for FEP.

Imports all standard checks to trigger their registration.
"""

from .attribute_exists import AttributeExistsCheck
from .exact_match import ExactMatchCheck
from .contains import ContainsCheck
from .is_empty import IsEmptyCheck
from .regex import RegexCheck
from .threshold import ThresholdCheck

__all__ = [
    "AttributeExistsCheck",
    "ContainsCheck",
    "ExactMatchCheck",
    "IsEmptyCheck",
    "RegexCheck",
    "ThresholdCheck",
]
