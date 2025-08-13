"""
Standard check implementations for FEP.

Imports all standard checks to trigger their registration.
"""

from .attribute_exists import AttributeExistsCheck_v1_0_0
from .exact_match import ExactMatchCheck_v1_0_0
from .contains import ContainsCheck_v1_0_0
from .is_empty import IsEmptyCheck_v1_0_0
from .regex import RegexCheck_v1_0_0
from .threshold import ThresholdCheck_v1_0_0

__all__ = [
    "AttributeExistsCheck_v1_0_0",
    "ContainsCheck_v1_0_0",
    "ExactMatchCheck_v1_0_0",
    "IsEmptyCheck_v1_0_0",
    "RegexCheck_v1_0_0",
    "ThresholdCheck_v1_0_0",
]
