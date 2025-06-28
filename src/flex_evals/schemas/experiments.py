"""Experiment metadata schema implementation for FEP."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ExperimentMetadata:
    """
    Provides optional context about the evaluation experiment for tracking and comparison purposes.

    Optional Fields:
    - name: Human-readable experiment identifier
    - metadata: Additional experiment-specific information
    """

    name: str | None = None
    metadata: dict[str, Any] | None = None
