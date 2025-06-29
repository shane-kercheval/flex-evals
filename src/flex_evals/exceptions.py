"""
Custom exception hierarchy for flex-evals.

Maps to FEP protocol error types and provides detailed context for debugging.
"""



class FlexEvalsError(Exception):
    """Base exception for flex-evals package."""

    pass


class JSONPathError(FlexEvalsError):
    """
    JSONPath resolution errors.

    Maps to protocol error type: jsonpath_error
    """

    def __init__(self, message: str, jsonpath_expression: str | None = None):
        super().__init__(message)
        self.jsonpath_expression = jsonpath_expression


class ValidationError(FlexEvalsError):
    """
    Schema validation errors.

    Maps to protocol error type: validation_error
    """

    pass


class CheckExecutionError(FlexEvalsError):
    """
    Check execution errors.

    Maps to protocol error type: unknown_error (unless more specific type available)
    """

    pass
