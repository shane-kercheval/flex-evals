"""Shared utilities for set operation checks."""


def _elements_equal(a: object, b: object, case_sensitive: bool) -> bool:
    """Compare two elements, applying case-insensitive matching for strings."""
    if not case_sensitive and isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    return a == b


def _partition_elements(
    source: list | set,
    target: list | set,
    case_sensitive: bool,
) -> tuple[list, list]:
    """
    Partition `source` elements into those found and not found in `target`.

    Returns (found, not_found) using original (pre-normalized) values from source.
    Comparison uses Python == for non-string elements. For string elements,
    applies case-insensitive comparison when case_sensitive=False.
    Uses list iteration (not Python set()) to support unhashable elements.
    Duplicates in source are ignored — each unique element is checked once.
    Target is not deduplicated because membership lookup (is this element in target?)
    is unaffected by duplicates in the search space.
    """
    target_list = list(target)

    # Deduplicate source preserving first occurrence
    unique_source: list = []
    for item in source:
        if not any(_elements_equal(item, existing, case_sensitive) for existing in unique_source):
            unique_source.append(item)

    found: list = []
    not_found: list = []
    for item in unique_source:
        if any(_elements_equal(item, t, case_sensitive) for t in target_list):
            found.append(item)
        else:
            not_found.append(item)

    return found, not_found
