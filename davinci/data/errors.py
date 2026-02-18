"""Custom exceptions for data module."""


class DataError(Exception):
    """Base exception for data-related errors."""

    pass


class EvaluationDataLoadError(DataError):
    """
    Raised when evaluation data cannot be loaded.

    This can happen when:
    - Data directory does not exist
    - Data files are missing or corrupted
    - Data format is invalid (wrong shape, dtype, etc.)
    """

    pass


class EvaluationDataNotFoundError(DataError):
    """
    Raised when no evaluation data is available.

    The subnet owner has not yet provided evaluation data files.
    """

    pass
