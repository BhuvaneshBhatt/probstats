"""Exceptions raised by the Bayesian model core."""


class BayesianInferenceError(Exception):
    """Base exception for this package."""


class ModelValidationError(BayesianInferenceError, ValueError):
    """Raised when a probabilistic model is internally inconsistent."""


class ObservationError(ModelValidationError):
    """Raised when observations cannot be attached to a model."""
