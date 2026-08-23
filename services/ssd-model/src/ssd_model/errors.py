class ModelNotReadyError(RuntimeError):
    pass


class InvalidImageError(ValueError):
    pass


class InferenceFailedError(RuntimeError):
    pass
