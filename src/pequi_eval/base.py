from .config import SetupConfig

class EvalVLA:
    """
    Base classe for VLA Evaluation, which holds the default variables to be used.
    """

    def __init__(self, config: SetupConfig) -> None:
        self.config = config

    def metric(fn):
        """Marks metódo from base class as a métric, to use in the report."""
        fn.__pequi_metric__ = True
        return fn
