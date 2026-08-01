"""Matrix-free learning and error-controlled recovery of causal QAOA landscapes."""

__version__ = "3.2.0"


def reproduce(*args, **kwargs):
    """Load the experiment lazily so ``python -m`` remains warning-free."""

    from .experiment import reproduce as _reproduce

    return _reproduce(*args, **kwargs)


__all__ = ["__version__", "reproduce"]
