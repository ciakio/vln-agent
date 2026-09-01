"""Process-local runtime selection for existing Skills."""

_runtime = None


def set_runtime(runtime):
    global _runtime
    _runtime = runtime


def get_runtime():
    return _runtime


def clear_runtime():
    set_runtime(None)
