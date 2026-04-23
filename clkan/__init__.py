from importlib import import_module

__all__ = ["CliffordKAN", "Norms", "train_kans"]

version = "0.0.1"


def __getattr__(name):
    if name in {"CliffordKAN", "Norms"}:
        module = import_module(".models.CliffordKAN", __name__)
        return getattr(module, name)
    if name == "train_kans":
        module = import_module(".train.train_loop", __name__)
        return module.train_kans
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
