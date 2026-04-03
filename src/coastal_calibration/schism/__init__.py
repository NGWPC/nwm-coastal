"""Main module for the schism subpackage."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coastal_calibration.schism.project_reader import NWMSCHISMProject
    from coastal_calibration.schism.subsetter import divide_mesh

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "NWMSCHISMProject": ("coastal_calibration.schism.project_reader", "NWMSCHISMProject"),
    "divide_mesh": ("coastal_calibration.schism.subsetter", "divide_mesh"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "NWMSCHISMProject",
    "divide_mesh",
]
