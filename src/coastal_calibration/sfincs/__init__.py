"""SFINCS model workflow package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coastal_calibration.sfincs.outputs import load_sfincs_water_level

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "load_sfincs_water_level": ("coastal_calibration.sfincs.outputs", "load_sfincs_water_level"),
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


__all__ = ["load_sfincs_water_level"]
