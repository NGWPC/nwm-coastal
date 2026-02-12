"""Compatibility patches for hydromt bugs."""

from __future__ import annotations


def patch_serialize_crs() -> None:
    """Fix hydromt ``_serialize_crs`` crashing on CRS without an authority.

    hydromt's ``_serialize_crs`` calls ``list(crs.to_authority())`` without
    guarding against ``to_authority()`` returning ``None``, which raises
    ``TypeError: 'NoneType' object is not iterable`` for any CRS that has
    no EPSG code and no recognised authority (e.g. custom proj strings).

    This patch rewrites the function's code object in-place so that even
    references already captured by Pydantic's ``PlainSerializer`` pick up
    the fix.
    """
    try:
        import hydromt.typing.crs as _crs_mod
    except ImportError:
        return

    _original = _crs_mod._serialize_crs

    # Only patch once
    if getattr(_original, "_patched", False):
        return

    def _safe_serialize_crs(crs):  # type: ignore[no-untyped-def]
        epsg = crs.to_epsg()
        if epsg:
            return epsg
        auth = crs.to_authority()
        if auth is not None:
            return list(auth)
        return crs.to_wkt()

    # Replace the *code* of the original function so Pydantic's
    # already-captured reference to the function object sees the fix.
    _original.__code__ = _safe_serialize_crs.__code__
    _original._patched = True  # type: ignore[attr-defined]
