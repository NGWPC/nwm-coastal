"""Compatibility patches for hydromt bugs."""

from __future__ import annotations


def patch_serialize_crs() -> None:
    """Fix hydromt ``_serialize_crs`` crashing on CRS without an authority.

    hydromt's ``_serialize_crs`` calls ``list(crs.to_authority())`` without
    guarding against ``to_authority()`` returning ``None``, which raises
    ``TypeError: 'NoneType' object is not iterable`` for any CRS that has
    no EPSG code and no recognised authority (e.g. custom proj strings).

    This patch replaces the function with a safe version that falls back
    to WKT when both ``to_epsg()`` and ``to_authority()`` return ``None``.
    """
    try:
        import hydromt.typing.crs as _crs_mod
    except ImportError:
        return

    _original = _crs_mod._serialize_crs

    def _safe_serialize_crs(crs):  # type: ignore[no-untyped-def]
        epsg = crs.to_epsg()
        if epsg:
            return epsg
        auth = crs.to_authority()
        if auth is not None:
            return list(auth)
        return crs.to_wkt()

    # Only patch once
    if getattr(_crs_mod._serialize_crs, "_patched", False):
        return

    _crs_mod._serialize_crs = _safe_serialize_crs
    _crs_mod._serialize_crs._patched = True  # type: ignore[attr-defined]
