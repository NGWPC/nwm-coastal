"""Translate an NWM discharge crosswalk into a NextGen hydrofabric one.

``nwmReaches.csv`` maps SCHISM source/sink elements to **NWM COMIDs**.  For
``ngen_forecast`` runs the discharge instead comes from t-route, which keys
its output on the **16-digit NextGen hydrofabric flowpath id** (``fp_id``).

The NextGen hydrofabric's ``reference_flowpaths`` layer already contains the
mapping we need: ``ref_fp_id`` (the NWM COMID) alongside ``fp_id`` (the
16-digit NextGen id).  This module swaps each COMID for its ``fp_id`` while
preserving the element ids **and** the source/sink block structure — so the
already-computed source/sink assignment in ``nwmReaches.csv`` is reused
rather than re-derived geometrically.

COMIDs that map only to a *virtual* NextGen flowpath (a null ``fp_id`` in
``reference_flowpaths``) are dropped: t-route does not route virtual
flowpaths, so there is no discharge to inject at them.

The block format is identical to ``nwmReaches.csv`` (source block, blank
line, sink block), so the output ``ngenReaches.csv`` is consumed unchanged
by :func:`coastal_calibration.schism.prep.make_discharge` and subset by the
same subsetter path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from coastal_calibration.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

ReachPair = tuple[int, int]  # (element_id, feature_id)


@dataclass
class TranslateStats:
    """Summary of an nwm -> ngen reaches translation."""

    sources_in: int = 0
    sources_out: int = 0
    sinks_in: int = 0
    sinks_out: int = 0
    dropped_comids: set[int] = field(default_factory=set)

    @property
    def dropped(self) -> int:
        """Number of (element, comid) rows dropped (unmapped/virtual)."""
        return (self.sources_in - self.sources_out) + (self.sinks_in - self.sinks_out)


def _read_reaches_blocks(path: Path) -> tuple[list[ReachPair], list[ReachPair]]:
    """Parse ``nwmReaches.csv`` into (sources, sinks) ``(element, comid)`` lists.

    The file is two count-prefixed blocks separated by a blank line: the
    first is sources, the second is sinks.  Matches the reader in
    :func:`coastal_calibration.schism.prep.make_discharge`.
    """
    blocks: list[list[ReachPair]] = []
    with path.open() as f:
        lines = f.readlines()
    i = 0
    n = len(lines)
    while i < n and len(blocks) < 2:
        stripped = lines[i].strip()
        i += 1
        if not stripped:
            continue
        try:
            count = int(stripped)
        except ValueError:
            continue
        pairs: list[ReachPair] = []
        for _ in range(count):
            parts = lines[i].split()
            i += 1
            if len(parts) >= 2:
                pairs.append((int(parts[0]), int(parts[1])))
        blocks.append(pairs)
    if not blocks:
        raise ValueError(f"No reach blocks parsed from {path}")
    sources = blocks[0]
    sinks = blocks[1] if len(blocks) > 1 else []
    return sources, sinks


def load_comid_to_fpid(gpkg: Path) -> dict[int, int]:
    """Build a ``COMID -> 16-digit fp_id`` map from a NextGen hydrofabric.

    Uses a two-hop bridge that works across CONUS and oCONUS domains:

    1. ``nhd`` layer: ``nhd_feature_id`` (the NWM COMID used in
       ``nwmReaches.csv``) -> ``ref_id`` (the NextGen reference id).
    2. ``reference_flowpaths`` layer: ``ref_fp_id`` (== ``ref_id``) ->
       ``fp_id`` (the 16-digit NextGen flowpath id t-route keys on).

    For CONUS the NextGen ``ref_id`` happens to equal the NWM COMID, so hop
    1 is an identity; for Hawaii/PRVI the two ids differ and the ``nhd``
    layer is what connects them.  Reference flowpaths with a null ``fp_id``
    (virtual flowpaths, not routed by t-route) are skipped.

    Notes
    -----
    ``fp_id`` is read as ``float`` when the column carries nulls, but the
    id magnitudes (~1e15) are below float64's exact-integer limit
    (2**53 ~= 9e15), so the int cast is loss-free.
    """
    import geopandas as gpd

    nhd = gpd.read_file(gpkg, layer="nhd", columns=["nhd_feature_id", "ref_id"]).dropna(
        subset=["nhd_feature_id", "ref_id"]
    )
    rfp = gpd.read_file(gpkg, layer="reference_flowpaths", columns=["ref_fp_id", "fp_id"]).dropna(
        subset=["ref_fp_id", "fp_id"]
    )

    # hop 2: ref_id -> fp_id
    ref_to_fp: dict[int, int] = {}
    for ref_fp_id, fpid in zip(rfp["ref_fp_id"], rfp["fp_id"], strict=True):
        r = int(ref_fp_id)
        if r not in ref_to_fp:
            ref_to_fp[r] = int(fpid)

    # hop 1 + compose: comid -> ref_id -> fp_id
    mapping: dict[int, int] = {}
    for comid, ref_id in zip(nhd["nhd_feature_id"], nhd["ref_id"], strict=True):
        c = int(comid)
        if c in mapping:
            continue
        fpid = ref_to_fp.get(int(ref_id))
        if fpid is not None:
            mapping[c] = fpid
    logger.info("Loaded %d COMID->fp_id mappings from %s", len(mapping), gpkg.name)
    return mapping


def _translate_block(
    pairs: Sequence[ReachPair], comid2fp: dict[int, int], dropped: set[int]
) -> list[ReachPair]:
    """Swap each ``(element, comid)`` for ``(element, fp_id)``; drop unmapped."""
    out: list[ReachPair] = []
    for elem, comid in pairs:
        fpid = comid2fp.get(comid)
        if fpid is None:
            dropped.add(comid)
            continue
        out.append((elem, fpid))
    return out


def _write_reaches_blocks(
    path: Path, sources: Sequence[ReachPair], sinks: Sequence[ReachPair]
) -> None:
    """Write the source/sink blocks in ``nwmReaches.csv`` format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"{len(sources)}\n")
        for elem, fid in sources:
            f.write(f"{elem} {fid}\n")
        f.write("\n")
        f.write(f"{len(sinks)}\n")
        for elem, fid in sinks:
            f.write(f"{elem} {fid}\n")
        f.write("\n")


def translate_nwm_to_ngen_reaches(
    nwm_reaches: Path,
    gpkg: Path,
    output: Path,
) -> TranslateStats:
    """Translate ``nwmReaches.csv`` -> ``ngenReaches.csv`` via a NextGen gpkg.

    Parameters
    ----------
    nwm_reaches : Path
        Existing ``nwmReaches.csv`` (element -> NWM COMID, with source/sink
        blocks).
    gpkg : Path
        NextGen hydrofabric GeoPackage (v1.2.2+ / 16-digit ``fp_id``) whose
        ``reference_flowpaths`` layer provides the COMID -> ``fp_id`` map.
    output : Path
        Destination ``ngenReaches.csv``.

    Returns
    -------
    TranslateStats
        Counts of sources/sinks in and out, and the dropped (virtual/
        unmapped) COMIDs.
    """
    sources, sinks = _read_reaches_blocks(nwm_reaches)
    comid2fp = load_comid_to_fpid(gpkg)

    stats = TranslateStats(sources_in=len(sources), sinks_in=len(sinks))
    new_sources = _translate_block(sources, comid2fp, stats.dropped_comids)
    new_sinks = _translate_block(sinks, comid2fp, stats.dropped_comids)
    stats.sources_out = len(new_sources)
    stats.sinks_out = len(new_sinks)

    _write_reaches_blocks(output, new_sources, new_sinks)

    logger.info(
        "Wrote %s: sources %d/%d, sinks %d/%d (dropped %d rows, %d unique COMIDs "
        "with no routed fp_id)",
        output,
        stats.sources_out,
        stats.sources_in,
        stats.sinks_out,
        stats.sinks_in,
        stats.dropped,
        len(stats.dropped_comids),
    )
    return stats


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Translate nwmReaches.csv -> ngenReaches.csv via a NextGen hydrofabric."
    )
    parser.add_argument("nwm_reaches", type=Path, help="Path to nwmReaches.csv")
    parser.add_argument("gpkg", type=Path, help="Path to NextGen hydrofabric .gpkg (16-digit fp_id)")
    parser.add_argument("output", type=Path, help="Output ngenReaches.csv path")
    args = parser.parse_args()
    translate_nwm_to_ngen_reaches(args.nwm_reaches, args.gpkg, args.output)


if __name__ == "__main__":
    _main()
