"""Data download stage for SCHISM workflow."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast

from coastal_calibration.base import WorkflowStage
from coastal_calibration.data.downloader import download_data

if TYPE_CHECKING:
    from coastal_calibration.data.downloader import CoastalSource


class DownloadStage(WorkflowStage):
    """Download required input data."""

    name = "download"

    @property
    def description(self) -> str:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Build description from the actual configured data sources."""
        sources = ["NWM"]
        try:
            sources.append(self.config.boundary.source.upper())
        except Exception:
            sources.append("coastal")
        return f"Download input data ({', '.join(sources)})"

    def run(self) -> dict[str, Any]:
        """Execute data download."""
        self._update_substep("Configuring download")

        cfg = self.config
        sim = cfg.simulation
        paths = cfg.paths
        download_cfg = cfg.download

        domain = sim.coastal_domain
        meteo_source = sim.meteo_source
        # BoundaryConfig.__post_init__ normalizes the deprecated "tpxo"
        # alias to "harmonic", so by the time we get here the value is
        # a member of CoastalSource. The cast lets pyright see that.
        coastal_source = cast("CoastalSource", cfg.boundary.source)

        end_time = sim.start_date + timedelta(hours=sim.duration_hours + 1)
        output_dir = paths.download_dir

        # Tidal atlas directory (only needed when source is harmonic)
        atlas_path = paths.tidal_atlas_dir if coastal_source == "harmonic" else None

        self._update_substep("Downloading data")
        results = download_data(
            start_time=sim.start_date,
            end_time=end_time,
            output_dir=output_dir,
            domain=domain,
            meteo_source=meteo_source,
            hydro_source="nwm",
            coastal_source=coastal_source,
            tidal_atlas_path=atlas_path,
            timeout=download_cfg.timeout,
            raise_on_error=download_cfg.raise_on_error,
        )

        # Track STOFS file path in result instead of mutating config
        stofs_file = None
        if coastal_source == "stofs" and results.coastal.file_paths:
            stofs_file = results.coastal.file_paths[0]

        errors = []
        for result in results:
            if result.errors and result.failed > 0:
                errors.extend(result.errors)

        if errors and download_cfg.raise_on_error:
            raise RuntimeError(f"Download failed: {'; '.join(errors)}")

        self._log(f"Download complete. Raw files stored in {output_dir}")

        return {
            "output_dir": str(output_dir),
            "meteo": {
                "source": results.meteo.source,
                "total": results.meteo.total_files,
                "successful": results.meteo.successful,
                "failed": results.meteo.failed,
            },
            "hydro": {
                "source": results.hydro.source,
                "total": results.hydro.total_files,
                "successful": results.hydro.successful,
                "failed": results.hydro.failed,
            },
            "coastal": {
                "source": results.coastal.source,
                "total": results.coastal.total_files,
                "successful": results.coastal.successful,
                "failed": results.coastal.failed,
                "stofs_file": str(stofs_file) if stofs_file else None,
            },
            "status": "completed" if not errors else "completed_with_errors",
        }
