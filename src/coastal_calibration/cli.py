"""Command-line interface for coastal calibration workflow."""

from __future__ import annotations

import atexit
import contextlib
import sys
from pathlib import Path

import rich_click as click

from coastal_calibration.config.schema import CoastalCalibConfig, CoastalDomain, ModelType
from coastal_calibration.runner import CoastalCalibRunner
from coastal_calibration.utils.logging import configure_logger, logger


# --- Suppress "Error in sys.excepthook" cascade at shutdown -----------
# During interpreter teardown, Rich / hydromt destructors may raise while
# stderr is already half-closed.  A no-op hook installed via atexit
# (which runs *before* module globals are set to None) prevents the
# cascade entirely.  The lambda uses no module globals so it survives
# teardown even if the cli module's namespace is already cleared.
def _silence_shutdown_stderr() -> None:
    """Redirect stderr to ``/dev/null`` at the OS level.

    During interpreter teardown ``sys.__dict__`` is cleared, so any
    Python-level ``sys.excepthook`` we install is gone by the time
    destructors run.  CPython's C code writes ``Error in
    sys.excepthook:`` directly to fd 2 — the only way to suppress it
    is to redirect the file descriptor itself.
    """
    import os as _os

    with contextlib.suppress(Exception):
        sys.stderr.flush()
    try:
        _devnull = _os.open(_os.devnull, _os.O_WRONLY)
        _os.dup2(_devnull, 2)
        _os.close(_devnull)
    except OSError:
        pass


atexit.register(_silence_shutdown_stderr)


class CLIError(click.ClickException):
    """CLI error with formatted message."""

    def format_message(self) -> str:
        """Return the error message without 'Error:' prefix."""
        return self.message


def _raise_cli_error(message: str) -> None:
    """Raise a CLIError with the given message."""
    raise CLIError(message)


@click.group()
@click.version_option(package_name="coastal_calibration")
def cli() -> None:
    """Coastal calibration workflow manager (SCHISM, SFINCS)."""


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--start-from",
    type=str,
    help="Stage to start from (skip earlier stages).",
)
@click.option(
    "--stop-after",
    type=str,
    help="Stage to stop after (skip later stages).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate configuration without executing.",
)
def run(
    config: Path,
    start_from: str | None,
    stop_after: str | None,
    dry_run: bool,
) -> None:
    """Run the calibration workflow.

    CONFIG is the path to a YAML configuration file.
    """
    config_path = config.resolve()

    try:
        cfg = CoastalCalibConfig.from_yaml(config_path)
        runner = CoastalCalibRunner(cfg)
        configure_logger(level="INFO")

        if dry_run:
            logger.info("Dry run mode - validating configuration...")

        result = runner.run(
            start_from=start_from,
            stop_after=stop_after,
            dry_run=dry_run,
        )

        if result.success:
            logger.info("Workflow completed successfully.")
        else:
            for error in result.errors:
                logger.error(f"  - {error}")
            _raise_cli_error("Workflow failed with errors (see above).")

    except CLIError:
        raise
    except Exception as e:
        _raise_cli_error(str(e))


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--start-from",
    type=str,
    help="Stage to start from (skip earlier stages).",
)
@click.option(
    "--stop-after",
    type=str,
    help="Stage to stop after (skip later stages).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate configuration without executing.",
)
def create(
    config: Path,
    start_from: str | None,
    stop_after: str | None,
    dry_run: bool,
) -> None:
    """Create a SFINCS model from an AOI polygon.

    CONFIG is the path to a YAML configuration file.
    """
    import os

    from coastal_calibration.config.create_schema import SfincsCreateConfig
    from coastal_calibration.creator import SfincsCreator

    config_path = config.resolve()

    # Redirect stdout to /dev/null for the entire create workflow.
    # hydromt-sfincs's quadtree builders use raw print() calls that
    # cannot be silenced through the logging system.  All our own
    # output goes to stderr via RichHandler so nothing is lost.
    _devnull = os.open(os.devnull, os.O_WRONLY)
    _saved_stdout = os.dup(1)
    os.dup2(_devnull, 1)
    os.close(_devnull)

    try:
        cfg = SfincsCreateConfig.from_yaml(config_path)
        creator = SfincsCreator(cfg)
        configure_logger(level="INFO")

        if dry_run:
            logger.info("Dry run mode - validating configuration...")

        result = creator.run(
            start_from=start_from,
            stop_after=stop_after,
            dry_run=dry_run,
        )

        if result.success:
            logger.info("Model creation completed successfully.")
            if not dry_run:
                logger.info(f"Output directory: {cfg.output_dir}")
        else:
            for error in result.errors:
                logger.error(f"  - {error}")
            _raise_cli_error("Model creation failed with errors (see above).")

    except CLIError:
        raise
    except Exception as e:
        _raise_cli_error(str(e))
    finally:
        # Do NOT restore stdout — leave it redirected to /dev/null.
        # hydromt-sfincs's quadtree builder fires raw print() calls
        # during lazy XUGrid construction triggered by SfincsModel
        # garbage collection after this function returns.  All our
        # own output goes to stderr (RichHandler), so nothing is lost.
        os.close(_saved_stdout)


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
def validate(config: Path) -> None:
    """Validate a configuration file.

    CONFIG is the path to a YAML configuration file.
    """
    config_path = config.resolve()

    try:
        cfg = CoastalCalibConfig.from_yaml(config_path)
        runner = CoastalCalibRunner(cfg)
        errors = runner.validate()

        if errors:
            for error in errors:
                logger.error(f"  - {error}")
            _raise_cli_error("Validation failed (see above).")

        logger.info("Configuration is valid.")

    except CLIError:
        raise
    except Exception as e:
        _raise_cli_error(str(e))


@cli.command("prepare-topobathy")
@click.argument("aoi", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--domain",
    type=str,
    required=True,
    help="Coastal domain (atlgulf, hi, prvi, pacific, ak).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for GeoTIFF + catalog (default: same as AOI).",
)
@click.option(
    "--buffer-deg",
    type=float,
    default=0.1,
    help="BBox buffer in degrees (default: 0.1).",
)
def prepare_topobathy(
    aoi: Path,
    domain: str,
    output_dir: Path | None,
    buffer_deg: float,
) -> None:
    """Download NWS topobathy DEM clipped to an AOI bounding box.

    Fetches the NWS 30 m topo-bathymetric DEM from icechunk (S3),
    clips it to the AOI extent, saves a local GeoTIFF, and writes a
    HydroMT data-catalog YAML so the ``create`` workflow can use it
    as an elevation source.

    Requires AWS credentials (via environment or ~/.aws) and the
    ``icechunk`` Python package.
    """
    from coastal_calibration.utils.topobathy import fetch_topobathy

    if output_dir is None:
        output_dir = aoi.resolve().parent

    configure_logger(level="INFO")

    try:
        tif_path, cat_path = fetch_topobathy(
            domain=domain,
            aoi=aoi.resolve(),
            output_dir=output_dir.resolve(),
            buffer_deg=buffer_deg,
        )
    except Exception as e:
        _raise_cli_error(str(e))
        return  # unreachable, but helps type-checkers

    click.echo(f"\nSaved:   {tif_path}")
    click.echo(f"Catalog: {cat_path}")
    click.echo(
        "\nUpdate your create config:\n"
        "  elevation:\n"
        "    datasets:\n"
        "      - name: nws_topobathy\n"
        "        zmin: -20000\n"
        "  data_catalog:\n"
        "    data_libs:\n"
        f"      - {cat_path}\n"
    )


@cli.command()
@click.argument(
    "output",
    type=click.Path(path_type=Path),
)
@click.option(
    "--domain",
    type=click.Choice(["prvi", "hawaii", "atlgulf", "pacific"]),
    default="pacific",
    help="Coastal domain.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Overwrite existing file without prompting.",
)
@click.option(
    "--model",
    type=click.Choice(["schism", "sfincs"]),
    default="schism",
    help="Model type (default: schism).",
)
def init(output: Path, domain: CoastalDomain, force: bool, model: ModelType) -> None:
    """Create a minimal configuration file.

    OUTPUT is the path where the configuration will be written.

    The generated config includes only required fields. Paths are auto-generated
    based on user, domain, and source settings.
    """
    from coastal_calibration.downloader import get_default_sources

    output_path = output.resolve()

    if (
        output_path.exists()
        and not force
        and not click.confirm(f"File {output_path} exists. Overwrite?")
    ):
        raise click.Abort()

    meteo_source, boundary_source, start_date = get_default_sources(domain)
    start_date_str = start_date.strftime("%Y-%m-%d")

    if model == "sfincs":
        config_content = f"""\
# Minimal SFINCS configuration for {domain} domain
#
# Paths are auto-generated based on $USER, domain, and source:
#   work_dir: /ngen-test/coastal/${{user}}/sfincs_${{simulation.coastal_domain}}_${{boundary.source}}_${{simulation.meteo_source}}/sfincs_${{simulation.start_date}}
#   raw_download_dir: /ngen-test/coastal/${{user}}/sfincs_${{simulation.coastal_domain}}_${{boundary.source}}_${{simulation.meteo_source}}/raw_data
#
# Usage:
#   coastal-calibration validate {output_path.name}
#   coastal-calibration run {output_path.name}
#   coastal-calibration run {output_path.name} --dry-run

model: sfincs

simulation:
  start_date: {start_date_str}
  duration_hours: 12
  coastal_domain: {domain}
  meteo_source: {meteo_source}

boundary:
  source: {boundary_source}

model_config:
  prebuilt_dir: /path/to/prebuilt/sfincs/model
"""
    else:
        config_content = f"""\
# Minimal SCHISM configuration for {domain} domain
#
# Paths are auto-generated based on $USER, domain, and source:
#   work_dir: /ngen-test/coastal/${{user}}/schism_${{simulation.coastal_domain}}_${{boundary.source}}_${{simulation.meteo_source}}/schism_${{simulation.start_date}}
#   raw_download_dir: /ngen-test/coastal/${{user}}/schism_${{simulation.coastal_domain}}_${{boundary.source}}_${{simulation.meteo_source}}/raw_data
#
# Usage:
#   coastal-calibration validate {output_path.name}
#   coastal-calibration run {output_path.name}
#   coastal-calibration run {output_path.name} --dry-run

model: schism

simulation:
  start_date: {start_date_str}
  duration_hours: 12
  coastal_domain: {domain}
  meteo_source: {meteo_source}

boundary:
  source: {boundary_source}

model_config:
  include_noaa_gages: true
"""

    output_path.write_text(config_content)
    logger.info(f"Configuration written to: {output_path}")


@cli.command("update-dem-index")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Write index to this path instead of the packaged location.",
)
@click.option(
    "--max-datasets",
    type=int,
    default=None,
    help="Limit S3 scan to N datasets (for testing).",
)
def update_dem_index(output: Path | None, max_datasets: int | None) -> None:
    """Rebuild the NOAA DEM spatial index from S3 STAC metadata.

    Scans the ``noaa-nos-coastal-lidar-pds`` S3 bucket (public,
    anonymous access) for coastal DEM datasets and writes a JSON
    index used by the ``create_fetch_elevation`` stage.
    """
    import importlib.resources
    import json

    from coastal_calibration.utils.noaa_dem import build_index_from_s3

    configure_logger(level="INFO")

    entries = build_index_from_s3(max_datasets=max_datasets)
    if not entries:
        _raise_cli_error("No DEM entries found on S3")

    if output is None:
        ref = importlib.resources.files("coastal_calibration.data_catalog").joinpath(
            "noaa_dem_index.json"
        )
        output = Path(str(ref))

    output.write_text(json.dumps(entries, indent=2) + "\n")
    click.echo(f"Wrote {len(entries)} entries to {output}")


@cli.command()
@click.option(
    "--model",
    type=click.Choice(["schism", "sfincs", "create"]),
    default=None,
    help="Show stages for a specific model/workflow (default: show all).",
)
def stages(model: str | None) -> None:
    """List available workflow stages."""
    schism_stages = [
        ("download", "Download NWM/STOFS data (optional)"),
        ("pre_forcing", "Prepare NWM forcing data"),
        ("nwm_forcing", "Generate atmospheric forcing (MPI)"),
        ("post_forcing", "Post-process forcing data"),
        ("schism_obs", "Add NOAA observation stations"),
        ("update_params", "Create SCHISM param.nml"),
        ("boundary_conditions", "Generate boundary conditions (TPXO/STOFS)"),
        ("pre_schism", "Prepare SCHISM inputs"),
        ("schism_run", "Run SCHISM model (MPI)"),
        ("post_schism", "Post-process SCHISM outputs"),
        ("schism_plot", "Plot simulated vs observed water levels"),
    ]

    sfincs_stages = [
        ("download", "Download NWM/STOFS data (optional)"),
        ("sfincs_symlinks", "Create .nc symlinks for NWM data"),
        ("sfincs_data_catalog", "Generate HydroMT data catalog"),
        ("sfincs_init", "Initialise SFINCS model (pre-built)"),
        ("sfincs_timing", "Set SFINCS timing"),
        ("sfincs_forcing", "Add water level forcing"),
        ("sfincs_obs", "Add observation points"),
        ("sfincs_discharge", "Add discharge sources"),
        ("sfincs_precip", "Add precipitation forcing"),
        ("sfincs_wind", "Add wind forcing"),
        ("sfincs_pressure", "Add atmospheric pressure forcing"),
        ("sfincs_write", "Write SFINCS model"),
        ("sfincs_run", "Run SFINCS model (Singularity)"),
        ("sfincs_plot", "Plot simulated vs observed water levels"),
    ]

    create_stages_list = [
        ("create_grid", "Create SFINCS grid from AOI polygon"),
        ("create_fetch_elevation", "Fetch NOAA topobathy DEM for AOI"),
        ("create_elevation", "Add elevation and bathymetry data"),
        ("create_mask", "Create active cell mask"),
        ("create_boundary", "Create water level boundary cells"),
        ("create_subgrid", "Create subgrid tables"),
        ("create_write", "Write SFINCS model to disk"),
    ]

    def _print_stages(title: str, stage_list: list[tuple[str, str]]) -> None:
        click.echo(f"{title}:")
        for i, (name, desc) in enumerate(stage_list, 1):
            click.echo(f"  {i}. {name}: {desc}")

    if model == "schism":
        _print_stages("SCHISM workflow stages", schism_stages)
    elif model == "sfincs":
        _print_stages("SFINCS workflow stages", sfincs_stages)
    elif model == "create":
        _print_stages("SFINCS creation stages (create subcommand)", create_stages_list)
    else:
        _print_stages("SCHISM workflow stages", schism_stages)
        click.echo()
        _print_stages("SFINCS workflow stages", sfincs_stages)
        click.echo()
        _print_stages("SFINCS creation stages (create subcommand)", create_stages_list)


def main() -> None:
    """Run the main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
