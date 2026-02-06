"""Command-line interface for coastal calibration workflow."""

from __future__ import annotations

from pathlib import Path

import rich_click as click

from coastal_calibration.config.schema import CoastalCalibConfig, CoastalDomain
from coastal_calibration.runner import CoastalCalibRunner
from coastal_calibration.utils.logging import configure_logger, logger


class CLIError(click.ClickException):
    """CLI error with formatted message."""

    def format_message(self) -> str:
        """Return the error message without 'Error:' prefix."""
        return self.message


def _raise_cli_error(message: str) -> None:
    """Raise a CLIError with the given message."""
    raise CLIError(message)


@click.group()
@click.version_option()
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
@click.option(
    "--inside-slurm",
    is_flag=True,
    hidden=True,
    help="Indicate running inside a SLURM job (internal use).",
)
def run(
    config: Path,
    start_from: str | None,
    stop_after: str | None,
    dry_run: bool,
    inside_slurm: bool,
) -> None:
    """Run the calibration workflow.

    CONFIG is the path to a YAML configuration file.
    """
    config_path = config.resolve()

    try:
        cfg = CoastalCalibConfig.from_yaml(config_path)
        runner = CoastalCalibRunner(cfg)

        if dry_run:
            logger.info("Dry run mode - validating configuration...")

        result = runner.run(
            start_from=start_from,
            stop_after=stop_after,
            dry_run=dry_run,
            inside_slurm=inside_slurm,
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
    "-i",
    "--interactive",
    is_flag=True,
    help="Wait for job completion and show status updates.",
)
def submit(config: Path, interactive: bool) -> None:
    """Submit workflow as a SLURM job.

    CONFIG is the path to a YAML configuration file.

    By default, submits the job and returns immediately after submission.
    Use --interactive to wait and monitor the job until completion.
    The download step always runs on the login node before submission.
    """
    config_path = config.resolve()

    try:
        cfg = CoastalCalibConfig.from_yaml(config_path)
        cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)

        configure_logger(level="INFO")

        runner = CoastalCalibRunner(cfg)
        result = runner.submit(wait=interactive, log_file=None)

        if result.success:
            if interactive:
                logger.info(f"Job {result.job_id} completed successfully.")
            else:
                # Job submitted, show where to find SLURM output
                slurm_log_path = cfg.paths.work_dir / f"slurm-{result.job_id}.out"
                logger.info(f"Job {result.job_id} submitted.")
                logger.info(
                    f"Once the job starts running, SLURM logs will be written to: {slurm_log_path}"
                )
                logger.info(f"Check job status with: squeue -j {result.job_id}")
        else:
            for error in result.errors:
                logger.error(f"  - {error}")
            if result.job_id:
                slurm_err_path = cfg.paths.work_dir / f"slurm-{result.job_id}.err"
                logger.error(f"Check SLURM error log for details: {slurm_err_path}")
            _raise_cli_error("Job failed (see above).")

    except CLIError:
        raise
    except Exception as e:
        _raise_cli_error(str(e))


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


@cli.command()
@click.argument(
    "output",
    type=click.Path(path_type=Path),
    default="coastal_calibration.yaml",
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
def init(output: Path, domain: CoastalDomain, force: bool) -> None:
    """Create a minimal configuration file.

    OUTPUT is the path where the configuration will be written
    (default: coastal_calibration.yaml).

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

    import os

    meteo_source, boundary_source, start_date = get_default_sources(domain)
    start_date_str = start_date.strftime("%Y-%m-%d")
    username = os.environ.get("USER", "YOUR_USERNAME")

    # Generate minimal config with comments
    config_content = f"""\
# Minimal SCHISM configuration for {domain} domain
#
# Paths are auto-generated based on user, domain, and source:
#   work_dir: /ngen-test/coastal/${{slurm.user}}/schism_${{simulation.coastal_domain}}_${{boundary.source}}_${{simulation.meteo_source}}/schism_${{simulation.start_date}}
#   raw_download_dir: /ngen-test/coastal/${{slurm.user}}/schism_${{simulation.coastal_domain}}_${{boundary.source}}_${{simulation.meteo_source}}/raw_data
#
# Usage:
#   coastal-calibration validate {output_path.name}
#   coastal-calibration submit {output_path.name}
#   coastal-calibration submit {output_path.name} -i  # wait for completion

slurm:
  job_name: coastal_calibration
  user: {username}

simulation:
  start_date: {start_date_str}
  duration_hours: 3
  coastal_domain: {domain}
  meteo_source: {meteo_source}

boundary:
  source: {boundary_source}
"""

    output_path.write_text(config_content)
    logger.info(f"Configuration written to: {output_path}")


@cli.command()
def stages() -> None:
    """List available workflow stages."""
    stage_list = [
        ("download", "Download NWM/STOFS data (optional)"),
        ("pre_forcing", "Prepare NWM forcing data"),
        ("nwm_forcing", "Generate atmospheric forcing (MPI)"),
        ("post_forcing", "Post-process forcing data"),
        ("update_params", "Create SCHISM param.nml"),
        ("boundary_conditions", "Generate boundary conditions (TPXO/STOFS)"),
        ("pre_schism", "Prepare SCHISM inputs"),
        ("schism_run", "Run SCHISM model (MPI)"),
        ("post_schism", "Post-process SCHISM outputs"),
    ]

    click.echo("Workflow stages:")
    for i, (name, desc) in enumerate(stage_list, 1):
        click.echo(f"  {i}. {name}: {desc}")


def main() -> None:
    """Run the main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
