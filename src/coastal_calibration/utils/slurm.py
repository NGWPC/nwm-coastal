"""SLURM job submission and monitoring utilities."""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from coastal_calibration.config.schema import CoastalCalibConfig
    from coastal_calibration.utils.logging import WorkflowMonitor


class JobState(StrEnum):
    """SLURM job states."""

    PENDING = "PENDING"
    CONFIGURING = "CONFIGURING"
    RUNNING = "RUNNING"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    NODE_FAIL = "NODE_FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass
class JobStatus:
    """SLURM job status information."""

    job_id: str
    state: JobState
    name: str = ""
    node_list: str = ""
    exit_code: int | None = None
    elapsed_time: str = ""
    reason: str = ""


class SlurmManager:
    """Manage SLURM job submission and monitoring."""

    def _check_slurm_available(self) -> None:
        """Check if SLURM commands are available."""
        try:
            subprocess.run(
                ["squeue", "--version"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            msg = "SLURM commands not available. Ensure you are on a cluster with SLURM."
            raise RuntimeError(msg) from e

    def __init__(
        self,
        config: CoastalCalibConfig,
        monitor: WorkflowMonitor | None = None,
    ) -> None:
        self.config = config
        self.monitor = monitor
        self._check_slurm_available()

    def _log(self, message: str, level: str = "info") -> None:
        """Log message if monitor is available."""
        if self.monitor:
            getattr(self.monitor, level)(message)

    def submit_job(self, script_path: Path) -> str:
        """Submit a SLURM job and return job ID."""
        self._log(f"Submitting job: {script_path}")

        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            error_msg = f"Failed to submit job: {result.stderr}"
            self._log(error_msg, "error")
            raise RuntimeError(error_msg)

        job_id_match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not job_id_match:
            msg = f"Could not parse job ID from: {result.stdout}"
            raise RuntimeError(msg)

        job_id = job_id_match.group(1)
        self._log(f"Job submitted with ID: {job_id}")
        return job_id

    def _parse_state(self, state_str: str) -> JobState:
        """Parse SLURM state string to JobState enum."""
        state_str = state_str.strip().upper()
        if state_str.startswith("CANCELLED"):
            return JobState.CANCELLED
        try:
            return JobState(state_str)
        except ValueError:
            return JobState.UNKNOWN

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of a SLURM job."""
        result = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "--format=JobID,JobName,State,NodeList,ExitCode,Elapsed,Reason",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        # Try sacct first, then fall back to squeue
        sacct_status = None
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = line.split("|")
                if parts[0] == job_id or parts[0].startswith(f"{job_id}."):
                    if ".batch" in parts[0] or ".extern" in parts[0]:
                        continue

                    exit_code = None
                    if len(parts) > 4 and parts[4]:
                        with contextlib.suppress(ValueError):
                            exit_code = int(parts[4].split(":")[0])

                    sacct_status = JobStatus(
                        job_id=parts[0],
                        name=parts[1] if len(parts) > 1 else "",
                        state=self._parse_state(parts[2]) if len(parts) > 2 else JobState.UNKNOWN,
                        node_list=parts[3] if len(parts) > 3 else "",
                        exit_code=exit_code,
                        elapsed_time=parts[5] if len(parts) > 5 else "",
                        reason=parts[6] if len(parts) > 6 else "",
                    )
                    break

        terminal_states = {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.TIMEOUT,
            JobState.NODE_FAIL,
        }

        # For terminal states, sacct is authoritative
        if sacct_status and sacct_status.state in terminal_states:
            return sacct_status

        # For active jobs, squeue has the most accurate real-time state
        sq_result = subprocess.run(
            [
                "squeue",
                "-j",
                job_id,
                "--format=%i|%j|%T|%N|%r",
                "--noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if sq_result.returncode == 0 and sq_result.stdout.strip():
            parts = sq_result.stdout.strip().split("|")
            return JobStatus(
                job_id=parts[0],
                name=parts[1] if len(parts) > 1 else "",
                state=self._parse_state(parts[2]) if len(parts) > 2 else JobState.UNKNOWN,
                node_list=parts[3] if len(parts) > 3 else "",
                reason=parts[4] if len(parts) > 4 else "",
            )

        # Return sacct result if we had one, otherwise unknown
        return sacct_status or JobStatus(job_id=job_id, state=JobState.UNKNOWN)

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: int | None = None,
    ) -> JobStatus:
        """Wait for a job to complete and return final status."""
        self._log(f"Waiting for job {job_id} to complete...")
        start_time = time.time()
        last_state: JobState | None = None

        while True:
            status = self.get_job_status(job_id)

            if status.state != last_state:
                self._log(f"Job {job_id} state: {status.state.value}")
                last_state = status.state

            if status.state in (
                JobState.COMPLETED,
                JobState.FAILED,
                JobState.CANCELLED,
                JobState.TIMEOUT,
                JobState.NODE_FAIL,
            ):
                return status

            if timeout and (time.time() - start_time) > timeout:
                self._log(f"Timeout waiting for job {job_id}", "warning")
                return status

            time.sleep(poll_interval)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a SLURM job."""
        self._log(f"Cancelling job {job_id}")
        result = subprocess.run(
            ["scancel", job_id],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    def generate_job_script(self, output_path: Path, log_file: Path | None = None) -> Path:
        """Generate SLURM job submission script from config.

        Parameters
        ----------
        output_path : Path
            Path where the job script will be written.
        log_file : Path, optional
            Custom path for SLURM output log. If not provided, logs are
            written to <work_dir>/slurm-<job_id>.out.

        Returns
        -------
        Path
            Path to the generated job script.
        """
        slurm = self.config.slurm
        sim = self.config.simulation
        paths = self.config.paths

        work_dir = paths.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

        script_lines = [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={slurm.job_name}",
            f"#SBATCH --partition={slurm.partition}",
        ]

        # Model-specific compute directives (nodes, tasks, exclusive, etc.)
        script_lines.extend(self.config.model_config.generate_job_script_lines(self.config))

        if slurm.time_limit:
            script_lines.append(f"#SBATCH --time={slurm.time_limit}")
        if slurm.account:
            script_lines.append(f"#SBATCH --account={slurm.account}")
        if slurm.qos:
            script_lines.append(f"#SBATCH --qos={slurm.qos}")

        # Use custom log file or default SLURM pattern
        if log_file:
            script_lines.append(f"#SBATCH --output={log_file}")
            script_lines.append(f"#SBATCH --error={log_file}")
        else:
            script_lines.append(f"#SBATCH --output={work_dir}/slurm-%j.out")
            script_lines.append(f"#SBATCH --error={work_dir}/slurm-%j.err")
        script_lines.append("#SBATCH --open-mode=append")

        script_lines.extend(
            [
                "",
                "set -euox pipefail",
                "",
                f"export STARTPDY={sim.start_pdy}",
                f"export STARTCYC={sim.start_cyc}",
                f"export FCST_LENGTH_HRS={int(sim.duration_hours)}",
                f"export HOT_START_FILE='{paths.hot_start_file or ''}'",
                f'export USE_TPXO="{"YES" if self.config.boundary.source == "tpxo" else "NO"}"',
                f"export COASTAL_DOMAIN={sim.coastal_domain}",
                f"export METEO_SOURCE={sim.meteo_source.upper()}",
                f"export COASTAL_WORK_DIR={work_dir}",
                "",
            ]
        )

        if paths.raw_download_dir:
            script_lines.append(f"export RAW_DOWNLOAD_DIR={paths.raw_download_dir}")

        script_lines.extend(
            [
                "",
                "# Disable output buffering for real-time logging",
                "export PYTHONUNBUFFERED=1",
                "",
                "# Source the main workflow runner",
                f"cd {work_dir}",
                "stdbuf -oL -eL ./sing_run_generated.bash",
            ]
        )

        script_content = "\n".join(script_lines) + "\n"

        with output_path.open("w") as f:
            f.write(script_content)

        output_path.chmod(0o755)
        self._log(f"Generated job script: {output_path}")

        return output_path


def get_node_info() -> dict[str, str]:
    """Get information about the current compute node."""
    info = {
        "hostname": os.environ.get("HOSTNAME", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST", ""),
        "slurm_ntasks": os.environ.get("SLURM_NTASKS", ""),
    }
    return info
