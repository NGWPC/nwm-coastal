"""Vendored SCHISM test case generator and runner from schism-subsetter.

Generates minimal rectangular SCHISM grids with tidal forcing and runs
them through the full partition + pschism workflow.  Only needs numpy
and the standard library.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = ["SCHISMRunError", "generate_test_case", "run_schism"]


# ---------------------------------------------------------------------------
# Test-case generator (from schism_subsetter.test_case_generator)
# ---------------------------------------------------------------------------


def _validate_inputs(
    grid_size: tuple[int, int],
    resolution: tuple[float, float],
    boundary_type: Literal["shore", "ocean", "island"],
    depth: float,
    run_days: float,
    dt: float,
    tidal_amplitude: float,
    tidal_period_hours: float,
    manning_coefficient: float,
) -> None:
    nx, ny = grid_size
    dx, dy = resolution
    if not isinstance(nx, int) or not isinstance(ny, int):
        raise TypeError(f"Grid size must be integers, got nx={type(nx)}, ny={type(ny)}")
    if nx < 7 or ny < 7:
        raise ValueError(f"Grid size must be at least 7x7, got {nx}x{ny}")
    if nx % 2 == 0 or ny % 2 == 0:
        raise ValueError(f"Grid size must be odd in both dimensions, got {nx}x{ny}")
    if dx <= 0 or dy <= 0:
        raise ValueError(f"Resolution must be positive, got dx={dx}, dy={dy}")
    if boundary_type not in {"shore", "ocean", "island"}:
        raise ValueError(f"Boundary type must be 'shore', 'ocean', or 'island', got '{boundary_type}'")
    if depth <= 0:
        raise ValueError(f"Depth must be positive, got {depth}")
    if run_days <= 0:
        raise ValueError(f"Run days must be positive, got {run_days}")
    if dt <= 0:
        raise ValueError(f"Time step must be positive, got {dt}")
    if tidal_amplitude < 0:
        raise ValueError(f"Tidal amplitude must be non-negative, got {tidal_amplitude}")
    if tidal_period_hours <= 0:
        raise ValueError(f"Tidal period must be positive, got {tidal_period_hours}")
    if manning_coefficient <= 0:
        raise ValueError(f"Drag coefficient must be positive, got {manning_coefficient}")


def _generate_coast(nx: int, ny: int) -> tuple[list[list[int]], list[list[int]]]:
    split_row = ny // 2
    open_boundary_nodes: list[int] = []
    open_boundary_nodes.extend([j * nx + nx for j in range(split_row, ny)])
    open_boundary_nodes.extend([(ny - 1) * nx + i for i in range(nx - 1, 1, -1)])
    open_boundary_nodes.extend([j * nx + 1 for j in range(ny - 1, split_row - 1, -1)])
    land_boundary_nodes = [split_row * nx + 1]
    land_boundary_nodes.extend([j * nx + 1 for j in range(split_row - 1, -1, -1)])
    land_boundary_nodes.extend(range(2, nx + 1))
    land_boundary_nodes.extend([j * nx + nx for j in range(1, split_row + 1)])
    return [open_boundary_nodes], [land_boundary_nodes]


def _generate_small_island(nx: int, ny: int) -> tuple[list[list[int]], list[list[int]]]:
    open_boundary_nodes: list[int] = []
    open_boundary_nodes.extend([j * nx + 1 for j in range(ny // 2 - 1, 0, -1)])
    open_boundary_nodes.extend(range(1, nx + 1))
    open_boundary_nodes.extend([j * nx + nx for j in range(1, ny)])
    open_boundary_nodes.extend([(ny - 1) * nx + i for i in range(nx - 1, 0, -1)])
    open_boundary_nodes.extend([j * nx + 1 for j in range(ny - 2, ny // 2, -1)])
    open_boundary_nodes_list = [
        open_boundary_nodes,
        [j * nx + 1 for j in range(ny // 2 + 1, ny // 2 - 2, -1)],
    ]
    i_center = nx // 2 + 2
    j_center = ny // 2 + 1
    land_boundary_nodes = [
        [
            (j_center - 1) * nx + i_center,
            (j_center - 1) * nx + i_center + 1,
            j_center * nx + i_center + 1,
            j_center * nx + i_center,
        ]
    ]
    return open_boundary_nodes_list, land_boundary_nodes


def _generate_central_island(nx: int, ny: int) -> tuple[list[list[int]], list[list[int]]]:
    open_boundary_nodes: list[int] = []
    open_boundary_nodes.extend([j * nx + 1 for j in range(ny // 2 - 1, 0, -1)])
    open_boundary_nodes.extend(range(1, nx + 1))
    open_boundary_nodes.extend([j * nx + nx for j in range(1, ny)])
    open_boundary_nodes.extend([(ny - 1) * nx + i for i in range(nx - 1, 0, -1)])
    open_boundary_nodes.extend([j * nx + 1 for j in range(ny - 2, ny // 2, -1)])
    open_boundary_nodes_list = [
        open_boundary_nodes,
        [j * nx + 1 for j in range(ny // 2 + 1, ny // 2 - 2, -1)],
    ]
    i_center = nx // 2
    j_center = ny // 2
    i_start = i_center - 1
    j_start = j_center - 1
    land_boundary_nodes: list[int] = []
    for i in range(3):
        land_boundary_nodes.append(j_start * nx + (i_start + i) + 1)
    for j in range(1, 3):
        land_boundary_nodes.append((j_start + j) * nx + (i_start + 2) + 1)
    for i in range(1, -1, -1):
        land_boundary_nodes.append((j_start + 2) * nx + (i_start + i) + 1)
    land_boundary_nodes.append((j_start + 1) * nx + i_start + 1)
    return open_boundary_nodes_list, [land_boundary_nodes]


def _get_station_node(nx: int, ny: int) -> int:
    i_center = nx // 2 + 1
    j_center = ny // 2 + 2
    return j_center * nx + i_center + 1


def _write_hgrid(
    base_dir: Path,
    nx: int,
    ny: int,
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    boundary_type: Literal["shore", "ocean", "island"],
    open_boundary_nodes: list[list[int]],
    land_boundary_nodes: list[list[int]] | None,
) -> None:
    hgrid_lines = [f"{nx}x{ny} grid with rectangular domain", f"{len(elements)} {len(nodes)}"]
    hgrid_lines.extend(
        [f"{int(node[0])} {node[1]:.6f} {node[2]:.6f} {node[3]:.6f}" for node in nodes]
    )
    hgrid_lines.extend(
        [
            f"{int(elem[0])} {int(elem[1])} {int(elem[2])} {int(elem[3])} {int(elem[4])}"
            for elem in elements
        ]
    )
    total_open = sum(len(boundary) for boundary in open_boundary_nodes)
    hgrid_lines.extend(
        [
            f"{len(open_boundary_nodes)} = Number of open boundaries",
            f"{total_open} = Total number of open boundary nodes",
        ]
    )
    for idx, bnd in enumerate(open_boundary_nodes, start=1):
        hgrid_lines.append(f"{len(bnd)} = Number of nodes for open boundary {idx}")
        hgrid_lines.extend([str(node) for node in bnd])
    if land_boundary_nodes is not None:
        total_land = sum(len(boundary) for boundary in land_boundary_nodes)
        hgrid_lines.extend(
            [
                f"{len(land_boundary_nodes)} = Number of land boundaries (including islands)",
                f"{total_land} = Total number of land boundary nodes",
            ]
        )
        for idx, bnd in enumerate(land_boundary_nodes, start=1):
            btype, bname = (1, "island") if boundary_type != "shore" else (0, "exterior")
            hgrid_lines.append(f"{len(bnd)} {btype} = Number of nodes for {bname} boundary {idx}")
            hgrid_lines.extend([str(node) for node in bnd])
    (base_dir / "hgrid.gr3").write_text("\n".join(hgrid_lines) + "\n")


def _write_manning(
    base_dir: Path,
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    manning_coefficient: float,
) -> None:
    manning_lines = [
        f"Constant Manning coefficient = {manning_coefficient}",
        f"{len(elements)} {len(nodes)}",
    ]
    manning_lines.extend(
        [f"{int(node[0])} {node[1]:.6f} {node[2]:.6f} {manning_coefficient:.6f}" for node in nodes]
    )
    manning_lines.extend(
        [
            f"{int(elem[0])} {int(elem[1])} {int(elem[2])} {int(elem[3])} {int(elem[4])}"
            for elem in elements
        ]
    )
    (base_dir / "manning.gr3").write_text("\n".join(manning_lines) + "\n")


def _write_vgrid(base_dir: Path) -> None:
    vgrid_text = """2 !ivcor
2 1 1.e6
Z levels
1 -1.e6
S levels
40. 1. 1.e-4
1 -1.
2 0.
"""
    (base_dir / "vgrid.in").write_text(vgrid_text)


def _write_bctides(base_dir: Path, open_boundary_nodes: list[list[int]]) -> None:
    bctides_lines = [
        "!Rectangular test case",
        "0 0.0 ntip",
        "0 nbfr",
        f"{len(open_boundary_nodes)} nope",
    ]
    for bnd in open_boundary_nodes:
        bctides_lines.extend([f"{len(bnd)} 1 0 0 0"])
    (base_dir / "bctides.in").write_text("\n".join(bctides_lines) + "\n")


def _write_elev_th(
    base_dir: Path,
    run_days: float,
    dt: float,
    tidal_amplitude: float,
    tidal_period_hours: float,
    num_open_boundaries: int,
) -> None:
    tidal_period_sec = tidal_period_hours * 3600.0
    omega = 2.0 * np.pi / tidal_period_sec
    n_steps = int(np.ceil(run_days * 86400 / dt)) + 10
    elev_lines = []
    for step in range(n_steps):
        time_seconds = step * dt
        elevation = tidal_amplitude * np.sin(omega * time_seconds)
        elev_values = " ".join([f"{elevation:.6f}"] * num_open_boundaries)
        elev_lines.append(f"{time_seconds:.2f} {elev_values}")
    (base_dir / "elev.th").write_text("\n".join(elev_lines) + "\n")


def _write_station_in(base_dir: Path, station_x: float, station_y: float) -> None:
    station_lines = [
        "1 0 0 0 0 0 0 0 0",
        "1",
        f"1 {station_x} {station_y} 0.0",
    ]
    (base_dir / "station.in").write_text("\n".join(station_lines) + "\n")


def _write_param_nml(base_dir: Path, run_days: float, dt: float, station_output: bool) -> None:
    nspool = max(1, int(3600 / dt))
    ihfskip = nspool
    nspool_sta = max(1, int(600 / dt)) if station_output else 1
    nhot_write = ihfskip
    iout_sta = 1 if station_output else 0

    param_nml = f"""&CORE
  ipre = 0
  ibc = 1
  ibtp = 0
  rnday = {run_days}
  dt = {dt}
  nspool = {nspool}
  ihfskip = {ihfskip}
  msc2 = 24
  mdc2 = 30
  ntracer_gen = 0
  ntracer_age = 0
  sed_class = 0
  eco_class = 0
  nbins_veg_vert = 2
  nmarsh_types = 2
/

&OPT
  start_year = 2000
  start_month = 1
  start_day = 1
  start_hour = 0.
  utc_start = 0.

  ics = 1
  ihot = 0

  ieos_type = 0
  ieos_pres = 0

  nchi = -1
  hmin_man = 1.

  ncor = 0
  coricoef = 0.

  nramp_elev = 1

  itur = 0
  dfv0 = 1.e-6
  dfh0 = 1.e-6

  inu_elev = 0
  inu_uv = 0

  ihhat = 1
  inunfl = 0
  h0 = 1.e-5

  moitn0 = 50
  mxitn0 = 1500
  rtol0 = 1.e-12

  inter_mom = 0
  kr_co = 1

  indvel = 0
  ihorcon = 0
  hvis_coef0 = 0.025
  ishapiro = 1
  niter_shap = 1
  shapiro0 = 0.5

  thetai = 1

  ihdif = 0

  nadv = 1
  dtb_min = 10.
  dtb_max = 30.

  itr_met = 3
  h_tvd = 5.

  h_bcc1 = 100.

  hw_depth = 1.e6
  hw_ratio = 0.5

  s1_mxnbt = 1.0
  s2_mxnbt = 4.0

  slam0 = 0.
  sfea0 = 0.

  iunder_deep = 0

  dramp = 1.
  drampbc = 1.

  iflux = 0
  rmaxvel = 10.
/

&SCHOUT
  iout_sta = {iout_sta}
  nspool_sta = {nspool_sta}

  nhot = 0
  nhot_write = {nhot_write}

  ! Minimal output: elevation only
  iof_hydro(1) = 1 !0: off; 1: on - elev. [m]
  iof_hydro(26) = 0 !horizontal vel vector defined @side [m/s]
/
"""
    (base_dir / "param.nml").write_text(param_nml)


def _renumber_after_island_removal(
    to_remove_id: int,
    nx: int,
    ny: int,
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    open_boundaries: list[list[int]],
    land_boundaries: list[list[int]],
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[list[int]], list[list[int]]]:
    old_to_new = {n if n < to_remove_id else n + 1: n for n in range(1, nx * ny)}
    nodes = np.delete(nodes, to_remove_id - 1, axis=0)
    for i, (node_id, *_) in enumerate(nodes):
        nodes[i, 0] = old_to_new[node_id]
    for i, elem in enumerate(elements):
        for j in range(2, 5):
            elements[i, j] = old_to_new[int(elem[j])]
    new_open_boundaries = []
    for bnd in open_boundaries:
        new_bnd = [old_to_new[node_id] for node_id in bnd if node_id != to_remove_id]
        new_open_boundaries.append(new_bnd)
    new_land_boundaries = []
    for bnd in land_boundaries:
        new_bnd = [old_to_new[node_id] for node_id in bnd if node_id != to_remove_id]
        new_land_boundaries.append(new_bnd)
    return nodes, elements, new_open_boundaries, new_land_boundaries


def generate_test_case(
    grid_size: tuple[int, int],
    resolution: tuple[float, float],
    boundary_type: Literal["shore", "ocean", "island"],
    base_dir: str | Path,
    depth: float = 1.0,
    run_days: float = 1.0,
    dt: float = 100.0,
    tidal_amplitude: float = 1.0,
    tidal_period_hours: float = 12.42,
    manning_coefficient: float = 0.025,
    station_output: bool = True,
) -> None:
    """Create a minimal SCHISM test case with rectangular grid and tidal forcing."""
    _validate_inputs(
        grid_size, resolution, boundary_type, depth, run_days, dt,
        tidal_amplitude, tidal_period_hours, manning_coefficient,
    )
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    nx, ny = grid_size
    dx, dy = resolution

    nodes = []
    node_id = 1
    for j in range(ny):
        for i in range(nx):
            nodes.append([node_id, i * dx, j * dy, depth])
            node_id += 1
    nodes_arr = np.array(nodes)

    if boundary_type == "island":
        i_center = nx // 2 - 1
        j_center = ny // 2 - 1
        exclude_elements = set()
        for je in range(j_center, j_center + 2):
            for ie in range(i_center, i_center + 2):
                exclude_elements.add((ie, je))
    elif boundary_type == "ocean":
        i_center = nx // 2 + 2
        j_center = ny // 2 + 1
        exclude_elements = {(i_center - 1, j_center - 1)}
    else:
        exclude_elements = set()

    elements = []
    elem_id = 1
    for j in range(ny - 1):
        for i in range(nx - 1):
            if (i, j) in exclude_elements:
                continue
            n1 = j * nx + i + 1
            n2 = j * nx + i + 2
            n3 = (j + 1) * nx + i + 1
            n4 = (j + 1) * nx + i + 2
            elements.append([elem_id, 3, n1, n2, n4])
            elem_id += 1
            elements.append([elem_id, 3, n1, n4, n3])
            elem_id += 1
    elements_arr = np.array(elements)

    if boundary_type == "shore":
        open_boundary_nodes, land_boundary_nodes = _generate_coast(nx, ny)
    elif boundary_type == "ocean":
        open_boundary_nodes, land_boundary_nodes = _generate_small_island(nx, ny)
    else:
        open_boundary_nodes, land_boundary_nodes = _generate_central_island(nx, ny)
        to_remove_id = ny // 2 * nx + nx // 2 + 1
        nodes_arr, elements_arr, open_boundary_nodes, land_boundary_nodes = (
            _renumber_after_island_removal(
                to_remove_id, nx, ny, nodes_arr, elements_arr,
                open_boundary_nodes, land_boundary_nodes,
            )
        )

    _write_hgrid(base_dir, nx, ny, nodes_arr, elements_arr, boundary_type, open_boundary_nodes, land_boundary_nodes)
    _write_manning(base_dir, nodes_arr, elements_arr, manning_coefficient)
    _write_vgrid(base_dir)
    _write_bctides(base_dir, open_boundary_nodes)
    _write_elev_th(base_dir, run_days, dt, tidal_amplitude, tidal_period_hours, len(open_boundary_nodes))

    if station_output:
        station_node = _get_station_node(nx, ny)
        station_x = nodes_arr[station_node - 1, 1]
        station_y = nodes_arr[station_node - 1, 2]
        _write_station_in(base_dir, station_x, station_y)

    _write_param_nml(base_dir, run_days, dt, station_output)
    logger.info("SCHISM test case created in %s", base_dir)


# ---------------------------------------------------------------------------
# SCHISM runner (from schism_subsetter.run_schism)
# ---------------------------------------------------------------------------


class SCHISMRunError(Exception):
    """Exception raised when SCHISM execution fails."""


def _check_schism_output(result: subprocess.CompletedProcess, project_dir: Path) -> None:
    fatal_error_file = project_dir / "fatal.error"
    if fatal_error_file.exists():
        error_content = fatal_error_file.read_text()
        raise SCHISMRunError(f"SCHISM failed with fatal error:\n{error_content}")
    if result.stdout and "ABORT:" in result.stdout:
        abort_lines = [line for line in result.stdout.splitlines() if "ABORT:" in line]
        if abort_lines:
            unique_aborts = sorted(set(abort_lines))
            raise SCHISMRunError("SCHISM aborted with errors:\n" + "\n".join(unique_aborts))
    if result.stderr and "MPI_ABORT was invoked" in result.stderr:
        raise SCHISMRunError(f"SCHISM invoked MPI_ABORT:\n{result.stderr}")


def _run_command(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    error_msg: str,
    timeout: int | None = None,
    check_output: bool = False,
) -> subprocess.CompletedProcess:
    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=False, env=env,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.stdout:
            logger.debug("STDOUT:\n%s", result.stdout)
        if result.stderr:
            logger.debug("STDERR:\n%s", result.stderr)
        if check_output:
            _check_schism_output(result, cwd)
        if result.returncode != 0:
            logger.error("%s (rc=%d)\nSTDOUT:\n%s\nSTDERR:\n%s",
                         error_msg, result.returncode, result.stdout, result.stderr)
            raise SCHISMRunError(f"{error_msg} (return code: {result.returncode})")
        return result
    except subprocess.TimeoutExpired as e:
        raise SCHISMRunError(f"{error_msg} (timed out after {timeout}s)") from e
    except SCHISMRunError:
        raise
    except Exception as e:
        raise SCHISMRunError(f"{error_msg}: {e}") from e


def run_schism(
    project_dir: str | Path,
    exec_dir: str | Path,
    num_procs: int = 4,
    num_scribes: int = 2,
    lib_path: str | Path | None = None,
    executable_name: str = "pschism",
    clean_outputs: bool = True,
    mpi_command: str = "mpiexec",
    mpi_args: list[str] | None = None,
    ufactor: float = 1.01,
    seed: int = 15,
    timeout: int | None = None,
) -> None:
    """Run SCHISM simulation with automatic mesh partitioning.

    Performs: metis_prep -> gpmetis -> mpiexec pschism -> validate outputs.
    """
    project_dir = Path(project_dir).resolve()
    exec_dir = Path(exec_dir).resolve()

    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")
    if not exec_dir.exists():
        raise FileNotFoundError(f"Executable directory not found: {exec_dir}")

    metis_prep = exec_dir / "metis_prep"
    gpmetis = exec_dir / "gpmetis"
    pschism = exec_dir / executable_name

    missing = [n for n, p in [("metis_prep", metis_prep), ("gpmetis", gpmetis), (executable_name, pschism)] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing executables in {exec_dir}: {', '.join(missing)}")

    required_files = ["hgrid.gr3", "vgrid.in", "param.nml", "bctides.in"]
    missing_files = [f for f in required_files if not (project_dir / f).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing input files in {project_dir}: {', '.join(missing_files)}")

    if num_procs < 3:
        raise ValueError(f"num_procs must be >= 3 (got {num_procs})")
    if num_scribes < 2:
        raise ValueError(f"num_scribes must be >= 2 (got {num_scribes})")

    num_partitions = num_procs - num_scribes

    env = os.environ.copy()
    if lib_path is not None:
        lib_path_resolved = Path(lib_path).resolve()
        if not lib_path_resolved.exists():
            warnings.warn(f"Library path does not exist: {lib_path_resolved}", stacklevel=2)
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_path_resolved}:{current}" if current else str(lib_path_resolved)

    # Step 1: metis_prep
    logger.info("Running metis_prep ...")
    _run_command(
        cmd=[str(metis_prep.absolute()), "hgrid.gr3", "vgrid.in"],
        cwd=project_dir, env=env, error_msg="metis_prep failed",
    )
    if not (project_dir / "graphinfo").exists():
        raise SCHISMRunError("metis_prep failed to generate graphinfo")

    # Step 2: gpmetis
    logger.info("Running gpmetis (%d partitions) ...", num_partitions)
    _run_command(
        cmd=[str(gpmetis.absolute()), "graphinfo", str(num_partitions),
             f"-ufactor={ufactor}", f"-seed={seed}"],
        cwd=project_dir, env=env, error_msg="gpmetis failed",
    )
    partition_input = project_dir / f"graphinfo.part.{num_partitions}"
    if not partition_input.exists():
        raise SCHISMRunError(f"gpmetis failed to generate {partition_input.name}")

    # Create partition.prop
    lines = partition_input.read_text().splitlines(keepends=True)
    with open(project_dir / "partition.prop", "w") as f:
        f.writelines(f"{i} {line}" for i, line in enumerate(lines, start=1))

    # Step 3: Run SCHISM
    outputs_dir = project_dir / "outputs"
    if clean_outputs and outputs_dir.exists():
        shutil.rmtree(outputs_dir)
    outputs_dir.mkdir(exist_ok=True)

    cmd = [mpi_command]
    if mpi_args:
        cmd.extend(mpi_args)
    cmd.extend(["-n", str(num_procs), str(pschism.absolute()), str(num_scribes)])

    logger.info("Running SCHISM: %s", " ".join(cmd))
    _run_command(
        cmd=cmd, cwd=project_dir, env=env,
        error_msg="SCHISM simulation failed",
        timeout=timeout, check_output=True,
    )

    # Step 4: Validate
    if (project_dir / "station.in").exists():
        staout = outputs_dir / "staout_1"
        if not staout.exists():
            raise SCHISMRunError("Expected output file not found: staout_1")
        if staout.stat().st_size == 0:
            raise SCHISMRunError("Output file staout_1 is empty")
        data_lines = [l for l in staout.read_text().splitlines() if l and not l.strip().startswith("!")]
        if not data_lines:
            raise SCHISMRunError("Output file staout_1 contains no data")
    logger.info("SCHISM workflow complete")
