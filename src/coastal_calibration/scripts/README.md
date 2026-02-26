# SCHISM Coastal Calibration Scripts

## Overview

This directory contains shell and Slurm scripts to run SCHISM calibration workflows.
Inputs include NWM retrospective streamflow and forcing, STOFS forecasts, or TPXO
water-level forecasts.

## Prerequisites

1. `OTPSnc` and the `TPXO10_atlas` data. Download `OTPSnc` from
    <https://www.tpxo.net/otps>. The `TPXO10_atlas` archive is available on S3 at
    `s3://ngwpc-data/Coastal_and_atmospheric_forcing_for_calibration/TPXO_atlas/TPXO10_atlas_v2_nc.zip`.

    > **Note:** After unpacking `OTPSnc.tar.Z`, update the `Makefile` in the `OTPS`
    > folder for your environment. An example Linux snippet:

    ```makefile
    ARCH = $(shell uname -s)
    ifeq ($(ARCH),Linux)
        FC = /contrib/software/gcc/8.5.0/bin/gfortran
        NCLIB = /contrib/software/netcdf/4.7.4/lib
        NCINCLUDE = /contrib/software/netcdf/4.7.4/include
        NCLIBS = -lnetcdf -lnetcdff
    endif

    predict_tide: predict_tide.f90 subs.f90 constit.h
    $(FC) -o predict_tide predict_tide.f90 subs.f90 -L$(NCLIB) $(NCLIBS) -I$(NCINCLUDE)

    extract_HC: extract_HC.f90 subs.f90
    $(FC) -o extract_HC extract_HC.f90 subs.f90 -L$(NCLIB) $(NCLIBS) -I$(NCINCLUDE)
    ```

    Run `make` to build the binaries (for example `predict_tide`). Unpack the
    `TPXO10_atlas` files into the `OTPSnc` folder.

    > **Local installation:** On Parallel Works the `OTPSnc` + `TPXO10_atlas`
    > installation is available at `/contrib/software/OTPSnc`.

1. NWM v3 installation. The NWM v3 package is available in the Parallel Works S3 bucket
    `s3://6668bc97ecf9731800ee83e8`. See
    `s3://6668bc97ecf9731800ee83e8/nwmv3_oe_install_rocky8/README.TXT` for install
    instructions.

    > **Warning:** The SCHISM binary in that S3 bucket is compiled for AWS `hpc7a`. If
    > your cluster uses a different node type, rebuild SCHISM for your target nodes.

    Example build script for NextGen OE clusters (adapt paths and compilers to your
    environment):

    ```bash
    #!/usr/bin/env bash
    mkdir -p ./build
    cd ./build || exit 1
    rm -rf *
    cmake -C ../SCHISM.local.build -C ../SCHISM.local.aws_intel19 ../src/
    make -j8 pschism
    cp ./build/bin/pschism_wcoss2_NO_PARMETIS_TVD-VL ../../exec/pschism_wcoss2_NO_PARMETIS_TVD-VL_intel
    ```

    > **Note:** Adjust `PATH`, `LD_LIBRARY_PATH`, and compiler environment variables as
    > required by your site.

1. Historical STOFS archives from
    <https://noaa-gestofs-pds.s3.amazonaws.com/index.html>.

1. Retrospective NWM streamflow and forcing from
    <https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/index.html>.

1. Hot restart file (only if a hot restart is required for your calibration run).

## Scripts

### Without Singularity

- `run_coastal_workflow.bash` — main calibration driver; edit environment variables to
    match your case.
- `regrid_stofs.bash` — regrid STOFS forecast into SCHISM `elev2D.th.nc` for boundary
    conditions.
- `make_tpxo_ocean.bash` — create SCHISM boundary file from the TPXO10 atlas.
- `nwm_forcing_coastal.bash` — create atmospheric inputs for SCHISM from NWM forcing.
- `initial_discharge.bash` — create the discharge file from NWM forcing.
- `combine_sink_source.bash` — combine sink and source files.
- `merge_source_sink.bash` — merge sink and source files.
- `update_param.bash` — create `parm.nml` according to user settings.
- `nwm_coastal.bash` — run SCHISM with prepared inputs.

To use these scripts, follow the instructions below:

1. Edit environment variables in `run_coastal_workflow.bash` (lines ~1–66 in the file).

    - Lines 1–14: Slurm job settings.
    - Lines 15–66: environment-variable settings for the calibration run.

1. Submit from the `coastal/calib` directory:

```bash
cd ngen-forcing/coastal/calib
sbatch run_coastal_workflow.bash
```

Use `squeue` to check status and `scancel <jobid>` to cancel.

### With Singularity

- `sing_run.bash` — main driver when using a Singularity container (update SLURM and env
    vars).
- `run_sing_coastal_workflow_pre_forcing_coastal.bash` — prepares container execution
    for `workflow_driver.py` (requires `mpi4py`).
- `pre_nwm_forcing_coastal.bash` — invoked by the previous script to prepare data for
    `workflow_driver.py`.
- `run_sing_coastal_workflow_post_forcing_coastal.bash` — container
    post-processing/cleanup for `workflow_driver.py`.
- `post_nwm_forcing_coastal.bash` — invoked after `workflow_driver.py` to perform post
    steps.
- `run_sing_coastal_workflow_update_params.bash` — creates parameter files inside the
    container.
- `run_sing_coastal_workflow_make_tpxo_ocean.bash` — prepares TPXO-based water-level
    boundary files inside the container.
- `run_sing_coastal_workflow_pre_make_stofs_ocean.bash` — prepares execution of
    `regrid_estofs.py` inside the container.
- `pre_regrid_stofs.bash` / `post_regrid_stofs.bash` — helper scripts invoked
    before/after `regrid_estofs.py` (MPI required).
- `run_sing_coastal_workflow_pre_schism.bash` /
    `run_sing_coastal_workflow_post_schism.bash` — prepare and post-process SCHISM runs
    inside the container.

To use these scripts, follow the instructions below:

1. Edit SLURM settings and environment variables in `sing_run.bash` (lines ~1–50). -
    Lines 3–7: Slurm job settings. - Lines 8–50: environment-variable settings for the
    calibration run.

> **Note:** Ensure `NODES` and `NCORES` match the SLURM configuration in
> `sing_run.bash`.

```bash
sbatch ngen-forcing/coastal/calib/sing_run.bash
```

## Download helper scripts

- `download_nwm_ana_archived_chout.bash` — download archived NWM AnA-cycle streamflow
    data for a given time period and domain. Usage:

```bash
./download_nwm_ana_archived_chout.bash 20240401 20240405 hawaii
```

Files are saved in the current directory.

- `download_nwm_ana_archived_forcing.bash` — download archived NWM AnA-cycle forcing
    data (start, end, domain):

```bash
./download_nwm_ana_archived_forcing.bash 20240401 20240405 hawaii
```

- `download_stofs.bash` — download archived STOFS data for a given time period (start,
    end):

```bash
./download_stofs.bash 20240401 20240405
```

Downloaded STOFS files are saved in subdirectories named `stofs_YYYYMMDD` under the
current directory.
