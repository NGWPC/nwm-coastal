# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lavaca Bay SFINCS — Post-run Plotting Demo
#
# This notebook exercises the post-processing plotting API on an
# already-run Lavaca Bay SFINCS model at
# `docs/examples/lavaca-tx/run/sfincs_model`. It exists to:
#
# 1. Smoke-test :func:`coastal_calibration.sfincs.outputs.load_sfincs_water_level`
#    on a real quadtree SFINCS run.
# 2. Smoke-test the shared :func:`coastal_calibration.plotting.plot_water_level`
#    renderer on the ``ugrid-quadtree`` dispatch path.
# 3. Demonstrate plotting either water-surface elevation (`zs`) or water
#    depth (`h`), with **wet-cell masking** so dry land does not pollute the
#    colour scale.
# 4. Produce PNG snapshots and an MP4 animation that a reviewer can eyeball
#    to verify the field looks sensible.
#
# All figures are written into
# `docs/examples/lavaca-tx/run/sfincs_model/figs/`.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import Image, Video, display

notebook_dir = Path.cwd()  # assumes notebook is run from docs/examples/notebooks/
os.chdir(notebook_dir.parent / "lavaca-tx")

# %% [markdown]
# ## 1. Load the time-dependent water-level + depth field
#
# The reader returns one canonical dataset containing:
#
# - `zs(time, face)` — water-surface elevation (m, MSL).
# - `h(time, face)` — water depth, derived as `zs − zb`.
# - `zb(face)` — static bed elevation.
# - Mesh geometry (`node_x`, `node_y`, `face_nodes`) + `mesh_type` attr so the
#   shared renderer knows how to dispatch.
#
# Both `zs` and `h` are always present, so users pick what they want to plot
# at call time.

# %%
from coastal_calibration.sfincs.outputs import load_sfincs_water_level

run_dir = Path("run/sfincs_model")
assert run_dir.exists(), f"Run directory not found: {run_dir.resolve()}"

ds = load_sfincs_water_level(run_dir)
print(f"mesh_type     : {ds.attrs['mesh_type']}")
print(f"dims          : {dict(ds.sizes)}")
print(f"variables     : {sorted(ds.data_vars)}")
print(f"time[0]       : {ds.time.values[0]}")
print(f"time[-1]      : {ds.time.values[-1]}")
print(f"zs range (m)  : {float(ds['zs'].min()):+.3f} .. {float(ds['zs'].max()):+.3f}")
print(f"h  range (m)  : {float(ds['h'].min()):+.3f} .. {float(ds['h'].max()):+.3f}")

# %% [markdown]
# ## 2. Pick a colour range from the wet cells only
#
# The renderer's ``mask_dry=True`` default hides dry cells (where
# `h ≤ dry_threshold`), but the quantile we use for the colour scale should
# also be computed on the wet subset. Otherwise large bed-elevation values
# at dry land cells can stretch the scale even though those cells are no
# longer drawn.

# %%
DRY_THRESHOLD = 0.05  # m — same default as plot_water_level
wet = ds["h"] > DRY_THRESHOLD
zs_wet = ds["zs"].where(wet)
q_lo, q_hi = 0.02, 0.98
vmin, vmax = (float(v) for v in zs_wet.quantile([q_lo, q_hi]).values)
print(f"zs (wet) {int(q_lo * 100):d}th percentile: {vmin:+.3f} m")
print(f"zs (wet) {int(q_hi * 100):d}th percentile: {vmax:+.3f} m")
print(f"colour range (zs): [{vmin:+.3f}, {vmax:+.3f}] m")

# %% [markdown]
# ## 3. Plot three water-surface (`zs`) snapshots
#
# Wet-cell masking is on by default, so dry land becomes transparent.
# All three frames share the same `vmin`/`vmax` for cross-frame comparison.

# %%
from coastal_calibration.plotting import plot_water_level

figs_dir = run_dir / "figs"
figs_dir.mkdir(exist_ok=True)

n_time = ds.sizes["time"]
snapshot_indices = [0, n_time // 2, n_time - 1]
snapshot_labels = ["first", "middle", "last"]
snapshots: list[Path] = []

for label, idx in zip(snapshot_labels, snapshot_indices, strict=True):
    fig, ax = plt.subplots(figsize=(11, 8))
    plot_water_level(
        ds,
        time=idx,
        variable="zs",
        ax=ax,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        colorbar=True,
        # mask_dry=True is the default; spelled out here for clarity:
        mask_dry=True,
        dry_threshold=DRY_THRESHOLD,
    )
    out_png = figs_dir / f"water_level_snapshot_{label}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    snapshots.append(out_png)
    print(f"wrote {out_png.relative_to(run_dir)}  ({out_png.stat().st_size / 1024:.1f} KB)")

# %%
for png in snapshots:
    display(Image(filename=str(png), width=800))

# %% [markdown]
# ## 4. Plot water depth (`h`)
#
# Same renderer, just `variable="h"`. The wet-cell mask now hides cells with
# `h ≤ dry_threshold` (essentially zero depth), so the plot shows the actual
# inundation depth across the wet domain.

# %%
h_vmin, h_vmax = (float(v) for v in ds["h"].where(wet).quantile([q_lo, q_hi]).values)
print(f"colour range (h): [{h_vmin:.3f}, {h_vmax:.3f}] m")

fig, ax = plt.subplots(figsize=(11, 8))
plot_water_level(
    ds,
    time=snapshot_indices[1],
    variable="h",
    ax=ax,
    cmap="Blues",
    vmin=h_vmin,
    vmax=h_vmax,
    colorbar=True,
)
depth_png = figs_dir / "water_depth_snapshot.png"
fig.savefig(depth_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {depth_png.relative_to(run_dir)}  ({depth_png.stat().st_size / 1024:.1f} KB)")

display(Image(filename=str(depth_png), width=800))

# %% [markdown]
# ## 5. Water-level anomaly from the time-mean
#
# A diverging colormap is most useful when zero is a *meaningful* reference
# and values can sit on either side. For a single model run, the natural
# diverging story is the **anomaly from the per-cell time-mean**:
#
# `zs_anom(t, x) = zs(t, x) − mean(zs over time at x)`
#
# Subtracting each cell's static reference removes the bed-elevation
# contamination at inundated upland cells (their local mean is essentially
# their bed elevation, so the anomaly there is ≈ 0). What's left is the
# *dynamic* signal — tidal phase, storm surge, set-up — order ±1 m even
# though the raw `zs` field spans tens of metres.

# %%
zs_anom = ds["zs"] - ds["zs"].mean("time")
ds_anom = ds.assign(zs_anom=zs_anom)
ds_anom["zs_anom"].attrs.update({"long_name": "water-level anomaly from time-mean", "units": "m"})

amp = float(abs(zs_anom.where(wet)).quantile(0.98).values)
print(f"anomaly symmetric range: [-{amp:.3f}, +{amp:.3f}] m")

fig, ax = plt.subplots(figsize=(11, 8))
plot_water_level(
    ds_anom,
    time=snapshot_indices[1],
    variable="zs_anom",
    ax=ax,
    cmap="RdBu_r",
    vmin=-amp,
    vmax=+amp,
    colorbar=True,
    title=f"Lavaca Bay water-level anomaly @ {ds.time.values[snapshot_indices[1]]}",
)
anomaly_png = figs_dir / "water_level_anomaly_view.png"
fig.savefig(anomaly_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {anomaly_png.relative_to(run_dir)}  ({anomaly_png.stat().st_size / 1024:.1f} KB)")

display(Image(filename=str(anomaly_png), width=800))

# %% [markdown]
# ### What this view tells us
#
# At this snapshot (`2025-06-03T02:00`) a genuine spatial gradient
# emerges across the model:
#
# - The upper bay (northern quarter of wet cells) has a median anomaly
#   of about +0.05 m, with individual cells reaching +0.07 m.
# - The mid bay sits within ±0.02 m of its time-mean.
# - The open shelf (southern quarter) has a median anomaly of about
#   -0.02 m.
#
# That's a ≈ 0.07 m head-to-shelf difference in the water-level anomaly.
# The classic interpretation is a wind or surge set-up: water piled up
# toward the head of the bay relative to its own long-term mean, with
# the open shelf slightly below. Only a few percent of cells stray
# outside roughly ±0.05 m, so the gradient is the dominant signal at
# this timestamp rather than noise.
#
# The `zs` figure shows the same absolute values but the bay's mean-MSL
# field drowns out a 0.07 m anomaly. The anomaly view removes that
# baseline and lets the surge-like structure stand out.

# %% [markdown]
# ## 6. Animate the water-level evolution
#
# `animate_water_level` reuses the frame builder from `plot_water_level`,
# so the wet-cell mask is also applied to every frame.

# %%
from coastal_calibration.plotting import animate_water_level

anim_path = animate_water_level(
    ds,
    figs_dir / "water_level_animation.mp4",
    variable="zs",
    fps=10,
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    title_prefix="Lavaca Bay",
    mask_dry=True,
    dry_threshold=DRY_THRESHOLD,
)
print(
    f"wrote {anim_path.relative_to(run_dir.resolve())}  ({anim_path.stat().st_size / 1024:.1f} KB)"
)

Video(str(anim_path), embed=True, width=800)

# %% [markdown]
# ## 7. Water-level time series at user-specified points
#
# The plot stage accepts a CSV of observation points via
# ``SfincsModelConfig.obs_points_csv``; it interpolates the water-surface
# elevation at each point by nearest-face lookup on the quadtree mesh
# and writes ``obs_water_level.parquet`` next to the model output.
#
# For this demo we pick three points that trace the head-to-shelf
# set-up gradient revealed by the anomaly view:
#
# - `upper_bay_head`: inland tip of the bay where the anomaly is largest, ~ (−96.57, +28.64).
# - `mid_bay`: near the geographic centre of the wet domain, ~ (−96.47, +28.53).
# - `open_shelf`: south of the bay mouth, ~ (−96.40, +28.35).

# %%
import pandas as pd

from coastal_calibration.observations import (
    extract_water_level_series,
    load_obs_points,
    validate_points_in_domain,
)

obs_csv = run_dir / "user_obs_points.csv"
pd.DataFrame(
    {
        "id": ["upper_bay_head", "mid_bay", "open_shelf"],
        "lon": [-96.5743, -96.4669, -96.3992],
        "lat": [+28.6361, +28.5297, +28.3462],
    }
).to_csv(obs_csv, index=False)

points = load_obs_points(obs_csv)
validate_points_in_domain(points, ds)
series = extract_water_level_series(ds, points, variable="zs")
parquet_path = run_dir / "obs_water_level.parquet"
series.to_parquet(parquet_path)
print(
    f"wrote {parquet_path.relative_to(run_dir.parent)}  ({parquet_path.stat().st_size / 1024:.1f} KB)"
)
print(series.describe().loc[["min", "50%", "mean", "max"]].round(3))

# %% [markdown]
# Overlay the three time series so the head-to-shelf set-up gradient
# (from the anomaly view) is visible as a per-point offset:

# %%
fig, ax = plt.subplots(figsize=(11, 4.5))
for col in series.columns:
    ax.plot(series.index, series[col], label=col, linewidth=1.4)
ax.set_xlabel("time")
ax.set_ylabel("water-surface elevation (m, MSL)")
ax.set_title("Lavaca Bay: simulated water level at three obs points")
ax.legend(loc="best")
ax.grid(alpha=0.3)
ts_png = figs_dir / "obs_timeseries.png"
fig.savefig(ts_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {ts_png.relative_to(run_dir)}  ({ts_png.stat().st_size / 1024:.1f} KB)")

display(Image(filename=str(ts_png), width=900))

# %% [markdown]
# ### What the time series show
#
# The dominant feature is a clear phase delay as the tidal wave
# propagates up the bay. The tide arrives at `open_shelf` first,
# reaches `mid_bay` about three hours later, and peaks at
# `upper_bay_head` about six hours after the shelf. The 03 June peak
# makes this visible at a glance: shelf at 15:00, mid at 18:00, head
# at 21:00. The shallow bay acts as a long channel that retards the
# tidal front.
#
# Amplitudes are similar (~±0.25 m) but mean levels carry a small
# head-to-shelf set-up, with `upper_bay_head` averaging +0.086 m,
# `mid_bay` +0.061 m, and `open_shelf` +0.043 m. That 0.04 m
# difference is the same set-up gradient the anomaly snapshot
# surfaced earlier as a spatial pattern; the time series shows it
# persists across the whole run rather than being unique to one
# timestamp.

# %% [markdown]
# ## 8. Validation summary
#
# A reviewer should check:
#
# - The first snapshot shows the near-zero initial condition across the
#   wet bay; dry land outside the wet domain is transparent.
# - Middle and last snapshots reveal storm surge / tidal progression;
#   colours stay realistic (a few tenths of a metre across most of the
#   bay) rather than being washed out by dry-cell outliers.
# - The water-depth (`h`) snapshot shows the wet-domain bathymetry pattern
#   directly; deep channels stand out in dark blue.
# - The anomaly view shows the *dynamic* part of the water-level field:
#   the per-cell deviation from the time mean. Tidal phase, surge, and
#   set-up patterns dominate; static bed-elevation contamination is gone.
# - The animation moves smoothly with stable colours, dry cells stay
#   masked, and surge events appear and recede.
#
# All figures live under `run/sfincs_model/figs/`.
