# Shared by gen_configs_ana.ecf, schism_ana.ecf, sfincs_ana.ecf,
# run_stofs_download_ana.ecf -- sets SKIP_COASTAL (0/1). Requires CYCLE_DT
# to already be set (every caller sets this right after %include <head.h>).
# Recomputes its own PREV_CYCLE below rather than trusting a caller-provided
# value, so this stays correct regardless of what else a given .ecf has
# computed by this point. (gen_configs_ana.ecf also keeps its own separate
# PREV_CYCLE, used for other things in that file -- this is a harmless,
# intentional redundant reassignment, not a bug.)
#
# Bootstrap skip: the true first cycle in the suite's history has no
# troute_ana_a state from PREV_CYCLE either (troute itself is at its own
# bootstrap, no -lsf to load). Confirmed live that troute's T-3 sample is
# NOT an echo of loaded state when a hotstart IS available (real flow
# values evolve smoothly hour to hour, see troute_ana_a.ecf/troute_ana_b.ecf
# -lb fix notes) -- but a truly cold-started troute_ana_b (no prior state
# at all) still has genuine cold-start spin-up in its first simulated
# hour(s), a real routing-model concern distinct from that echo theory.
# Rather than cold-starting SCHISM/SFINCS against that degraded first hour,
# skip the whole coastal-model chain for this one cycle: run only
# ngen_forcing_ana/troute_ana_a/troute_ana_b (unaffected, already triggered
# independently of this task) so troute gets a full cycle of real history,
# then let coastal models cold-start cleanly next cycle once troute_ana_b
# itself has a real hotstart to load and produces a genuine T-3-to-T0
# window.
#
# IMPORTANT: this check must be troute_ana_a's own PREV_CYCLE state (NOT
# SCHISM_HOT_START_FILE/SFINCS_RST_FILE) -- those stay empty for
# TWO cycles (the skipped bootstrap cycle produces no coastal-model output
# at all), which would make every cycle skip forever. troute_ana_a runs
# unconditionally every cycle (including the bootstrap one), so by the
# very next cycle it already has real PREV_CYCLE state regardless of
# whether coastal models have ever run -- making this check self-limiting
# to exactly the one true bootstrap cycle, same LOAD_STATE_DIR path
# troute_ana_a.ecf itself already checks.

PREV_CYCLE=$(date -u -d "-1 hour ${CYCLE_DT}" +"%%Y%%m%%d%%H")
TROUTE_PREV_STATE_SAVE="%TROUTE_REGIONALIZATION_ROOT%/region_ana_a_${PREV_CYCLE}/%VPU%/state_save/troute"
if [ ! -f "${TROUTE_PREV_STATE_SAVE}" ]; then
  echo "No prior troute_ana_a state at ${TROUTE_PREV_STATE_SAVE} (bootstrap cycle) -- skipping ${CYCLE}"
  SKIP_COASTAL=1
else
  SKIP_COASTAL=0
fi
