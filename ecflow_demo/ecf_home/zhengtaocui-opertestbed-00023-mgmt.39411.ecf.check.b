#5.6.0
defs_state MIGRATE state>:aborted flag:message state_change:4483 modify_change:35 cal_count:227
edit ECF_MICRO '%' # server
edit ECF_HOME '/contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/ecf_home' # server
edit ECF_JOB_CMD '%ECF_JOB% 1> %ECF_JOBOUT% 2>&1' # server
edit ECF_KILL_CMD 'kill -15 %ECF_RID%' # server
edit ECF_STATUS_CMD 'ps --pid %ECF_RID% -f > %ECF_JOB%.stat 2>&1' # server
edit ECF_URL_CMD '${BROWSER:=firefox} -new-tab %ECF_URL_BASE%/%ECF_URL%' # server
edit ECF_URL_BASE 'https://confluence.ecmwf.int' # server
edit ECF_URL 'display/ECFLOW/ecflow+home' # server
edit ECF_LOG '/contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/ecf_home/zhengtaocui-opertestbed-00023-mgmt.39411.ecf.log' # server
edit ECF_INTERVAL '60' # server
edit ECF_LISTS 'zhengtaocui-opertestbed-00023-mgmt.39411.ecf.lists' # server
edit ECF_PASSWD 'zhengtaocui-opertestbed-00023-mgmt.39411.ecf.passwd' # server
edit ECF_CUSTOM_PASSWD 'zhengtaocui-opertestbed-00023-mgmt.39411.ecf.custom_passwd' # server
edit ECF_CHECK '/contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/ecf_home/zhengtaocui-opertestbed-00023-mgmt.39411.ecf.check' # server
edit ECF_CHECKOLD '/contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/ecf_home/zhengtaocui-opertestbed-00023-mgmt.39411.ecf.check.b' # server
edit ECF_CHECKINTERVAL '120' # server
edit ECF_CHECKMODE 'CHECK_ON_TIME' # server
edit ECF_TRIES '2' # server
edit ECF_VERSION '5.6.0' # server
edit ECF_PORT '39411' # server
edit ECF_HOST 'zhengtaocui-opertestbed-00023-mgmt' # server
edit ECF_CHECK_CMD 'ps --pid %ECF_RID% -f' # server
edit ECF_PID '2080910' # server
history /coastal_hourly/cycle/h14/run_sfincs MSG:[23:56:04 5.8.2026] --run force /coastal_hourly/cycle/h14/run_sfincs :lauren.schambach@zhengtaocui-opertestbed-00023-mgmt
history / MSG:[20:13:02 5.8.2026] --load=/contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/suite_def/coastal_hourly.def :lauren.schambach@zhengtaocui-opertestbed-00023-mgmtMSG:[20:13:02 5.8.2026] --begin=coastal_hourly :lauren.schambach@zhengtaocui-opertestbed-00023-mgmtMSG:[20:14:14 5.8.2026] --restart :lauren.schambach@zhengtaocui-opertestbed-00023-mgmt
history /coastal_hourly/cycle/h14/run_schism MSG:[23:56:04 5.8.2026] --run force /coastal_hourly/cycle/h14/run_schism :lauren.schambach@zhengtaocui-opertestbed-00023-mgmt
history /coastal_hourly MSG:[22:54:20 5.8.2026] --replace=/coastal_hourly /contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/suite_def/coastal_hourly.def parent :lauren.schambach@zhengtaocui-opertestbed-00023-mgmt
history /coastal_hourly/cycle/h14/ngen_forcing MSG:[22:56:36 5.8.2026] --run force /coastal_hourly/cycle/h14/ngen_forcing :lauren.schambach@zhengtaocui-opertestbed-00023-mgmt
history /coastal_hourly/cycle/h14/gen_configs MSG:[23:55:41 5.8.2026] --run force /coastal_hourly/cycle/h14/gen_configs :lauren.schambach@zhengtaocui-opertestbed-00023-mgmt
history /coastal_hourly/cycle/h14/ngen_troute MSG:[22:56:36 5.8.2026] --run force /coastal_hourly/cycle/h14/ngen_troute :lauren.schambach@zhengtaocui-opertestbed-00023-mgmtMSG:[23:51:51 5.8.2026] --kill /coastal_hourly/cycle/h14/ngen_troute :lauren.schambach@zhengtaocui-opertestbed-00023-mgmt
suite coastal_hourly # begun:1 state:aborted dur:00:56:40 flag:message suspended:1 rt:00:55:00
edit ECF_HOME '/contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/ecf_home'
edit ECF_FILES '/contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/ecf_home'
edit ECF_INCLUDE '/contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/ecf_home'
edit ECF_TRIES '1'
edit NWM_RTE_DIR '/contrib/LaurenSchambach/ngwpc/nwm-rte'
edit VPU '03S'
edit FORCING_OUT_DIR '/contrib/LaurenSchambach/ngwpc/run_ngen/data/scratch/short_range_coastal'
edit TROUTE_REGIONALIZATION_ROOT '/contrib/LaurenSchambach/ngwpc/run_ngen/regionalization'
edit NWM_COASTAL_CLI '/contrib/LaurenSchambach/ngwpc/nwm-coastal/nwm-coastal-cli'
edit NWM_COASTAL_PY '/contrib/LaurenSchambach/ngwpc/nwm-coastal/nwm-coastal-py'
edit GEN_SCRIPT '/contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/bin/gen_cycle_config.py'
edit SCHISM_BASE_YAML '/contrib/LaurenSchambach/ngwpc/run_coastal/schism_sims/run.yaml'
edit SFINCS_BASE_YAML '/contrib/LaurenSchambach/ngwpc/run_coastal/sfincs_sims/run.yaml'
edit SCHISM_CYCLES_DIR '/contrib/LaurenSchambach/ngwpc/run_coastal/schism_sims/cycles'
edit SFINCS_CYCLES_DIR '/contrib/LaurenSchambach/ngwpc/run_coastal/sfincs_sims/cycles'
calendar initTime:2026-Aug-05 22:54:20 suiteTime:2026-Aug-06 00:00:00 duration:01:05:40 initLocalTime:2026-Aug-05 22:54:20 lastTime:2026-Aug-06 00:00:00 calendarIncrement:00:01:00 dayChanged:1
family cycle # state:aborted dur:00:56:40 rt:00:55:00
repeat date YMD 20260805 20991231 1
family h00 # state:queued
edit HOUR '00'
time 00:00 # free
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h01 # state:queued
edit HOUR '01'
time 01:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h02 # state:queued
edit HOUR '02'
time 02:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h03 # state:queued
edit HOUR '03'
time 03:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h04 # state:queued
edit HOUR '04'
time 04:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h05 # state:queued
edit HOUR '05'
time 05:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h06 # state:queued
edit HOUR '06'
time 06:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h07 # state:queued
edit HOUR '07'
time 07:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h08 # state:queued
edit HOUR '08'
time 08:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h09 # state:queued
edit HOUR '09'
time 09:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h10 # state:queued
edit HOUR '10'
time 10:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h11 # state:queued
edit HOUR '11'
time 11:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h12 # state:queued
edit HOUR '12'
time 12:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h13 # state:queued
edit HOUR '13'
time 13:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h14 # state:aborted dur:00:56:40 rt:00:55:00
edit HOUR '14'
time 14:00
task ngen_forcing # try:1 state:complete dur:00:29:40 flag:message rt:00:28:00
task ngen_troute # passwd:H0OBpffl rid:2203495 abort<:ECF_JOB_CMD PID(2203480) path(/coastal_hourly/cycle/h14/ngen_troute) exited with status 137 [ /contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/ecf_home/coastal_hourly/cycle/h14/ngen_troute.job1 1> /contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/ecf_home/coastal_hourly/cycle/h14/ngen_troute.1 2>&1 ]>abort try:1 state:aborted dur:00:56:40 flag:task_aborted,ecfcmd_failed,killcmd_failed,killed,message rt:00:55:00
task gen_configs # try:1 state:complete dur:01:01:22 flag:message rt:00:00:42
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # passwd:3nbGZkpX rid:2247735 abort<:ECF_JOB_CMD PID(2247731) path(/coastal_hourly/cycle/h14/run_schism) exited with status 1 [ /contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/ecf_home/coastal_hourly/cycle/h14/run_schism.job1 1> /contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/ecf_home/coastal_hourly/cycle/h14/run_schism.1 2>&1 ]>abort try:1 state:aborted dur:01:01:45 flag:task_aborted,ecfcmd_failed,message rt:00:00:05
trigger gen_configs == complete
task run_sfincs # passwd:r48fWQVC rid:2247738 abort<:ECF_JOB_CMD PID(2247733) path(/coastal_hourly/cycle/h14/run_sfincs) exited with status 1 [ /contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/ecf_home/coastal_hourly/cycle/h14/run_sfincs.job1 1> /contrib/LaurenSchambach/ngwpc/nwm-coastal/ecflow_demo/ecf_home/coastal_hourly/cycle/h14/run_sfincs.1 2>&1 ]>abort try:1 state:aborted dur:01:01:55 flag:task_aborted,ecfcmd_failed,message rt:00:00:15
trigger gen_configs == complete
endfamily
family h15 # state:queued
edit HOUR '15'
time 15:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h16 # state:queued
edit HOUR '16'
time 16:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h17 # state:queued
edit HOUR '17'
time 17:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h18 # state:queued
edit HOUR '18'
time 18:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h19 # state:queued
edit HOUR '19'
time 19:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h20 # state:queued
edit HOUR '20'
time 20:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h21 # state:queued
edit HOUR '21'
time 21:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h22 # state:queued
edit HOUR '22'
time 22:00
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
family h23 # state:queued
edit HOUR '23'
time 23:00 # free
task ngen_forcing # state:queued
task ngen_troute # state:queued
task gen_configs # state:queued
trigger ngen_forcing == complete and ngen_troute == complete
task run_schism # state:queued
trigger gen_configs == complete
task run_sfincs # state:queued
trigger gen_configs == complete
endfamily
endfamily
endsuite
# enddef
