#!/usr/bin/env bash
# Verification campaign v2 (2026-10-13): every backend on the same code, seeds (1234+i) and
# scenarios, so all comparisons are paired. One lane per backend; lanes run concurrently.
#
#   scripts/campaign_v2.sh heuristic,random     # baselines (fallback n/a)
#   scripts/campaign_v2.sh jev                  # model lanes: also the no-fallback suite
#   EPISODES=10 MAP_EPISODES=5 scripts/campaign_v2.sh clef   # reduced lane for slow models
#
# Keys come from the environment (TYPESAFE_API_KEY_FILE / LAYA_API_KEY / SEMIF_API_KEY /
# CLEF_API_KEY). Calls .venv/bin/s1doom directly: `uv run` would re-sync the venv and replace
# the Framework's ROCm torch.
set -u
B=$1
N=${EPISODES:-20}
NM=${MAP_EPISODES:-10}
CAP=${MAX_CALLS:-15000}
E=".venv/bin/s1doom eval --seed 1234 --backends $B --max-calls $CAP"
run() { echo "=== $(date -Is) $B $*"; $E "$@"; }

run --scenario defend_the_center --episodes "$N" --tag v2-m1
run --scenario defend_the_center --episodes "$N" --tag v2-m1noopp --no-opportunistic-fire
run --scenario deadly_corridor,health_gathering_supreme,my_way_home --episodes "$N" --tag v2-m2
run --scenario freedoom2_map01 --episodes "$NM" --tag v2-m3
case $B in heuristic*|random*) exit 0 ;; esac
run --scenario defend_the_center --episodes "$N" --tag v2-nofb-m1 --no-fallback
run --scenario deadly_corridor,health_gathering_supreme,my_way_home --episodes "$N" --tag v2-nofb-m2 --no-fallback
run --scenario freedoom2_map01 --episodes "$NM" --tag v2-nofb-m3 --no-fallback
echo "=== $(date -Is) $B done"
