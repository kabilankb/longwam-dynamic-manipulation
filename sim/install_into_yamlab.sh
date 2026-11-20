#!/bin/bash
# Install the ConveyorCubePick-v0 task, its data generator and the lighting HDRIs into a YAMLab
# (Isaac Sim 6 / Isaac Lab 3 port) checkout. Idempotent: safe to run again after updates.
#   bash sim/install_into_yamlab.sh /path/to/yamlab6
set -euo pipefail
YAMLAB=${1:?usage: install_into_yamlab.sh /path/to/yamlab6}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$YAMLAB/yamlab/envs/tasks/__init__.py" ] || { echo "$YAMLAB is not a YAMLab checkout" >&2; exit 1; }

cp "$HERE/yamlab/configs/tasks/conveyor_cube_pick.yaml" "$YAMLAB/yamlab/configs/tasks/"
cp "$HERE"/yamlab/envs/tasks/conveyor_cube_pick_manager{,_cfg}.py "$YAMLAB/yamlab/envs/tasks/"
mkdir -p "$YAMLAB/scripts/conveyor" "$YAMLAB/scripts/eval" "$YAMLAB/assets/hdris"
cp "$HERE"/scripts/{generate_conveyor_data.py,probe_conveyor.py} "$YAMLAB/scripts/conveyor/"
cp "$HERE"/../eval/client/{eval_longwam_remote.py,longwam_wire.py} "$YAMLAB/scripts/eval/"
cp -r "$HERE"/assets/hdris/train "$HERE"/assets/hdris/eval "$YAMLAB/assets/hdris/"

# Register the task module and its robot.
INIT="$YAMLAB/yamlab/envs/tasks/__init__.py"
grep -q conveyor_cube_pick_manager "$INIT" || printf 'from .conveyor_cube_pick_manager_cfg import *\nfrom .conveyor_cube_pick_manager import *\n' >> "$INIT"
REG="$YAMLAB/yamlab/envs/robot_registry.py"
grep -q '"ConveyorCubePick-v0"' "$REG" || python3 - "$REG" <<'EOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = re.sub(r"(TASK_ROBOTS = \{\n)", r'\1    "ConveyorCubePick-v0": "yam",\n', s, count=1)
open(p, "w").write(s)
EOF
grep -q '"ConveyorCubePick-v0"' "$REG" && echo "installed ConveyorCubePick-v0 into $YAMLAB"
