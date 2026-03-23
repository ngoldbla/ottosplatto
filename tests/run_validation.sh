#!/usr/bin/env bash
# OttoSplatto "test rabbit" — proves the full pipeline works out of the box.
#
# Generates a synthetic 3D scene, runs it through every pipeline stage,
# and reports pass/fail. No real video or external data needed.
#
# Usage:
#   ./tests/run_validation.sh                   # default 500 iterations
#   ./tests/run_validation.sh 100               # faster, fewer iterations
set -euo pipefail
cd "$(dirname "$0")/.."

ITERS="${1:-500}"
WORK="/tmp/ottosplatto_validation"
PASS=0
FAIL=0

ok()   { echo "  ✓ $1"; PASS=$((PASS+1)); }
fail() { echo "  ✗ $1"; FAIL=$((FAIL+1)); }

echo "OttoSplatto Validation"
echo "══════════════════════"
echo ""

# 1. Device
echo "1. Device detection"
python3 main.py device 2>/dev/null && ok "device detected" || fail "device detection"
echo ""

# 2. Generate test scene
echo "2. Generating test scene"
rm -rf "$WORK"
python3 tests/generate_test_scene.py --output "$WORK/synth3d" && ok "scene generated" || fail "scene generation"
echo ""

# 3. Create project
echo "3. Creating project"
python3 main.py create --name rabbit --input "$WORK/synth3d/images" --output "$WORK" \
  && ok "project created" || fail "project creation"
echo ""

# 4. Extract (copy images)
echo "4. Extracting frames"
python3 main.py extract --project "$WORK/rabbit" \
  && ok "frames extracted" || fail "frame extraction"
echo ""

# 5. Reconstruct
echo "5. COLMAP reconstruction"
python3 main.py reconstruct --project "$WORK/rabbit" --matcher exhaustive \
  && ok "reconstruction complete" || fail "COLMAP reconstruction"
echo ""

# 6. Train
echo "6. Training ($ITERS iterations)"
python3 main.py train --project "$WORK/rabbit" --iterations "$ITERS" \
  && ok "training complete" || fail "training"
echo ""

# 7. Check PLY
echo "7. Checking output"
PLY="$WORK/rabbit/output/point_cloud/iteration_${ITERS}/point_cloud.ply"
if [ -f "$PLY" ]; then
  SIZE=$(du -h "$PLY" | cut -f1)
  ok "PLY output: $PLY ($SIZE)"
else
  fail "no PLY file at $PLY"
fi
echo ""

# 8. Viewer
echo "8. Testing viewer"
python3 -c "
from pipeline.viewer import launch_viewer
import urllib.request
r = launch_viewer('$PLY', port=8765, open_browser=False)
resp = urllib.request.urlopen(r['url'])
assert resp.status == 200, f'HTTP {resp.status}'
r['server'].shutdown()
print('  viewer OK')
" 2>/dev/null && ok "viewer serves PLY" || fail "viewer"
echo ""

# Summary
echo "══════════════════════"
echo "Results: $PASS passed, $FAIL failed"
if [ "$FAIL" -eq 0 ]; then
  echo ""
  echo "OttoSplatto is working end-to-end."
else
  echo ""
  echo "Some steps failed — check output above."
  exit 1
fi
