#!/usr/bin/env bash
# WP4 box setup: verify and unpack the capsule, build the venv from the pinned list, write env.sh, test.
# Run from /root/work after the capsule and CAPSULE-SHA256.txt are uploaded there.
set -euo pipefail
W=/root/work
cd "$W"
sha256sum -c CAPSULE-SHA256.txt
tar -xzf capsule.tar.gz
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git >/dev/null; }
python -m venv --system-site-packages /root/venv
/root/venv/bin/pip install -q --upgrade pip
/root/venv/bin/pip install -q -r "$W/tessera-st/repository/revision-r3/remote/requirements-box.txt"
cat > /root/env.sh <<EOF2
export PY=/root/venv/bin/python
export TESSERA_R3_PY_GRAPHST=/root/venv/bin/python TESSERA_R3_PY_STAGATE=/root/venv/bin/python TESSERA_R3_PY_TESSERA=/root/venv/bin/python
export TESSERA_STAGATE_ROOT=$W/STAGATE
export TESSERA_DATA=$W/tessera-st/data TESSERA_RESULTS=$W/tessera-st/results TESSERA_VALIDATION=$W/tessera-st/validation
export TESSERA_DATA_R3=$W/tessera-st/data-r3 TESSERA_R3_OUT=$W/tessera-st/results/revision-r3/W4
export PROJ=$W/tessera-st TQDM_DISABLE=1
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMBA_NUM_THREADS=3
EOF2
source /root/env.sh
$PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.__version__)"
$PY -m pytest "$PROJ/repository/revision-r3/external/test_tessera_adapter_r3.py" -q -p no:cacheprovider
timeout 1800 $PY -m pytest "$PROJ/repository/revision-r3/external/test_external_r3.py" -q -p no:cacheprovider
$PY "$PROJ/repository/revision-r3/external/run_external_r3.py" --encoders graphst stagate --datasets core --arms released --configs all --seeds 1 2 3 --input-modes common --dry-run --no-timing | tail -8
echo SETUP-OK
