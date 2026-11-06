#!/usr/bin/env bash
# Prepare a rented GPU box for Amendment A2 (PROTOCOL-R1.md).
# Image: pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime, the PyTorch build used on the laptop.
# Usage: bash setup_box.sh ROOT COMMIT
#   ROOT holds the uploaded capsule: inputs/data (lung inputs), inputs/E2 (tables A2 reads) and inputs.sha256.
#   COMMIT is the repository commit to run; its source is downloaded from GitHub on the box.
# PY selects the interpreter (default python); A2_SKIP_PIP=1 skips the package install (laptop dry runs only).
set -euo pipefail
PY=${PY:-python}
ROOT=${1:?root}
COMMIT=${2:?commit}
cd "$ROOT"
REC=r1out/E2/remote-a2
mkdir -p "$REC" r1out/E2
{
  echo "setup started $(date -u +%FT%TZ)"
  sha256sum -c --quiet inputs.sha256 && echo "capsule checksums: OK"
  "$PY" - "$COMMIT" <<'PY'
import io, sys, tarfile, urllib.request
commit = sys.argv[1]
url = f'https://github.com/PeterPonyu/tessera-st/archive/{commit}.tar.gz'
with urllib.request.urlopen(url, timeout=600) as r:
    data = r.read()
archive = tarfile.open(fileobj=io.BytesIO(data))
try:
    archive.extractall('.', filter='data')
except TypeError:  # Python without extraction filters
    archive.extractall('.')
print('source', commit, len(data), 'bytes')
PY
  mv "tessera-st-$COMMIT" repo
  if [ "${A2_SKIP_PIP:-0}" != 1 ]; then
    "$PY" -m pip install --no-cache-dir -q numpy==2.2.6 scipy==1.16.2 scikit-learn==1.6.1 pandas==2.3.3
  fi
  cp inputs/E2/*.csv r1out/E2/
  echo "setup finished $(date -u +%FT%TZ)"
} > "$REC/setup.log" 2>&1
nvidia-smi > "$REC/nvidia-smi.txt"
lscpu > "$REC/lscpu.txt" 2>/dev/null || true
"$PY" -m pip freeze > "$REC/pip-freeze.txt"
"$PY" - > "$REC/machine.json" <<'PY'
import json, os, platform, socket
import numpy, pandas, scipy, sklearn, torch
p = torch.cuda.get_device_properties(0)
print(json.dumps(dict(
    hostname=socket.gethostname(), platform=platform.platform(), python=platform.python_version(),
    cpu_count=os.cpu_count(), torch=torch.__version__, torch_cuda=torch.version.cuda,
    cudnn=torch.backends.cudnn.version(), gpu=p.name, gpu_memory_gb=round(p.total_memory / 1e9, 1),
    gpu_capability=f'{p.major}.{p.minor}', numpy=numpy.__version__, scipy=scipy.__version__,
    sklearn=sklearn.__version__, pandas=pandas.__version__), indent=1))
PY
echo "setup complete"
