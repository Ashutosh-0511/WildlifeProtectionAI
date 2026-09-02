#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install PytorchWildlife supervision
printf '\nML dependencies installed.\n'
python - <<'PY'
from PytorchWildlife.models import detection as pw_detection
print('Loading MegaDetector V6 (first run may download weights)...')
model = pw_detection.MegaDetectorV6(device='cpu')
print('MegaDetector V6 OK')
PY
