#!/usr/bin/env python
"""Quick MLFlow Registration - Register 3 models to make them visible in MLFlow"""

import sys
import os
from pathlib import Path

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, str(Path.cwd()))

from config import config, ModelFormat
from orchestrator import mlops_orchestrator
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = Path.cwd() / 'model_store' / 'raw'

# Get 3 most recent models
models = []
for md in sorted(MODELS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
    if not md.is_dir():
        continue
    vd = sorted([d for d in md.iterdir() if d.is_dir()], reverse=True)
    if vd:
        mf = list(vd[0].glob('*'))
        if mf and len(models) < 3:
            models.append((md.name, mf[0].name))

print(f'\n{"="*70}')
print(f"MLFlow Registration - Adding {len(models)} Models")
print(f"{"="*70}\n")

successes = 0
for idx, (model_id, file_name) in enumerate(models, 1):
    model_name = f'model-{idx}'
    fname_short = file_name[:22] + "..." if len(file_name) > 22 else file_name
    print(f"[{idx}/{len(models)}] {model_name:15s} ({fname_short:25s}) ", end='', flush=True)
    
    try:
        success, message, result = mlops_orchestrator.convert_and_register(
            model_id=model_id,
            model_name=model_name,
            target_format=ModelFormat.ONNX,
            user='registration'
        )
        
        if success:
            successes += 1
            print("[OK]")
        else:
            print(f"[FAIL]")
    except Exception as e:
        print(f"[ERROR]")

print(f"\n{"="*70}")
print(f"Results: {successes}/{len(models)} models registered with MLFlow")
print(f"{"="*70}")
print(f"\nMLFlow UI: http://127.0.0.1:5000")
print(f"You should now see {successes} new experiment runs\n")
