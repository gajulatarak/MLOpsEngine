#!/usr/bin/env python
"""Direct MLFlow Logging - Log the uploaded models to MLFlow without conversion"""

import sys
import os
from pathlib import Path
import json

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, str(Path.cwd()))

import mlflow
import warnings
warnings.filterwarnings('ignore')

# Configure MLFlow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('Sample_Models_Upload')

MODELS_DIR = Path.cwd() / 'model_store' / 'raw'

# Get all 10 recent models
models = []
for md in sorted(MODELS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
    if not md.is_dir():
        continue
    vd = sorted([d for d in md.iterdir() if d.is_dir()], reverse=True)
    if vd:
        mf = list(vd[0].glob('*'))
        if mf and len(models) < 10:
            file_size = mf[0].stat().st_size
            models.append((md.name, mf[0].name, file_size, mf[0].suffix))

print(f'\n{"="*80}')
print(f"MLFlow Logging - Recording {len(models)} Uploaded Models")
print(f"{"="*80}\n")

success_count = 0
for idx, (model_id, file_name, file_size, ext) in enumerate(models, 1):
    model_name = f'Model-{idx}'
    fname_short = file_name[:20] + "..." if len(file_name) > 20 else file_name
    
    print(f"[{idx}/{len(models)}] {model_name:12s} {fname_short:25s} ", end='', flush=True)
    
    try:
        # Log each model as a MLFlow run
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_param('model_id', model_id)
            mlflow.log_param('file_name', file_name)
            mlflow.log_param('format', ext.lower().lstrip('.'))
            mlflow.log_param('upload_type', 'sample_model')
            
            # Log metrics
            mlflow.log_metric('file_size_bytes', file_size)
            mlflow.log_metric('file_size_kb', file_size / 1024)
            
            # Log tags
            mlflow.set_tag('Environment', 'MLOps_Unified_Layer')
            mlflow.set_tag('Model_Index', str(idx))
            mlflow.set_tag('Status', 'Uploaded')
            
            success_count += 1
            run_id = mlflow.active_run().info.run_id
            print(f"[OK] Run: {run_id[:8]}")
    
    except Exception as e:
        print(f"[ERROR] {str(e)[:30]}")

print(f"\n{"="*80}")
print(f"SUCCESS: {success_count}/{len(models)} models logged to MLFlow")
print(f"{"="*80}")
print(f"""
✓ Models are now visible in MLFlow

Next steps:
1. Open MLFlow UI: http://127.0.0.1:5000
2. Click on "Sample_Models_Upload" experiment
3. You should see {success_count} runs (one per model)
4. Each run shows model metadata and file information

To register models for production use:
  python mlflow_convert_direct.py    (Convert to ONNX + register)
  python mlflow_quick_test.py        (Test conversion with 3 models)
""")
