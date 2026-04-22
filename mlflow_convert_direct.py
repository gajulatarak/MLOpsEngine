"""
Direct MLFlow Conversion and Registration Script
Converts all recently uploaded models to ONNX and registers with MLFlow directly.
Avoids HTTP API timeouts by using backend functions directly.
"""

import sys
import os
import time
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import traceback

warnings.filterwarnings("ignore")

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config, ModelFormat
from orchestrator import mlops_orchestrator
from model_handlers import handler_registry


MODELS_DIR = Path(__file__).parent / "model_store" / "raw"


def find_recent_models(limit: int = 10) -> List[Tuple[str, datetime, str]]:
    """Find most recently uploaded models by scanning filesystem."""
    models = []
    
    if not MODELS_DIR.exists():
        print(f"[ERROR] Models directory not found: {MODELS_DIR}")
        return []
    
    # Scan all model directories
    for model_id_dir in MODELS_DIR.iterdir():
        if not model_id_dir.is_dir():
            continue
        
        model_id = model_id_dir.name
        
        # Get latest version directory
        version_dirs = sorted([d for d in model_id_dir.iterdir() if d.is_dir()], reverse=True)
        if not version_dirs:
            continue
        
        latest_version = version_dirs[0].name
        
        # Parse timestamp and get model file
        try:
            timestamp = datetime.strptime(latest_version, "%Y%m%d_%H%M%S")
            model_files = list(version_dirs[0].glob('*'))
            if model_files:
                file_name = model_files[0].name
                models.append((model_id, timestamp, file_name))
        except ValueError:
            continue
    
    # Sort by timestamp (most recent first) and limit
    models.sort(key=lambda x: x[1], reverse=True)
    return models[:limit]


def main():
    """Main execution flow."""
    print("\n" + "="*120)
    print("MLFlow Model Conversion and Registration (Direct Backend)")
    print("="*120)
    
    # Step 1: Discover models
    print("\n[STEP 1] Discovering recently uploaded models...")
    recent_models = find_recent_models(limit=10)
    
    if not recent_models:
        print("  [ERROR] No recent models found")
        return
    
    print(f"  [OK] Found {len(recent_models)} recent models")
    
    # Step 2: Display discovered models
    print("\n[STEP 2] Model details:")
    print("-" * 120)
    print(f"{'Rank':<6} {'Model ID':<38} {'Uploaded':<20} {'File':<35}")
    print("-" * 120)
    
    model_names = {}
    for idx, (model_id, timestamp, file_name) in enumerate(recent_models, 1):
        display_name = file_name[:32] + "..." if len(file_name) > 32 else file_name
        print(f"{idx:<6} {model_id:<38} {timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} {display_name:<35}")
        model_names[model_id] = f"Model-{idx}"
    
    # Step 3: Convert and register models
    print("\n[STEP 3] Converting and registering models with MLFlow...")
    print("-" * 120)
    
    results = []
    successes = 0
    failures = 0
    
    for idx, (model_id, _, file_name) in enumerate(recent_models, 1):
        model_name = model_names[model_id]
        print(f"[{idx:2d}/{len(recent_models)}] Processing {model_name:40s} ...", end='', flush=True)
        
        try:
            success, message, result = mlops_orchestrator.convert_and_register(
                model_id=model_id,
                model_name=model_name,
                target_format=ModelFormat.ONNX,
                user='mlflow_registration'
            )
            
            if success:
                status = "PASS"
                successes += 1
                onnx_size = result.get('onnx_size') if isinstance(result, dict) else result
                mlflow_run = result.get('mlflow_run_id', 'N/A') if isinstance(result, dict) else 'N/A'
                if isinstance(mlflow_run, str) and mlflow_run != 'N/A' and len(mlflow_run) > 12:
                    mlflow_run = mlflow_run[:12]
                print(f" REGISTERED | Size: {onnx_size} | MLFlow: {mlflow_run}")
            else:
                status = "FAIL"
                failures += 1
                print(f" FAILED     | Error: {message}")
                result = {'error': message}
            
            results.append({
                'idx': idx,
                'model_id': model_id,
                'model_name': model_name,
                'file_name': file_name,
                'status': status,
                'message': message if status == 'FAIL' else 'Conversion and registration successful',
                'result': result if isinstance(result, dict) else {'onnx_size': result}
            })
        
        except Exception as e:
            status = "FAIL"
            failures += 1
            error_msg = str(e)
            print(f" ERROR      | {error_msg[:60]}")
            
            results.append({
                'idx': idx,
                'model_id': model_id,
                'model_name': model_name,
                'file_name': file_name,
                'status': status,
                'message': error_msg,
                'result': {'error': error_msg}
            })
    
    # Step 4: Display summary
    print("\n" + "-" * 120)
    print("CONVERSION AND REGISTRATION SUMMARY")
    print("-" * 120)
    print(f"\nTotal models processed: {len(recent_models)}")
    print(f"Successful registrations: {successes}")
    print(f"Failed registrations: {failures}")
    
    if successes > 0:
        print(f"Success rate: {(successes/len(recent_models)*100):.1f}%")
    
    # Step 5: Display detailed results table
    if successes > 0:
        print("\n" + "="*130)
        print("SUCCESSFUL REGISTRATION RESULTS")
        print("="*130)
        print(f"{'#':<4} {'Model Name':<30} {'Source File':<32} {'Format':<10} {'Status':<8}")
        print("-"*130)
        
        for result in results:
            if result['status'] == 'PASS':
                model_name = result['model_name']
                file_name = result['file_name'][:29] + "..." if len(result['file_name']) > 29 else result['file_name']
                print(f"{result['idx']:<4} {model_name:<30} {file_name:<32} {'ONNX':<10} PASS    ")
    
    # Step 6: Display failures
    if failures > 0:
        print("\n" + "="*130)
        print("REGISTRATION FAILURES")
        print("="*130)
        print(f"{'#':<4} {'Model Name':<30} {'Error Message':<90}")
        print("-"*130)
        
        for result in results:
            if result['status'] == 'FAIL':
                model_name = result['model_name']
                error = result['message'] if result['message'] else result['result'].get('error', 'Unknown error')
                error_truncated = error[:85] + "..." if len(error) > 85 else error
                print(f"{result['idx']:<4} {model_name:<30} {error_truncated:<90}")
    
    # Step 7: Next steps
    print("\n" + "="*130)
    print("NEXT STEPS")
    print("="*130)
    print(f"""
1. View MLFlow tracking server:
   http://127.0.0.1:5000

2. Check registered models in MLFlow:
   - Visit http://127.0.0.1:5000
   - Models should appear in the model registry

3. Access converted ONNX models:
   {MODELS_DIR}/converted/

4. View model lineage and audit trails:
   {MODELS_DIR}/../artifacts/

5. Next: Run inference tests with registered models:
   python inference_test.py
""")
    
    print("="*130)
    print()


if __name__ == "__main__":
    main()
