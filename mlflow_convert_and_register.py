"""
MLFlow Conversion and Registration Script
Converts all recently uploaded sample models to ONNX format and registers them with MLFlow.
Displays comprehensive results table.
"""

import sys
import os
import time
import json
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
import requests

warnings.filterwarnings("ignore")

# Configuration
API_BASE = "http://127.0.0.1:5001"
MODELS_DIR = Path(__file__).parent / "model_store" / "raw"


def find_recent_models(limit: int = 10) -> List[Tuple[str, datetime]]:
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
        
        # Parse timestamp from version name (YYYYMMDD_HHMMSS format)
        try:
            timestamp = datetime.strptime(latest_version, "%Y%m%d_%H%M%S")
            models.append((model_id, timestamp))
        except ValueError:
            continue
    
    # Sort by timestamp (most recent first) and limit
    models.sort(key=lambda x: x[1], reverse=True)
    return models[:limit]


def get_model_info(model_id: str) -> Dict:
    """Get model information from filesystem."""
    model_dir = MODELS_DIR / model_id
    
    if not model_dir.exists():
        return None
    
    version_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()], reverse=True)
    if not version_dirs:
        return None
    
    latest_version = version_dirs[0]
    model_files = list(latest_version.glob('*'))
    
    if not model_files:
        return None
    
    model_file = model_files[0]
    
    # Try to detect format and get metadata
    info = {
        'model_id': model_id,
        'version': latest_version.name,
        'file_name': model_file.name,
        'file_size': model_file.stat().st_size,
        'file_format': model_file.suffix
    }
    
    return info


def convert_and_register_model(model_id: str, model_name: str, retries: int = 2) -> Tuple[bool, Dict]:
    """Convert and register a model via API with retry logic."""
    for attempt in range(retries):
        try:
            url = f"{API_BASE}/api/models/{model_id}/convert-register"
            payload = {
                'model_name': model_name,
                'target_format': 'onnx',
                'user': 'mlflow_registration'
            }
            
            response = requests.post(url, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return True, result.get('data', {})
            elif response.status_code >= 500:
                # Server error, may be recoverable
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                error_msg = response.json().get('error', f'Server error {response.status_code}')
                return False, {'error': error_msg, 'status': response.status_code}
            else:
                error_msg = response.json().get('error', 'Unknown error')
                return False, {'error': error_msg, 'status': response.status_code}
        
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            return False, {'error': 'Request timeout', 'status': -1}
        except requests.exceptions.ConnectionError as e:
            if attempt < retries - 1:
                time.sleep(3)
                continue
            return False, {'error': f'Connection failed: {str(e)}', 'status': -1}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            return False, {'error': str(e), 'status': -1}
    
    return False, {'error': 'Maximum retries exceeded', 'status': -1}


def display_progress(current: int, total: int, model_name: str, status: str):
    """Display progress indicator."""
    progress = f"[{current:2d}/{total}]"
    status_symbol = "*" if status == "processing" else ("+" if status == "success" else "-")
    print(f"{progress} {status_symbol} {model_name:40s} ...", end='', flush=True)


def main():
    """Main execution flow."""
    print("\n" + "="*100)
    print("MLFlow Model Conversion and Registration")
    print("="*100)
    
    # Step 1: Discover models
    print("\n[STEP 1] Discovering recently uploaded models...")
    recent_models = find_recent_models(limit=10)
    
    if not recent_models:
        print("  [ERROR] No recent models found")
        return
    
    print(f"  [OK] Found {len(recent_models)} recent models")
    
    # Step 2: Display discovered models
    print("\n[STEP 2] Model details:")
    print("-" * 100)
    print(f"{'Rank':<6} {'Model ID':<38} {'Uploaded':<20} {'File':<30}")
    print("-" * 100)
    
    model_names = {}
    for idx, (model_id, timestamp) in enumerate(recent_models, 1):
        info = get_model_info(model_id)
        if info:
            file_name = info['file_name'][:25] + "..." if len(info['file_name']) > 25 else info['file_name']
            print(f"{idx:<6} {model_id:<38} {timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} {file_name:<30}")
            model_names[model_id] = f"Model-{idx}"
    
    # Step 3: Convert and register models
    print("\n[STEP 3] Converting and registering models...")
    print("-" * 100)
    
    results = []
    successes = 0
    failures = 0
    
    for idx, (model_id, _) in enumerate(recent_models, 1):
        model_name = model_names[model_id]
        display_progress(idx, len(recent_models), model_name, "processing")
        
        success, result = convert_and_register_model(model_id, model_name, retries=3)
        
        if success:
            status = "PASS"
            successes += 1
            onnx_size = result.get('onnx_size', 'N/A')
            mlflow_run = result.get('mlflow_run_id', 'N/A')
            print(f"\r[{idx:2d}/{len(recent_models)}] + {model_name:40s} REGISTERED | ONNX: {onnx_size} | MLFlow: {mlflow_run[:8]}")
        else:
            status = "FAIL"
            failures += 1
            error = result.get('error', 'Unknown error')
            print(f"\r[{idx:2d}/{len(recent_models)}] - {model_name:40s} FAILED    | Error: {error}")
        
        results.append({
            'idx': idx,
            'model_id': model_id,
            'model_name': model_name,
            'status': status,
            'result': result
        })
        
        # Wait between requests (longer for stability)
        time.sleep(2)
    
    # Step 4: Display summary
    print("\n" + "-" * 100)
    print("CONVERSION AND REGISTRATION SUMMARY")
    print("-" * 100)
    print(f"\nTotal models processed: {len(recent_models)}")
    print(f"Successful registrations: {successes}")
    print(f"Failed registrations: {failures}")
    
    if successes > 0:
        print(f"Success rate: {(successes/len(recent_models)*100):.1f}%")
    
    # Step 5: Display detailed results table
    if successes > 0:
        print("\n" + "="*120)
        print("DETAILED REGISTRATION RESULTS")
        print("="*120)
        print(f"{'#':<4} {'Model Name':<30} {'Format':<10} {'Size':<12} {'MLFlow Run':<15} {'Status':<8}")
        print("-"*120)
        
        for result in results:
            if result['status'] == 'PASS':
                model_name = result['model_name']
                res_data = result['result']
                format_type = res_data.get('target_format', 'onnx').upper()
                size_val = res_data.get('onnx_size', 'N/A')
                if isinstance(size_val, (int, float)):
                    size_str = f"{size_val/1024:.1f} KB"
                else:
                    size_str = str(size_val)
                mlflow_run = res_data.get('mlflow_run_id', 'N/A')
                if mlflow_run != 'N/A':
                    mlflow_run = mlflow_run[:12]
                
                print(f"{result['idx']:<4} {model_name:<30} {format_type:<10} {size_str:<12} {str(mlflow_run):<15} PASS     ")
    
    # Step 6: Display failures
    if failures > 0:
        print("\n" + "="*120)
        print("REGISTRATION FAILURES")
        print("="*120)
        print(f"{'#':<4} {'Model Name':<30} {'Error':<80}")
        print("-"*120)
        
        for result in results:
            if result['status'] == 'FAIL':
                model_name = result['model_name']
                error = result['result'].get('error', 'Unknown error')
                error_truncated = error[:75] + "..." if len(error) > 75 else error
                print(f"{result['idx']:<4} {model_name:<30} {error_truncated:<80}")
    
    # Step 7: Next steps
    print("\n" + "="*120)
    print("NEXT STEPS")
    print("="*120)
    print(f"""
1. View MLFlow tracking server:
   http://127.0.0.1:5000

2. Query registered models via API:
   curl {API_BASE}/api/models

3. Access converted ONNX models:
   {MODELS_DIR}/converted/

4. Check model lineage and audit trails:
   curl {API_BASE}/api/models/{{model_id}}/audit-trail

5. Make predictions with registered models:
   curl -X POST {API_BASE}/api/inference/{{model_name}}/predict -H "Content-Type: application/json" -d '{{"features": [...]}}'
""")
    
    print("="*120)


if __name__ == "__main__":
    main()
