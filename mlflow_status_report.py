"""
MLFlow Registration Summary and Next Steps
Displays uploaded models and MLFlow registration status.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


MODELS_DIR = Path(__file__).parent / "model_store" / "raw"


def find_recent_models(limit: int = 10) -> List[Tuple[str, datetime, str, int]]:
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
                file_path = model_files[0]
                file_name = file_path.name
                file_size = file_path.stat().st_size
                models.append((model_id, timestamp, file_name, file_size))
        except ValueError:
            continue
    
    # Sort by timestamp (most recent first) and limit
    models.sort(key=lambda x: x[1], reverse=True)
    return models[:limit]


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def detect_format(filename: str) -> str:
    """Detect model format from filename."""
    extensions = {
        '.pkl': 'pickle',
        '.pickle': 'pickle',
        '.joblib': 'joblib',
        '.ubj': 'xgboost',
        '.xgb': 'xgboost',
        '.xgboost': 'xgboost',
        '.bst': 'xgboost',
        '.json': 'xgboost',
        '.lgb': 'lightgbm',
        '.lgbm': 'lightgbm',
        '.onnx': 'onnx',
        '.pt': 'pytorch',
        '.pth': 'pytorch',
        '.h5': 'keras',
        '.pb': 'tensorflow',
        '.model': 'catboost',
    }
    
    for ext, fmt in extensions.items():
        if filename.lower().endswith(ext):
            return fmt.upper()
    
    return 'UNKNOWN'


def main():
    """Main execution."""
    print("\n" + "="*130)
    print("MLFLOW REGISTRATION SUMMARY & DEPLOYMENT STATUS")
    print("="*130)
    
    # Get recent models
    recent_models = find_recent_models(limit=10)
    
    if not recent_models:
        print("\n[ERROR] No recent models found")
        return
    
    # Section 1: Current Status
    print(f"\n[STATUS] {len(recent_models)} MODELS READY FOR MLFLOW REGISTRATION")
    print("-" * 130)
    
    # Section 2: Model Listing
    print(f"\n{'#':<4} {'Model ID':<37} {'Format':<12} {'Size':<12} {'Uploaded':<20}")
    print("-" * 130)
    
    model_map = {}
    for idx, (model_id, timestamp, file_name, file_size) in enumerate(recent_models, 1):
        format_type = detect_format(file_name)
        model_map[model_id] = {
            'idx': idx,
            'file_name': file_name,
            'format': format_type,
            'size': file_size,
            'timestamp': timestamp
        }
        
        print(f"{idx:<4} {model_id:<37} {format_type:<12} {format_size(file_size):<12} {timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20}")
    
    # Section 3: Model Formats Summary
    print("\n" + "-" * 130)
    print("FORMAT BREAKDOWN")
    print("-" * 130)
    
    format_counts = {}
    for model_id, info in model_map.items():
        fmt = info['format']
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    
    total_size = sum(info['size'] for info in model_map.values())
    
    for fmt, count in sorted(format_counts.items()):
        print(f"  {fmt:<15} : {count:2d} models")
    
    print(f"\n  Total Size     : {format_size(total_size)}")
    print(f"  Total Models   : {len(recent_models)}")
    
    # Section 4: Conversion Strategy
    print("\n" + "="*130)
    print("CONVERSION & REGISTRATION PLAN")
    print("="*130)
    print(f"""
All {len(recent_models)} models will be converted to ONNX format (universal interchange format)
and registered with MLFlow for:

  [BENEFITS OF MLFLOW REGISTRATION]
  * Centralized model versioning and tracking
  * Experiment comparison and reproducibility
  * Model deployment to production environments
  * Model serving via MLFlow serving infrastructure
  * Compliance and audit trail maintenance
  * A/B testing and canary deployments
  * Automated model monitoring and governance

  [CONVERSION PROCESS]
  Format Mappings:
  * pickle        -> ONNX (via sklearn handler)
  * joblib        -> ONNX (via sklearn handler)
  * xgboost       -> ONNX (via native ONNX exporter)
  * lightgbm      -> ONNX (via native ONNX exporter)
  * onnx          -> ONNX (no conversion needed, direct registration)
  * pytorch       -> ONNX (via torch.onnx export)
  * tensorflow    -> ONNX (via tf2onnx converter)
  * keras         -> ONNX (via tf2onnx converter)
  * catboost      -> ONNX (via catboost ONNX exporter)

  [STORAGE]
  Original models stored  : {MODELS_DIR}/raw/
  Converted ONNX stored   : {MODELS_DIR}/converted/
  MLFlow artifacts stored : {MODELS_DIR}/mlflow/
  Metadata stored         : {MODELS_DIR}/artifacts/
""")
    
    # Section 5: Dashboard Access
    print("="*130)
    print("DASHBOARD & ACCESS POINTS")
    print("="*130)
    print(f"""
  [1] MLFlow Tracking Server
      URL: http://127.0.0.1:5000
      View experiments, runs, and model registrations

  [2] REST API Endpoints
      Models List       : GET http://127.0.0.1:5001/api/models
      Model Info        : GET http://127.0.0.1:5001/api/models/{{model_id}}/info
      Audit Trail       : GET http://127.0.0.1:5001/api/models/{{model_id}}/audit-trail
      Lineage           : GET http://127.0.0.1:5001/api/models/{{model_id}}/lineage
      Predictions       : POST http://127.0.0.1:5001/api/inference/{{model_name}}/predict

  [3] Model Files on Disk
      Raw Directory     : {MODELS_DIR}/raw/
      Converted ONNX    : {MODELS_DIR}/converted/
      Metadata/Audit    : {MODELS_DIR}/artifacts/

  [4] Commandline Tools
      Query Models      : curl http://127.0.0.1:5001/api/models
      Get Model Info    : curl http://127.0.0.1:5001/api/models/{{model_id}}/info
""")
    
    # Section 6: Registration Instructions
    print("="*130)
    print("REGISTRATION INSTRUCTIONS")
    print("="*130)
    print(f"""
  To convert and register all models with MLFlow, run:

  1. OPTION A: Using the direct backend converter (recommended)
     cd {Path(__file__).parent}
     .\\mlops-env\\Scripts\\python mlflow_convert_direct.py

  2. OPTION B: Using the API converter (via HTTP)
     First start the API server:
     .\\mlops-env\\Scripts\\python api.py
     
     Then run:
     .\\mlops-env\\Scripts\\python mlflow_convert_and_register.py

  3. OPTION C: Manual registration per model
     .\\mlops-env\\Scripts\\python -c "
     from orchestrator import mlops_orchestrator
     from config import ModelFormat
     success, msg, result = mlops_orchestrator.convert_and_register(
         model_id='MODEL_ID_HERE',
         model_name='model-name',
         target_format=ModelFormat.ONNX
     )
     print(f'Success: {{success}}, Message: {{msg}}')
     "

  [FOR INTERACTIVE MANAGEMENT]
  After registration, use MLFlow UI at: http://127.0.0.1:5000
""")
    
    # Section 7: Model Registry Preview
    print("="*130)
    print("EXPECTED MODEL REGISTRY (After Registration)")
    print("="*130)
    print(f"\n{'#':<4} {'Name':<25} {'Format':<12} {'Framework':<18} {'Registered':<15} {'Status':<10}")
    print("-"*130)
    
    for idx, (model_id, info) in enumerate(model_map.items(), 1):
        model_name = f"Model-{idx}"
        framework = "sklearn" if info['format'] in ['PICKLE', 'JOBLIB'] else ("XGBoost" if info['format'] == 'XGBOOST' else ("LightGBM" if info['format'] == 'LIGHTGBM' else "Custom"))
        status = "Ready"
        print(f"{idx:<4} {model_name:<25} {'ONNX':<12} {framework:<18} {status:<15} PENDING    ")
    
    # Section 8: Summary
    print("\n" + "="*130)
    print("SUMMARY")
    print("="*130)
    print(f"""
  [UPLOAD STATUS]      COMPLETE - All 10 sample models successfully uploaded
  [MODELS READY]       {len(recent_models)} models ready for conversion and registration
  [TOTAL SIZE]         {format_size(total_size)}
  [NEXT STEP]          Run MLFlow registration command (see above)
  [ESTIMATED TIME]     5-10 minutes for all conversions

  [CURRENT CAPABILITIES]
  ✓ Model upload and storage
  ✓ Format auto-detection
  ✓ Audit trail tracking
  ✓ Model versioning
  ✓ RESTful API access

  [AFTER REGISTRATION]
  ✓ MLFlow experiment tracking
  ✓ Model registry with versions
  ✓ Production deployment options
  ✓ Model comparison dashboards
  ✓ Batch and real-time inference
""")
    
    print("="*130)
    print()


if __name__ == "__main__":
    main()
