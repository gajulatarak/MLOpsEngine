"""
MLFlow Quick Registration Test
Tests registration of 3 sample models to validate the process works.
"""

import sys
import os
from pathlib import Path

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from config import config, ModelFormat
from model_handlers import handler_registry
from converters import conversion_service
import mlflow
import json

MODELS_DIR = Path(__file__).parent / "model_store" / "raw"


def list_recent_models(limit=3):
    """Get most recent model IDs."""
    models = []
    if not MODELS_DIR.exists():
        return []
    
    for model_id_dir in MODELS_DIR.iterdir():
        if not model_id_dir.is_dir():
            continue
        
        version_dirs = sorted([d for d in model_id_dir.iterdir() if d.is_dir()], reverse=True)
        if version_dirs:
            model_files = list(version_dirs[0].glob('*'))
            if model_files:
                models.append((model_id_dir.name, model_files[0].name))
    
    return sorted(models, reverse=True)[:limit]


def test_register_model(model_id, model_name):
    """Test registration of a single model."""
    try:
        print(f"\n  Testing: {model_name} ({model_id})")
        print("  " + "-"*60)
        
        # Find model
        model_dir = MODELS_DIR / model_id
        version_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()], reverse=True)
        if not version_dirs:
            return False, "No model versions found"
        
        model_path = list(version_dirs[0].glob('*'))[0]
        print(f"  Found model: {model_path.name} ({model_path.stat().st_size} bytes)")
        
        # Detect format
        detected_format = handler_registry.detect_format(model_path)
        print(f"  Detected format: {detected_format}")
        
        # Setup MLFlow
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment("mlops_test")
        
        # If already ONNX, skip conversion
        if detected_format == ModelFormat.ONNX:
            print(f"  Already ONNX format - skipping conversion")
            with mlflow.start_run() as run:
                mlflow.log_param("model_id", model_id)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("format", "onnx")
                
                # Load and log ONNX model
                import onnx
                onnx_model = onnx.load(str(model_path))
                mlflow.onnx.log_model(onnx_model, "model", registered_model_name=model_name)
                
                print(f"  MLFlow Run ID: {run.info.run_id}")
                return True, f"Successfully registered {model_name}"
        else:
            # Try to convert to ONNX
            print(f"  Converting {detected_format} -> ONNX...")
            
            try:
                success, conv_msg, converted_path = conversion_service.convert_model(
                    model_id=model_id,
                    model_name=model_name,
                    source_path=model_path,
                    source_format=detected_format,
                    target_format=ModelFormat.ONNX,
                    user='test'
                )
                
                if not success:
                    return False, f"Conversion failed: {conv_msg}"
                
                print(f"  Conversion successful: {converted_path}")
                
                # Register with MLFlow
                with mlflow.start_run() as run:
                    mlflow.log_param("model_id", model_id)
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("source_format", detected_format.value)
                    mlflow.log_param("target_format", "onnx")
                    
                    import onnx
                    onnx_model = onnx.load(str(converted_path))
                    mlflow.onnx.log_model(onnx_model, "model", registered_model_name=model_name)
                    
                    print(f"  MLFlow Run ID: {run.info.run_id}")
                
                return True, f"Successfully converted and registered {model_name}"
            
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                return False, f"Conversion error: {error_msg}"
    
    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 100:
            error_msg = error_msg[:100] + "..."
        return False, f"Error: {error_msg}"


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("MLFlow Registration Quick Test")
    print("="*70)
    
    # Get recent models
    models = list_recent_models(limit=3)
    
    if not models:
        print("\n[ERROR] No models found")
        return
    
    print(f"\n[STEP 1] Found {len(models)} recent models")
    print("[STEP 2] Attempting registration...\n")
    
    results = []
    for idx, (model_id, file_name) in enumerate(models, 1):
        model_name = f"test-model-{idx}"
        success, message = test_register_model(model_id, model_name)
        results.append((model_name, success, message))
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Status':<10} {'Message':<35}")
    print("-"*70)
    
    successes = 0
    for model_name, success, message in results:
        status = "PASS" if success else "FAIL"
        msg_short = message[:32] + "..." if len(message) > 32 else message
        print(f"{model_name:<20} {status:<10} {msg_short:<35}")
        if success:
            successes += 1
    
    print("-"*70)
    print(f"Successful: {successes}/{len(results)}")
    
    if successes > 0:
        print("\n[SUCCESS] Basic MLFlow registration working!")
        print(f"View results at: http://127.0.0.1:5000")
    else:
        print("\n[WARNING] Some registrations failed - check messages above")
    
    print("="*70)


if __name__ == "__main__":
    main()
