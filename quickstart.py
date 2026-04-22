"""
Quick Start Guide - Unified MLOps Layer

This script guides you through setting up and using the unified MLOps layer.
"""

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def installation_guide():
    """Guide for installation"""
    print_section("1. INSTALLATION & SETUP")
    
    commands = [
        ("Create virtual environment", "python -m venv venv"),
        ("Activate virtual environment", "source venv/bin/activate  # On Windows: venv\\Scripts\\activate"),
        ("Install dependencies", "pip install -r requirements.txt"),
        ("Initialize storage directories", "python -c \"from config import config; print('✓ Stored initialized')\""),
    ]
    
    for step, cmd in commands:
        print(f"→ {step}")
        print(f"  $ {cmd}\n")


def mlflow_setup():
    """Guide for MLFlow setup"""
    print_section("2. MLFLOW SETUP")
    
    print("Start MLFlow server in a separate terminal:\n")
    print("  $ mlflow server \\")
    print("      --backend-store-uri sqlite:///mlflow/mlflow.db \\")
    print("      --default-artifact-root ./mlflow/artifacts \\")
    print("      --host 0.0.0.0 \\")
    print("      --port 5000\n")
    
    print("✓ MLFlow UI will be available at: http://localhost:5000")
    print("✓ Check health: curl http://localhost:5000/health\n")


def api_startup():
    """Guide for API startup"""
    print_section("3. START THE API SERVER")
    
    print("In your main terminal:\n")
    print("  $ python api.py\n")
    print("✓ API will be available at: http://localhost:5001\n")
    print("Access the Web UI: http://localhost:5001\n")


def basic_workflow():
    """Guide for basic workflow"""
    print_section("4. BASIC WORKFLOW")
    
    steps = [
        ("Train a model", "Use your preferred ML framework (sklearn, PyTorch, TensorFlow, etc.)"),
        ("Upload model", "POST /api/models/upload with model file"),
        ("Register model", "POST /api/models/{model_id}/convert-register"),
        ("Make predictions", "POST /api/inference/{model_name}/predict"),
        ("Track lineage", "GET /api/models/{model_id}/lineage"),
    ]
    
    for i, (step, description) in enumerate(steps, 1):
        print(f"{i}. {step}")
        print(f"   → {description}\n")


def python_integration():
    """Guide for Python integration"""
    print_section("5. PYTHON INTEGRATION EXAMPLE")
    
    example = '''
from orchestrator import mlops_orchestrator
from config import ModelFormat
from pathlib import Path

# Upload model
success, msg, result = mlops_orchestrator.upload_model(
    source_file_path=Path("my_model.pkl"),
    model_name="my_classifier",
    user="scientist@company.com"
)

if success:
    model_id = result['model_id']
    
    # Convert and register
    success, msg, reg_result = mlops_orchestrator.convert_and_register(
        model_id=model_id,
        model_name="my_classifier",
        target_format=ModelFormat.ONNX
    )
    
    if success:
        print(f"Model registered: {reg_result['model_uri']}")
'''
    
    print(example)


def rest_api_examples():
    """Guide for REST API usage"""
    print_section("6. REST API EXAMPLES")
    
    print("1️⃣  Upload a model:")
    print('''
curl -X POST http://localhost:5001/api/models/upload \\
  -F "file=@model.pkl" \\
  -F "model_name=my_classifier" \\
  -F "user=scientist@company.com"
''')
    
    print("\n2️⃣  Convert and register:")
    print('''
curl -X POST http://localhost:5001/api/models/{model_id}/convert-register \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_name": "my_classifier",
    "target_format": "onnx",
    "user": "scientist@company.com"
  }'
''')
    
    print("\n3️⃣  Make prediction:")
    print('''
curl -X POST http://localhost:5001/api/inference/my_classifier_onnx/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "data": [[5.1, 3.5, 1.4, 0.2]],
    "stage": "Production"
  }'
''')
    
    print("\n4️⃣  Get audit trail:")
    print('''
curl http://localhost:5001/api/models/{model_id}/audit-trail
''')
    
    print("\n5️⃣  List models:")
    print('''
curl http://localhost:5001/api/models
''')


def monitoring_mlflow():
    """Guide for MLFlow monitoring"""
    print_section("7. MLFlow MONITORING DASHBOARD")
    
    print("Open http://localhost:5000 to see:\n")
    
    features = [
        "Model Registry - All registered models and versions",
        "Experiments - Training runs and metrics",
        "Model Stages - Staging, Production, Archived",
        "Parameters & Metrics - Track model performance",
        "Artifacts - Model files and artifacts",
        "Model Lineage - Parent-child relationships",
    ]
    
    for feature in features:
        print(f"  ✓ {feature}")


def audit_compliance():
    """Guide for audit and compliance"""
    print_section("8. AUDIT & COMPLIANCE TRACKING")
    
    print("Complete audit trail captures:\n")
    
    items = [
        "Model uploads with file hashes (MD5, SHA256)",
        "Format conversions and conversion methods",
        "MLFlow registration events",
        "Inference/prediction usage",
        "Model lifecycle transitions",
        "User and timestamp information",
        "Environment details at registration time",
        "Lineage traces showing model transformations",
    ]
    
    for item in items:
        print(f"  ✓ {item}")
    
    print("\nAccess audit data via:")
    print("  - http://localhost:5001/api/models/{model_id}/audit-trail")
    print("  - http://localhost:5001/api/models/{model_id}/lineage")
    print("  - File: logs/audit.log")
    print("  - Database: mlflow/audit.db")


def supported_formats_guide():
    """Guide for supported formats"""
    print_section("9. SUPPORTED MODEL FORMATS")
    
    formats = {
        "Deep Learning": ["ONNX", "TensorFlow SavedModel", "Keras H5", "PyTorch (.pt/.pth)"],
        "Classical ML": ["Pickle (.pkl)", "Joblib", "PMML"],
        "Tree-based": ["XGBoost", "LightGBM", "CatBoost"],
        "Export Formats": ["ONNX", "TorchScript", "TensorRT", "OpenVINO"],
    }
    
    for category, fmt_list in formats.items():
        print(f"{category}:")
        for fmt in fmt_list:
            print(f"  ✓ {fmt}")
        print()


def troubleshooting():
    """Guide for troubleshooting"""
    print_section("10. TROUBLESHOOTING")
    
    issues = [
        ("MLFlow connection error", 
         "Check if MLFlow server is running: curl http://localhost:5000/health"),
        
        ("Model format not detected",
         "Check file extension and ensure file format is correct"),
        
        ("Conversion failed",
         "Verify all required framework packages are installed: pip list"),
        
        ("Permission denied",
         "Check write permissions in:.\\nmodel_store/ directory"),
        
        ("Port already in use",
         "Change port in config.py or kill existing process"),
    ]
    
    for issue, solution in issues:
        print(f"❌ {issue}")
        print(f"   → Solution: {solution}\n")


def next_steps():
    """Guide for next steps"""
    print_section("11. NEXT STEPS")
    
    steps = [
        "Run examples.py to see various use cases",
        "Run tests: pytest test_mlops.py -v",
        "Explore MLFlow dashboard at http://localhost:5000",
        "Check audit logs at logs/audit.log",
        "Integrate with your CI/CD pipeline",
        "Configure for cloud deployment (AWS/Azure)",
    ]
    
    for step in steps:
        print(f"→ {step}")


def main():
    """Run complete quick start guide"""
    print("\n" + "="*70)
    print("  UNIFIED MLOps LAYER - QUICK START GUIDE")
    print("="*70)
    
    installation_guide()
    mlflow_setup()
    api_startup()
    basic_workflow()
    python_integration()
    rest_api_examples()
    monitoring_mlflow()
    audit_compliance()
    supported_formats_guide()
    troubleshooting()
    next_steps()
    
    print("\n" + "="*70)
    print("  QUICK START COMPLETE!")
    print("="*70)
    print("\nDocumentation: README.md")
    print("Examples: examples.py")
    print("Tests: test_mlops.py")
    print("\n")


if __name__ == '__main__':
    main()
