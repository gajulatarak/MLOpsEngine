# Unified MLOps Layer with MLFlow Integration

A production-grade, enterprise-ready unified MLOps layer that accepts multiple model formats, performs format conversion, manages models with MLFlow, and maintains comprehensive audit trails and lineage tracking.

## Features

### 🎯 Core Capabilities

- **Multi-Format Support**: Seamless handling of 14+ model formats across different frameworks
- **Automatic Format Detection**: Intelligent detection of uploaded model formats
- **Format Conversion**: Convert models between formats with ONNX as the standard interchange format
- **MLFlow Integration**: Full integration with MLFlow for model registry and serving
- **Audit & Lineage Tracking**: Complete audit trail and lineage tracking for compliance and reproducibility
- **Inference Service**: Deploy models and perform batch/single predictions
- **REST API**: Complete REST API for integration with existing systems
- **Web UI**: User-friendly web interface for model uploads and management

### 📦 Supported Formats

#### Deep Learning Frameworks
- ONNX (Open Neural Network Exchange)
- TensorFlow SavedModel / Protobuf
- Keras H5
- PyTorch (.pt / .pth)

#### Classical ML Formats
- Pickle (.pkl) — scikit-learn
- Joblib
- PMML (standardized format)

#### Gradient Boosting / Tree Models
- XGBoost binary format
- LightGBM model files
- CatBoost format

#### Export/Interoperability Formats
- ONNX (central interchange format)
- TorchScript
- TensorRT engines
- OpenVINO IR

## Architecture

```
MLOps Unified Layer
├── Config Layer (config.py)
│   └── Centralized configuration management
├── Model Handlers (model_handlers.py)
│   └── Format-specific model loaders & validators
├── Converters (converters.py)
│   └── Format-to-format conversion logic
├── Audit & Lineage (audit.py)
│   └── Comprehensive audit trail & traceability
├── MLFlow Integration (mlflow_integration.py)
│   └── Model registry & inference serving
├── Orchestrator (orchestrator.py)
│   └── Central service orchestrating all workflows
└── API Layer (api.py)
    └── REST API & Web UI
```

## Installation

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (recommended)

### Setup

1. **Clone or create the project**:
```bash
cd MLOPsEngine
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Initialize MLFlow**:
```bash
mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db \
              --default-artifact-root ./mlflow/artifacts \
              --host 0.0.0.0 --port 5000
```

## Usage

### 1. Starting the API Server

```bash
python api.py
```

The API will be available at `http://localhost:5001`

### 2. Upload a Model

#### Via Web UI
1. Navigate to `http://localhost:5001`
2. Enter model name
3. Select model file
4. Click "Upload Model"

#### Via REST API
```bash
curl -X POST http://localhost:5001/api/models/upload \
  -F "file=@path/to/model.pkl" \
  -F "model_name=my_classifier" \
  -F "user=data_scientist"
```

Response:
```json
{
  "success": true,
  "message": "Model uploaded successfully",
  "data": {
    "model_id": "550e8400-e29b-41d4-a716-446655440000",
    "model_name": "my_classifier",
    "version": "20240421_143022",
    "format": "pickle",
    "framework": "sklearn",
    "model_hash": {
      "md5": "5d41402abc4b2a76b9719d911017c592",
      "sha256": "2c26b46911185131006ba032f6b5d1f01ae14dcb180501ba6b61529ce2f37b9e",
      "file_size": 1024576
    },
    "metadata": {
      "model_type": "RandomForestClassifier",
      "has_predict": true,
      "has_predict_proba": true
    }
  }
}
```

### 3. Convert and Register Model

```bash
curl -X POST http://localhost:5001/api/models/{model_id}/convert-register \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my_classifier",
    "target_format": "onnx",
    "user": "data_scientist"
  }'
```

Response:
```json
{
  "success": true,
  "message": "Model converted and registered successfully",
  "data": {
    "model_id": "550e8400-e29b-41d4-a716-446655440000",
    "model_name": "my_classifier",
    "source_format": "pickle",
    "target_format": "onnx",
    "framework": "sklearn",
    "model_uri": "runs:/abc123/model",
    "metadata": {
      "framework": "sklearn",
      "format": "onnx",
      "num_inputs": 10,
      "num_outputs": 3
    }
  }
}
```

### 4. Make Predictions

#### Single Prediction
```bash
curl -X POST http://localhost:5001/api/inference/my_classifier_onnx/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[5.1, 3.5, 1.4, 0.2]],
    "stage": "Production"
  }'
```

#### Batch Prediction
```bash
curl -X POST http://localhost:5001/api/inference/my_classifier_onnx/batch-predict \
  -F "file=@data.csv" \
  -F "stage=Production"
```

### 5. Model Lifecycle Management

#### List Models
```bash
curl http://localhost:5001/api/models
```

#### Get Model Info
```bash
curl http://localhost:5001/api/models/my_classifier_onnx/info?stage=Production
```

#### Transition Model Stage
```bash
curl -X POST http://localhost:5001/api/models/my_classifier_onnx/transition-stage \
  -H "Content-Type: application/json" \
  -d '{
    "version": 1,
    "stage": "Production"
  }'
```

### 6. Audit & Lineage Tracking

#### Get Audit Trail
```bash
curl http://localhost:5001/api/models/{model_id}/audit-trail
```

Response:
```json
{
  "success": true,
  "count": 3,
  "data": [
    {
      "event_id": "event-1",
      "event_type": "model_upload",
      "model_id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp": "2024-04-21T14:30:22.000Z",
      "user": "data_scientist",
      "status": "success",
      "model_hash": {
        "md5": "5d41402abc4b2a76b9719d911017c592",
        "sha256": "2c26b46911185131006ba032f6b5d1f01ae14dcb180501ba6b61529ce2f37b9e",
        "file_size": 1024576
      },
      "environment_info": {
        "python_version": "3.11.2",
        "platform": "Linux",
        "framework_versions": {
          "sklearn": "1.3.0",
          "torch": "2.0.1"
        }
      }
    }
  ]
}
```

#### Get Lineage Trace
```bash
curl http://localhost:5001/api/models/{model_id}/lineage
```

Response:
```json
{
  "success": true,
  "count": 2,
  "data": [
    {
      "lineage_id": "lineage-1",
      "model_id": "550e8400-e29b-41d4-a716-446655440000",
      "parent_model_id": null,
      "source_format": "pickle",
      "target_format": "onnx",
      "conversion_method": "SklearnPickleToONNXConverter",
      "timestamp": "2024-04-21T14:32:15.000Z",
      "parameters": {},
      "data_loss_metrics": {
        "size_compression_ratio": 0.85,
        "feature_preservation_ratio": 1.0
      }
    }
  ]
}
```

## Python API Usage

### Direct Python Integration

```python
from orchestrator import mlops_orchestrator
from config import ModelFormat
from pathlib import Path

# 1. Upload model
success, msg, result = mlops_orchestrator.upload_model(
    source_file_path=Path("path/to/model.pkl"),
    model_name="iris_classifier",
    user="scientist@company.com",
    metadata={"accuracy": 0.95, "train_date": "2024-04-21"}
)

if success:
    model_id = result['model_id']
    print(f"Model uploaded: {model_id}")
    
    # 2. Convert and register
    success, msg, reg_result = mlops_orchestrator.convert_and_register(
        model_id=model_id,
        model_name="iris_classifier",
        target_format=ModelFormat.ONNX,
        user="scientist@company.com"
    )
    
    if success:
        print(f"Model registered: {reg_result['model_uri']}")
        
        # 3. Make predictions
        import pandas as pd
        data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
        
        success, predictions, msg = mlops_orchestrator.predict(
            model_name="iris_classifier_onnx",
            data=data,
            stage="Production"
        )
        
        if success:
            print(f"Predictions: {predictions}")
        
        # 4. Get audit trail
        success, msg, audit_trail = mlops_orchestrator.get_model_audit_trail(model_id)
        for entry in audit_trail:
            print(f"{entry['event_type']}: {entry['status']} at {entry['timestamp']}")
        
        # 5. Get lineage trace
        success, msg, lineage = mlops_orchestrator.get_model_lineage(model_id)
        for entry in lineage:
            print(f"{entry['source_format']} → {entry['target_format']} via {entry['conversion_method']}")
```

## Configuration

Edit `config.py` to customize:

```python
# Storage locations
StorageConfig.base_path = Path("./custom_storage")

# MLFlow settings
MLFlowConfig.tracking_uri = "http://your-mlflow-server:5000"

# API settings
APIConfig.port = 8080
APIConfig.max_upload_size_mb = 5000

# Audit settings
AuditConfig.enable_audit = True
```

## Model Storage Structure

```
model_store/
├── raw/
│   └── {model_id}/
│       └── {version}/
│           └── model_file
├── converted/
│   └── {model_id}/
│       └── {version}/
│           └── model.onnx
├── mlflow/
│   ├── mlflow.db
│   ├── registry.db
│   └── artifacts/
└── artifacts/
    ├── {model_id}_metadata.json
    └── predictions.csv
```

## Audit & Lineage Features

### Audit Trail Captures
- Model upload/download events
- Format conversion operations
- Model registration with MLFlow
- Inference/prediction events
- Model lifecycle transitions
- User and timestamp information
- File hashes (MD5, SHA256)
- Environment information (Python, frameworks)
- Data profiles and quality metrics

### Lineage Tracking Includes
- Complete conversion history
- Data loss metrics during conversion
- Validation results
- Parameter tracking
- Parent-child model relationships

## MLFlow Monitoring Dashboard

Access MLFlow UI at `http://localhost:5000` to:
- View model registry
- Track experiments
- Visualize metrics and parameters
- Compare model versions
- Manage model stages (Staging, Production, Archived)

## Error Handling

The system provides comprehensive error handling:

```json
{
  "error": "Model validation failed: Model missing predict method",
  "status_code": 400
}
```

Common error codes:
- `400`: Bad Request (invalid format, missing parameters)
- `404`: Not Found (model not in registry)
- `413`: Payload Too Large
- `500`: Internal Server Error

## Security Considerations

1. **File Upload Limits**: Configurable max upload size (default 1GB)
2. **Model Validation**: All models validated before storage
3. **Audit Trail**: Immutable audit logs for compliance
4. **Environment Capture**: Security context captured at model registration
5. **Access Control**: User tracking for all operations

## Performance Optimization

- Model caching for inference
- Batch prediction support
- Asynchronous format conversion
- Database indexing for audit queries
- Configurable batch sizes

## Testing

Run tests:
```bash
pytest tests/ -v
pytest tests/ --cov=. --cov-report=html
```

## Troubleshooting

### MLFlow Connection Issues
```bash
# Check MLFlow server
curl http://localhost:5000/health

# Restart MLFlow
pkill mlflow
mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db
```

### Model Conversion Failures
- Ensure all framework packages are installed
- Check framework versions match expected versions
- Verify model integrity with validation functions

### Permission Issues
- Check write permissions in storage directories
- Ensure MLFlow artifact directory is writable

## Contributing

To add support for a new model format:

1. Create handler in `model_handlers.py`
2. Create converter in `converters.py`
3. Add to format mappings in `config.py`
4. Add tests in `tests/`

## License

Proprietary - Takeda Pharmaceutical

## Support

For issues or questions:
- Check audit logs: `logs/audit.log`
- View MLFlow dashboard: `http://localhost:5000`
- Check API documentation: `http://localhost:5001`

## Version History

### v1.0.0 (2024-04-21)
- Initial release
- Support for 14+ model formats
- Full MLFlow integration
- Comprehensive audit & lineage tracking
- REST API with web UI
