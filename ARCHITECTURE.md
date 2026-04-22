"""
Architecture Documentation for Unified MLOps Layer

Comprehensive system architecture, data flows, and component interactions.
"""

def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def system_architecture():
    """System architecture overview"""
    print_header("SYSTEM ARCHITECTURE OVERVIEW")
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                     UNIFIED MLOps LAYER                             │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                         CLIENT INTERFACES                            │
├──────────────────────────────────────────────────────────────────────┤
│  • Web UI (http://localhost:5001)                                    │
│  • REST API (http://localhost:5001/api/...)                          │
│  • Python SDK (from orchestrator import mlops_orchestrator)          │
│  • CLI Commands (future enhancement)                                 │
└──────────────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────────────┐
│                        API LAYER (api.py)                            │
├──────────────────────────────────────────────────────────────────────┤
│  Flask Application                                                   │
│  • File upload handling (POST /api/models/upload)                   │
│  • Model conversion (POST /api/models/{id}/convert-register)        │
│  • Predictions (POST /api/inference/{model}/predict)               │
│  • Audit/Lineage (GET /api/models/{id}/audit-trail)               │
│  • Model management (GET/POST /api/models/...)                     │
└──────────────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION LAYER (orchestrator.py)               │
├──────────────────────────────────────────────────────────────────────┤
│  MLOpsOrchestrator - Central service coordinating:                  │
│  • Model upload workflows                                            │
│  • Format conversion & registration                                  │
│  • Prediction serving                                                │
│  • Audit/lineage tracking                                           │
└──────────────────────────────────────────────────────────────────────┘
         ↓                    ↓                    ↓
    ┌────────┐          ┌──────────┐         ┌──────────┐
    │ Handler│          │Converter │         │MLFlow    │
    │Registry│          │Registry  │         │Registrar │
    └────────┘          └──────────┘         └──────────┘
       ↓                    ↓                    ↓
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│Model Handlers│   │ Format Conv. │   │MLFlow Integration │
│- Pickle      │   │- Sklearn→ONNX│   │- Model Registry   │
│- Joblib      │   │- PyTorch→ONNX│   │- Stage Mgmt       │
│- XGBoost     │   │- TF→ONNX     │   │- Inference Serve  │
│- LightGBM    │   │- XGB→ONNX    │   │- Batch Predict    │
│- CatBoost    │   │- Others...   │   └──────────────────┘
│- PyTorch     │   └──────────────┘            ↓
│- TensorFlow  │                          ┌──────────────┐
│- Keras       │                          │MLFlow Server │
│- ONNX        │                          │(port 5000)   │
└──────────────┘                          └──────────────┘
       ↓
┌──────────────────────────────────────────────────────────────────────┐
│                    AUDIT & LINEAGE (audit.py)                        │
├──────────────────────────────────────────────────────────────────────┤
│  • File hashing (MD5, SHA256)                                        │
│  • Audit trail recording                                             │
│  • Lineage tracking                                                  │
│  • Environment capture                                               │
└──────────────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────────────┐
│                         STORAGE LAYER                                │
├──────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │
│  │ Raw Models     │  │ Converted      │  │ MLFlow         │         │
│  │ (uploaded)     │  │ Models (ONNX)  │  │ Artifacts      │         │
│  └────────────────┘  └────────────────┘  └────────────────┘         │
│                                                                      │
│  ┌──────────────────┐  ┌────────────────────────────┐               │
│  │ Audit Database   │  │ Configuration & Metadata   │               │
│  │ (SQLite/MySQL)   │  │ (JSON, YAML)               │               │
│  └──────────────────┘  └────────────────────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
""")


def data_flow_model_upload():
    """Data flow for model upload"""
    print_header("MODEL UPLOAD WORKFLOW")
    
    print("""
User uploads model file
         ↓
    [API Handler]
         ↓
    [File validation]
         ↓
    [Format detection] → handler_registry.detect_format()
         ↓
    [Model loading] → handler.load()
         ↓
    [Model validation] → handler.validate()
         ↓
    [Metadata extraction] → handler.get_metadata()
         ↓
    [File hashing] → AuditLogger.compute_file_hash()
         ↓
    [Storage] → model_store/raw/{model_id}/{version}/
         ↓
    [Audit logging] → audit_logger.log_audit_event()
         ↓
    [Response to user] with model_id
         ↓
User receives model_id for future operations
""")


def data_flow_conversion():
    """Data flow for model conversion"""
    print_header("MODEL CONVERSION & REGISTRATION WORKFLOW")
    
    print("""
User requests: POST /api/models/{model_id}/convert-register
         ↓
    [Load from raw storage]
         ↓
    [Detect source format]
         ↓
    [Get appropriate handler]
         ↓
    [Load source model]
         ↓
    [Get converter] → converter_registry.get_converter(src, tgt)
         ↓
    [Perform conversion] (e.g., sklearn → ONNX)
         ↓
    [Save converted model] → model_store/converted/{model_id}/{version}/
         ↓
    [Extract target metadata]
         ↓
    [Calculate data loss metrics]
         ↓
    [Register with MLFlow] → mlflow_registrar.register_model()
         ↓
    [Log audit event]
         ↓
    [Log lineage entry]
         ↓
    [Return model_uri]
         ↓
Model now available in MLFlow Registry
Available for deployment and inference
""")


def data_flow_inference():
    """Data flow for inference"""
    print_header("INFERENCE WORKFLOW")
    
    print("""
User sends prediction request
         ↓
    [API Handler] - POST /api/inference/{model_name}/predict
         ↓
    [Get from MLFlow registry] → mlflow_registrar.get_model()
         ↓
    [Load model] → mlflow.pyfunc.load_model()
         ↓
    [Validate input data]
         ↓
    [Run prediction] → model.predict()
         ↓
    [Format response]
         ↓
    [Log inference event] → audit_logger.log_audit_event()
         ↓
    [Return predictions to user]
         ↓
Predictions available with full audit trail
""")


def component_interactions():
    """Component interactions"""
    print_header("COMPONENT INTERACTION DIAGRAM")
    
    print("""
                         ┌────────────────┐
                         │   API Layer    │
                         │   (Flask)      │
                         └────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
            ┌───────▼────┐  ┌────▼────┐  ┌──▼──────┐
            │Upload      │  │Convert  │  │Inference│
            │Endpoint    │  │Endpoint │  │Endpoint │
            └────────────┘  └─────────┘  └─────────┘
                    │            │            │
                    └────────────┼────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Orchestrator        │
                    │  (Central Service)     │
                    └────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
    ┌───▼────┐        ┌──────────▼───┐        ┌──────────▼──────┐
    │Handler │        │Converter     │        │MLFlow Integration
    │Registry│        │Registry      │        │(Registrar +      │
    │        │        │              │        │ Inference)       │
    └────────┘        └────────────┬─┘        └──────────────────┘
        │                  │               │
        │              ┌────▼────┐        │
        └──────────┤  Audit &   │────────┘
                   │ Lineage    │
                   │ Tracker    │
                   └────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    ┌───▼────┐    ┌───────▼────┐    ┌──────▼──────┐
    │  Audit  │    │ MLFlow DB  │    │ Storage    │
    │  Logs   │    │ (Registry) │    │ (S3/Local) │
    └─────────┘    └────────────┘    └────────────┘
""")


def storage_structure():
    """File storage structure"""
    print_header("FILE STORAGE STRUCTURE")
    
    print("""
MLOPsEngine/
│
├── model_store/                          # Root storage directory
│   │
│   ├── raw/                               # Original uploaded models
│   │   ├── {model_id_1}/
│   │   │   ├── 20240421_143022/
│   │   │   │   ├── iris_classifier.pkl
│   │   │   │   └── metadata.json
│   │   │   └── 20240421_150530/
│   │   │       └── iris_classifier_v2.pkl
│   │   └── {model_id_2}/
│   │
│   ├── converted/                        # Format-converted models (typically ONNX)
│   │   ├── {model_id_1}/
│   │   │   └── 20240421_143022/
│   │   │       └── iris_classifier.onnx
│   │   └── {model_id_2}/
│   │
│   ├── mlflow/                           # MLFlow backend & artifacts
│   │   ├── mlflow.db                     # Main MLFlow database
│   │   ├── registry.db                   # Model registry database
│   │   └── artifacts/                    # Model artifacts
│   │       ├── 0/                        # Experiment 0
│   │       │   └── {run_id}/
│   │       │       ├── model/
│   │       │       └── artifacts/
│   │       └── 1/
│   │
│   └── artifacts/                        # Metadata and predictions
│       ├── {model_id}_metadata.json
│       ├── {model_id}_predictions.csv
│       └── {model_id}_audit_report.json
│
├── logs/                                 # Application logs
│   └── audit.log                         # Audit trail log file
│
├── config.py                             # Configuration management
├── model_handlers.py                     # Format-specific handlers
├── converters.py                         # Format converters
├── audit.py                              # Audit & lineage tracking
├── mlflow_integration.py                 # MLFlow integration
├── orchestrator.py                       # Central orchestrator
└── api.py                                # Flask API application

Database Schema:
───────────────

audit_events table:
  - event_id (PK)
  - event_type (model_upload, model_converted, model_registered, ...)
  - model_id (FK)
  - model_name
  - timestamp
  - user
  - status (success, failure)
  - details (JSON)
  - model_hash (MD5, SHA256, file_size)
  - environment_info (Python version, frameworks, ...)
  - data_profile (sample_count, features, statistics)

lineage table:
  - lineage_id (PK)
  - model_id (FK)
  - parent_model_id (FK) - parent model if converted from another
  - source_format
  - target_format
  - conversion_method
  - timestamp
  - parameters (JSON)
  - data_loss_metrics (size_ratio, feature_preservation, ...)
  - validation_results (JSON)
""")


def api_endpoints():
    """API endpoints reference"""
    print_header("REST API ENDPOINTS REFERENCE")
    
    print("""
BASE URL: http://localhost:5001

HEALTH & INFO
─────────────
GET  /health
  → Server health check

GET  /info
  → API information and supported formats


MODEL MANAGEMENT
────────────────
POST /api/models/upload
  → Upload new model
  Parameters: file, model_name, user, metadata

GET  /api/models
  → List all registered models

GET  /api/models/{model_name}/info
  → Get detailed model information
  Query: stage (Production|Staging|Archived)

POST /api/models/{model_id}/convert-register
  → Convert uploaded model and register with MLFlow
  Body: {model_name, target_format, user}


INFERENCE
─────────
POST /api/inference/{model_name}/predict
  → Single prediction or small batch
  Body: {data, stage}

POST /api/inference/{model_name}/batch-predict
  → Batch prediction on file
  File: CSV or Excel file


MODEL LIFECYCLE
───────────────
POST /api/models/{model_name}/transition-stage
  → Move model between stages
  Body: {version, stage}


AUDIT & TRACEABILITY
────────────────────
GET  /api/models/{model_id}/audit-trail
  → Get complete audit trail
  Returns: List of audit events chronologically

GET  /api/models/{model_id}/lineage
  → Get model lineage/transformation history
  Returns: List of lineage entries


ERROR RESPONSES
───────────────
400 Bad Request    - Invalid parameters or bad request
404 Not Found      - Resource not found
413 Payload Too Large - File exceeds max size
500 Internal Error - Server error

Success Response Format:
{
  "success": true,
  "message": "Operation completed",
  "data": { ... },
  "count": 0
}

Error Response Format:
{
  "error": "Error description"
}
""")


def security_architecture():
    """Security architecture"""
    print_header("SECURITY ARCHITECTURE")
    
    print("""
AUTHENTICATION & AUTHORIZATION
───────────────────────────────
┌──────────────────────────────────────────────────────┐
│  Client Request with JWT Token                       │
└──────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────┐
│  API Gateway / Load Balancer (TLS termination)       │
│  • Validate HTTPS/TLS certificate                   │
│  • Rate limiting                                     │
│  • Request validation                                │
└──────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────┐
│  Authentication Layer                                │
│  • Validate JWT token signature                      │
│  • Check token expiration                            │
│  • Extract user identity                             │
└──────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────┐
│  Authorization Layer (RBAC)                          │
│  • Check user roles                                  │
│  • Check resource permissions                        │
│  • Audit access attempt                              │
└──────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────┐
│  API Endpoint (if authorized)                        │
└──────────────────────────────────────────────────────┘


DATA PROTECTION
───────────────
At Rest:
  • Encryption: AES-256 (S3 SSE, database encryption)
  • Key Management: AWS KMS or HashiCorp Vault

In Transit:
  • TLS 1.3 for all API communication
  • Certificate pinning for critical connections

At Use:
  • Models loaded into trusted compute environment
  • No persistence of sensitive data in logs
  • Masking of PII in audit trails


AUDIT TRAIL IMMUTABILITY
─────────────────────────
  • Append-only database tables
  • Cryptographic hashing of audit entries
  • Regular integrity checks
  • Immutable storage (e.g., AWS S3 with Object Lock)


COMPLIANCE
──────────
GDPR:
  • User data deletion capability
  • Data export in machine-readable format
  • Privacy impact assessments

HIPAA:
  • Encryption of health data
  • Access controls and logging
  • Business associate agreements

SOC 2:
  • Regular security assessments
  • Access control reviews
  • Incident response procedures
""")


def scalability():
    """Scalability considerations"""
    print_header("SCALABILITY & PERFORMANCE")
    
    print("""
HORIZONTAL SCALING
───────────────────
┌─────────────────────────────┐
│   Load Balancer (Nginx)     │
│   (Round-robin or sticky)   │
└──────────────┬──────────────┘
       ┌───────┼───────┐
       │       │       │
   ┌───▼─┐ ┌──▼───┐ ┌──▼───┐
   │API 1│ │API 2 │ │API 3 │
   └─────┘ └──────┘ └──────┘
       │       │       │
       └───────┼───────┘
               │
      ┌────────▼─────────┐
      │  Shared Storage  │
      │  (S3, NFS, etc)  │
      └──────────────────┘


CACHING STRATEGY
────────────────
Level 1: Application Cache (Redis)
  • Model metadata cache
  • Recent prediction results cache
  • Configuration cache

Level 2: Model Cache
  • Load frequently used models into memory
  • Use model versioning for cache invalidation

Level 3: CDN Cache
  • Cache model artifacts
  • Cache API responses (with TTL)


DATABASE OPTIMIZATION
─────────────────────
Indexing Strategy:
  • Primary index on event_id, model_id
  • Composite index on (model_id, timestamp)
  • Index on event_type for filtering

Partitioning:
  • Partition audit_events by date
  • Archive old data to cold storage

Connection Pooling:
  • Min connections: 10, Max: 100
  • Connection timeout: 30s
  • Query timeout: 60s


PERFORMANCE TARGETS
────────────────────
API Response Times:
  • Health check: < 10ms
  • Model upload: < 5s (depends on size)
  • Prediction (small batch): < 100ms
  • Model list: < 500ms

Throughput:
  • 1000+ requests/second
  • 100+ concurrent predictions
  • 10Gbps network I/O

Resource Usage:
  • API Instance: 512MB - 2GB RAM per instance
  • MLFlow Instance: 4GB - 8GB RAM
  • Database: Variable based on audit volume
""")


def main():
    """Show architecture documentation"""
    print("\n" + "="*70)
    print("  UNIFIED MLOps LAYER - ARCHITECTURE DOCUMENTATION")
    print("="*70)
    
    system_architecture()
    data_flow_model_upload()
    data_flow_conversion()
    data_flow_inference()
    component_interactions()
    storage_structure()
    api_endpoints()
    security_architecture()
    scalability()
    
    print("\n" + "="*70)
    print("  ARCHITECTURE DOCUMENTATION COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
