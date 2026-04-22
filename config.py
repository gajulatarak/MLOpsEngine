"""
MLOps Unified Layer Configuration
Manages all configuration for model formats, MLFlow, audit tracking, and storage.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum


class ModelFormat(Enum):
    """Supported model formats"""
    # Deep Learning Formats
    ONNX = "onnx"
    TENSORFLOW_SAVEDMODEL = "tensorflow_savedmodel"
    KERAS_H5 = "keras_h5"
    PYTORCH_PT = "pytorch_pt"
    PYTORCH_PTH = "pytorch_pth"
    
    # Classical ML Formats
    PICKLE = "pickle"
    JOBLIB = "joblib"
    PMML = "pmml"
    
    # Gradient Boosting / Tree Models
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    
    # Export Formats
    TORCHSCRIPT = "torchscript"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"


class ModelFramework(Enum):
    """Supported ML frameworks"""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    ONNX = "onnx"
    KERAS = "keras"
    CUSTOM = "custom"


@dataclass
class StorageConfig:
    """Storage configuration"""
    # Base directories
    base_path: Path = Path(__file__).parent / "model_store"
    raw_models_path: Path = None
    converted_models_path: Path = None
    mlflow_models_path: Path = None
    artifacts_path: Path = None
    
    # S3 configuration (optional for cloud deployment)
    use_s3: bool = False
    s3_bucket: str = "mlops-models"
    s3_prefix: str = "models"
    
    def __post_init__(self):
        if self.raw_models_path is None:
            self.raw_models_path = self.base_path / "raw"
        if self.converted_models_path is None:
            self.converted_models_path = self.base_path / "converted"
        if self.mlflow_models_path is None:
            self.mlflow_models_path = self.base_path / "mlflow"
        if self.artifacts_path is None:
            self.artifacts_path = self.base_path / "artifacts"
        
        # Create directories if they don't exist
        for path in [self.raw_models_path, self.converted_models_path, 
                     self.mlflow_models_path, self.artifacts_path]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class MLFlowConfig:
    """MLFlow configuration"""
    tracking_uri: str = "http://127.0.0.1:5000"
    backend_store_uri: str = "sqlite:///model_store/mlflow/mlflow.db"
    default_artifact_root: str = "./model_store/mlflow/artifacts"
    registry_uri: str = "sqlite:///model_store/mlflow/mlflow.db"
    experiment_name: str = "unified_mlops"
    
    def __post_init__(self):
        # Create artifact root if it doesn't exist
        Path(self.default_artifact_root).mkdir(parents=True, exist_ok=True)


@dataclass
class AuditConfig:
    """Audit and lineage tracking configuration"""
    enable_audit: bool = True
    audit_db_path: str = "sqlite:///model_store/mlflow/audit.db"
    log_file_path: Path = Path(__file__).parent / "logs" / "audit.log"
    capture_model_hash: bool = True
    capture_environment: bool = True
    capture_data_profile: bool = True
    
    def __post_init__(self):
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class InferenceConfig:
    """Inference service configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 32
    inference_timeout: int = 30
    enable_caching: bool = True
    cache_size_mb: int = 512


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 5001
    debug: bool = False
    max_upload_size_mb: int = 1000
    allowed_origins: List[str] = None
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["http://localhost:3000", "http://localhost:8080"]


class MLOpsConfig:
    """Master configuration class"""
    
    def __init__(self):
        self.storage = StorageConfig()
        self.mlflow = MLFlowConfig()
        self.audit = AuditConfig()
        self.inference = InferenceConfig()
        self.api = APIConfig()
        
        # Format-to-extension mapping
        self.format_extensions: Dict[ModelFormat, str] = {
            ModelFormat.ONNX: ".onnx",
            ModelFormat.TENSORFLOW_SAVEDMODEL: ".pb",
            ModelFormat.KERAS_H5: ".h5",
            ModelFormat.PYTORCH_PT: ".pt",
            ModelFormat.PYTORCH_PTH: ".pth",
            ModelFormat.PICKLE: ".pkl",
            ModelFormat.JOBLIB: ".joblib",
            ModelFormat.PMML: ".pmml",
            ModelFormat.XGBOOST: ".xgb",
            ModelFormat.LIGHTGBM: ".lgb",
            ModelFormat.CATBOOST: ".cb",
            ModelFormat.TORCHSCRIPT: ".pt",
            ModelFormat.TENSORRT: ".engine",
            ModelFormat.OPENVINO: ".xml",
        }
        
        # Format-to-framework mapping
        self.format_framework_mapping: Dict[ModelFormat, ModelFramework] = {
            ModelFormat.ONNX: ModelFramework.ONNX,
            ModelFormat.TENSORFLOW_SAVEDMODEL: ModelFramework.TENSORFLOW,
            ModelFormat.KERAS_H5: ModelFramework.KERAS,
            ModelFormat.PYTORCH_PT: ModelFramework.PYTORCH,
            ModelFormat.PYTORCH_PTH: ModelFramework.PYTORCH,
            ModelFormat.PICKLE: ModelFramework.SKLEARN,
            ModelFormat.JOBLIB: ModelFramework.SKLEARN,
            ModelFormat.PMML: ModelFramework.CUSTOM,
            ModelFormat.XGBOOST: ModelFramework.XGBOOST,
            ModelFormat.LIGHTGBM: ModelFramework.LIGHTGBM,
            ModelFormat.CATBOOST: ModelFramework.CATBOOST,
            ModelFormat.TORCHSCRIPT: ModelFramework.PYTORCH,
            ModelFormat.TENSORRT: ModelFramework.CUSTOM,
            ModelFormat.OPENVINO: ModelFramework.CUSTOM,
        }
    
    @staticmethod
    def get_instance() -> "MLOpsConfig":
        """Get singleton instance of MLOpsConfig"""
        if not hasattr(MLOpsConfig, "_instance"):
            MLOpsConfig._instance = MLOpsConfig()
        return MLOpsConfig._instance


# Module-level convenience
config = MLOpsConfig.get_instance()
