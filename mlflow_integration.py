"""
MLFlow Integration Module
Handles model registration, serving, and deployment with MLFlow.
"""

from pathlib import Path
from typing import Any, Dict, Optional, List
import uuid
from datetime import datetime
import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import json as json_lib

from config import config, ModelFormat, ModelFramework
from audit import audit_logger, AuditEntry, AuditEventType, AuditLogger


class MLFlowRegistrar:
    """Handles model registration with MLFlow"""
    
    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLFlow tracking"""
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_registry_uri(self.config.mlflow.registry_uri)
    
    def register_model(
        self,
        model_id: str,
        model_name: str,
        model_version: str,
        model_path: Path,
        model_format: ModelFormat,
        framework: ModelFramework,
        metadata: Dict[str, Any],
        user: str = "system"
    ) -> tuple[bool, str, Optional[str]]:
        """Register model with MLFlow"""
        
        try:
            event_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            
            # Set experiment
            experiment_name = self.config.mlflow.experiment_name
            mlflow.set_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
                # Log parameters
                mlflow.log_param("model_id", model_id)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("model_version", model_version)
                mlflow.log_param("framework", framework.value)
                mlflow.log_param("format", model_format.value)
                mlflow.log_param("user", user)
                mlflow.log_param("timestamp", timestamp)
                
                # Log metadata as artifacts
                metadata_file = Path(self.config.storage.artifacts_path) / f"{model_id}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json_lib.dump(metadata, f, indent=2)
                mlflow.log_artifact(str(metadata_file))
                
                # Log model based on format
                registered_name = None
                if model_format == ModelFormat.ONNX:
                    try:
                        import onnx as onnx_lib
                        onnx_model = onnx_lib.load(str(model_path))
                        mlflow.onnx.log_model(
                            onnx_model=onnx_model,
                            artifact_path="model",
                            registered_model_name=f"{model_name}_onnx"
                        )
                        registered_name = f"{model_name}_onnx"
                    except Exception:
                        mlflow.log_artifact(str(model_path), artifact_path="model")
                elif model_format in [ModelFormat.PICKLE, ModelFormat.JOBLIB]:
                    try:
                        sk_model = self._load_model(model_path, model_format)
                        if sk_model is not None:
                            mlflow.sklearn.log_model(
                                sk_model=sk_model,
                                artifact_path="model",
                                registered_model_name=f"{model_name}_sklearn"
                            )
                            registered_name = f"{model_name}_sklearn"
                        else:
                            mlflow.log_artifact(str(model_path), artifact_path="model")
                    except Exception:
                        mlflow.log_artifact(str(model_path), artifact_path="model")
                elif model_format in [ModelFormat.PYTORCH_PT, ModelFormat.PYTORCH_PTH]:
                    try:
                        import torch
                        pt_model = self._load_model(model_path, model_format)
                        mlflow.pytorch.log_model(
                            pytorch_model=pt_model,
                            artifact_path="model",
                            registered_model_name=f"{model_name}_pytorch"
                        )
                        registered_name = f"{model_name}_pytorch"
                    except Exception:
                        mlflow.log_artifact(str(model_path), artifact_path="model")
                elif model_format in [ModelFormat.TENSORFLOW_SAVEDMODEL, ModelFormat.KERAS_H5]:
                    try:
                        mlflow.tensorflow.log_model(
                            tf_saved_model_dir=str(model_path),
                            artifact_path="model",
                            registered_model_name=f"{model_name}_tf"
                        )
                        registered_name = f"{model_name}_tf"
                    except Exception:
                        mlflow.log_artifact(str(model_path), artifact_path="model")
                else:
                    # Generic model logging for all other formats
                    mlflow.log_artifact(str(model_path), artifact_path="model")
                
                # Log tags for traceability
                mlflow.set_tags({
                    "model_format": model_format.value,
                    "framework": framework.value,
                    "registered_by": user,
                    "model_id": model_id,
                })
                
                run_id = run.info.run_id
            
            # Get model URI
            model_uri = f"runs:/{run_id}/model"
            
            # Compute file hash for audit
            model_hash = AuditLogger.compute_file_hash(model_path)
            environment_info = AuditLogger.capture_environment_info()
            
            # Log audit event
            audit_entry = AuditEntry(
                event_id=event_id,
                event_type=AuditEventType.MODEL_REGISTERED,
                model_id=model_id,
                model_name=model_name,
                model_version=model_version,
                source_format=model_format.value,
                target_format=model_format.value,
                timestamp=timestamp,
                user=user,
                status="success",
                details={
                    'model_uri': model_uri,
                    'run_id': run_id,
                    'experiment_name': experiment_name,
                    'metadata': metadata,
                },
                model_hash=model_hash,
                environment_info=environment_info,
            )
            audit_logger.log_audit_event(audit_entry)
            
            return True, f"Model registered successfully. Run ID: {run_id}", model_uri
        
        except Exception as e:
            return False, f"Model registration failed: {str(e)}", None
    
    def _load_model(self, model_path: Path, model_format: ModelFormat) -> Any:
        """Load model based on format"""
        from model_handlers import handler_registry
        
        handler = handler_registry.get_handler(model_path)
        if handler:
            return handler.load(model_path)
        return None
    
    def get_model_from_registry(self, model_name: str, stage: str = "Production") -> Optional[Dict[str, Any]]:
        """Get model from MLFlow registry"""
        try:
            client = mlflow.tracking.MlflowClient()
            try:
                versions = client.get_latest_versions(model_name, stages=[stage])
                if versions:
                    model_version = versions[0]
                    return {
                        'name': model_name,
                        'version': model_version.version,
                        'stage': model_version.current_stage,
                        'source': model_version.source,
                        'status': model_version.status,
                    }
                return None
            except Exception:
                return None
        except Exception as e:
            print(f"Error retrieving model: {str(e)}")
            return None
    
    def transition_model_stage(
        self,
        model_name: str,
        version: int,
        stage: str
    ) -> tuple[bool, str]:
        """Transition model to different stage (Staging, Production, Archived)"""
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(model_name, version, stage)
            
            # Log audit event
            event_id = str(uuid.uuid4())
            audit_entry = AuditEntry(
                event_id=event_id,
                event_type=AuditEventType.MODEL_UPDATED,
                model_id=model_name,
                model_name=model_name,
                model_version=str(version),
                source_format="mlflow",
                target_format="mlflow",
                timestamp=datetime.utcnow().isoformat(),
                user="admin",
                status="success",
                details={'stage_transition': stage},
            )
            audit_logger.log_audit_event(audit_entry)
            
            return True, f"Model transitioned to {stage}"
        except Exception as e:
            return False, f"Stage transition failed: {str(e)}"
    
    def list_registered_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        try:
            client = mlflow.tracking.MlflowClient()
            models = client.search_registered_models()
            
            result = []
            for model in models:
                result.append({
                    'name': model.name,
                    'creation_timestamp': model.creation_timestamp,
                    'last_updated_timestamp': model.last_updated_timestamp,
                    'latest_versions': len(model.latest_versions),
                })
            
            return result
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            return []


class MLFlowInferenceService:
    """Service for model inference using MLFlow"""
    
    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.registrar = MLFlowRegistrar(config_obj)
    
    def create_inference_endpoint(
        self,
        model_name: str,
        version: int,
        stage: str = "Production"
    ) -> tuple[bool, str]:
        """Create inference endpoint for model"""
        try:
            # Verify model exists in registry
            model_info = self.registrar.get_model_from_registry(model_name, stage)
            if not model_info:
                return False, f"Model {model_name} not found in registry"
            
            # Log endpoint creation
            event_id = str(uuid.uuid4())
            audit_entry = AuditEntry(
                event_id=event_id,
                event_type=AuditEventType.MODEL_DEPLOYED,
                model_id=model_name,
                model_name=model_name,
                model_version=str(version),
                source_format="mlflow",
                target_format="inference",
                timestamp=datetime.utcnow().isoformat(),
                user="admin",
                status="success",
                details={
                    'stage': stage,
                    'endpoint_url': f"http://{self.config.inference.host}:{self.config.inference.port}/predict/{model_name}"
                },
            )
            audit_logger.log_audit_event(audit_entry)
            
            return True, f"Inference endpoint created for {model_name}"
        except Exception as e:
            return False, f"Endpoint creation failed: {str(e)}"
    
    def predict(
        self,
        model_name: str,
        data: Any,
        stage: str = "Production"
    ) -> tuple[bool, Any, str]:
        """Make prediction using registered model"""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Get model info
            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                return False, None, f"Model {model_name} not found in stage {stage}"
            model_version = versions[0]
            
            # Load model
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Make prediction
            predictions = model.predict(data)
            
            # Log inference event
            event_id = str(uuid.uuid4())
            audit_entry = AuditEntry(
                event_id=event_id,
                event_type=AuditEventType.MODEL_INFERENCE,
                model_id=model_name,
                model_name=model_name,
                model_version=model_version.version,
                source_format="data",
                target_format="predictions",
                timestamp=datetime.utcnow().isoformat(),
                user="inference_service",
                status="success",
                details={
                    'input_shape': str(data.shape) if hasattr(data, 'shape') else "unknown",
                    'output_shape': str(predictions.shape) if hasattr(predictions, 'shape') else "unknown",
                    'stage': stage,
                },
            )
            audit_logger.log_audit_event(audit_entry)
            
            return True, predictions, "Prediction successful"
        except Exception as e:
            return False, None, f"Prediction failed: {str(e)}"
    
    def batch_predict(
        self,
        model_name: str,
        data_path: Path,
        stage: str = "Production"
    ) -> tuple[bool, Optional[Path], str]:
        """Batch prediction on data file"""
        try:
            import pandas as pd
            
            # Load data
            if data_path.suffix == '.csv':
                data = pd.read_csv(data_path)
            elif data_path.suffix in ['.xlsx', '.xls']:
                data = pd.read_excel(data_path)
            else:
                return False, None, "Unsupported data format"
            
            # Make predictions
            success, predictions, msg = self.predict(model_name, data, stage)
            if not success:
                return False, None, msg
            
            # Save predictions
            output_path = self.config.storage.artifacts_path / f"{model_name}_predictions.csv"
            predictions.to_csv(output_path, index=False)
            
            return True, output_path, "Batch predictions completed"
        except Exception as e:
            return False, None, f"Batch prediction failed: {str(e)}"


# Global instances
mlflow_registrar = MLFlowRegistrar()
mlflow_inference = MLFlowInferenceService()
