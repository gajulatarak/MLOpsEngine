"""
Unified MLOps Orchestration Service
Central service that orchestrates model upload, conversion, registration, and inference.
"""

from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import uuid
from datetime import datetime
import shutil

from config import config, ModelFormat, ModelFramework
from model_handlers import handler_registry
from converters import conversion_service
from mlflow_integration import mlflow_registrar, mlflow_inference
from audit import audit_logger, AuditEntry, AuditEventType, AuditLogger


class MLOpsOrchestrator:
    """
    Central orchestration service for the unified MLOps layer.
    Handles: Upload, Detection, Validation, Conversion, Registration, Inference
    """
    
    def __init__(self):
        self.config = config
    
    def upload_model(
        self,
        source_file_path: Path,
        model_name: str,
        user: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Upload and process a model file.
        
        Returns:
            (success, message, result_dict)
        """
        try:
            model_id = str(uuid.uuid4())
            event_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            
            # Validate source file
            if not source_file_path.exists():
                return False, "Source file not found", None
            
            # Detect format
            detected_format = handler_registry.detect_format(source_file_path)
            if not detected_format:
                return False, "Unable to detect model format", None
            
            # Load and validate model
            handler = handler_registry.get_handler(source_file_path)
            model = handler.load(source_file_path)
            is_valid, validation_msg = handler.validate(model)
            
            if not is_valid:
                return False, f"Model validation failed: {validation_msg}", None
            
            # Extract metadata
            model_metadata = handler.get_metadata(model, source_file_path)
            
            # Create storage directory
            model_dir = self.config.storage.raw_models_path / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_dir = model_dir / version
            versioned_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy source file
            stored_path = versioned_dir / source_file_path.name
            shutil.copy2(source_file_path, stored_path)
            
            # Compute file hash
            model_hash = AuditLogger.compute_file_hash(stored_path)
            environment_info = AuditLogger.capture_environment_info()
            
            # Log audit event
            audit_entry = AuditEntry(
                event_id=event_id,
                event_type=AuditEventType.MODEL_UPLOAD,
                model_id=model_id,
                model_name=model_name,
                model_version=version,
                source_format=detected_format.value,
                target_format=detected_format.value,
                timestamp=timestamp,
                user=user,
                status="success",
                details={
                    'original_filename': source_file_path.name,
                    'stored_path': str(stored_path),
                    'model_metadata': model_metadata,
                    'validation_message': validation_msg,
                    'user_metadata': metadata or {},
                },
                model_hash=model_hash,
                environment_info=environment_info,
            )
            audit_logger.log_audit_event(audit_entry)
            
            result = {
                'model_id': model_id,
                'model_name': model_name,
                'version': version,
                'format': detected_format.value,
                'framework': config.format_framework_mapping[detected_format].value,
                'stored_path': str(stored_path),
                'model_hash': {
                    'md5': model_hash.md5,
                    'sha256': model_hash.sha256,
                    'file_size': model_hash.file_size,
                },
                'metadata': model_metadata,
            }
            
            return True, "Model uploaded successfully", result
        
        except Exception as e:
            return False, f"Upload failed: {str(e)}", None
    
    def convert_and_register(
        self,
        model_id: str,
        model_name: str,
        target_format: ModelFormat = ModelFormat.ONNX,
        user: str = "system"
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Convert model and register with MLFlow.
        
        Returns:
            (success, message, result_dict)
        """
        try:
            # Find the latest uploaded model
            model_dir = self.config.storage.raw_models_path / model_id
            if not model_dir.exists():
                return False, f"Model {model_id} not found", None
            
            # Get latest version
            versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
            if not versions:
                return False, "No model versions found", None
            
            latest_version = versions[-1]
            model_path = list((model_dir / latest_version).glob('*'))[0]
            
            # Detect source format
            source_format = handler_registry.detect_format(model_path)
            if not source_format:
                return False, "Unable to detect model format", None
            
            # Convert model
            success, conversion_msg, converted_path = conversion_service.convert_model(
                model_id=model_id,
                model_name=model_name,
                source_path=model_path,
                source_format=source_format,
                target_format=target_format,
                user=user
            )
            
            if not success:
                return False, conversion_msg, None
            
            # Get metadata
            handler = handler_registry.get_handler(converted_path)
            converted_model = handler.load(converted_path)
            metadata = handler.get_metadata(converted_model, converted_path)
            
            # Determine framework
            framework = config.format_framework_mapping.get(
                target_format,
                ModelFramework.CUSTOM
            )
            
            # Register with MLFlow
            reg_success, reg_msg, model_uri = mlflow_registrar.register_model(
                model_id=model_id,
                model_name=model_name,
                model_version=latest_version,
                model_path=converted_path,
                model_format=target_format,
                framework=framework,
                metadata=metadata,
                user=user
            )
            
            if not reg_success:
                return False, f"Registration failed: {reg_msg}", None
            
            result = {
                'model_id': model_id,
                'model_name': model_name,
                'source_format': source_format.value,
                'target_format': target_format.value,
                'converted_path': str(converted_path),
                'model_uri': model_uri,
                'framework': framework.value,
                'registration_message': reg_msg,
                'metadata': metadata,
            }
            
            return True, "Model converted and registered successfully", result
        
        except Exception as e:
            return False, f"Conversion and registration failed: {str(e)}", None
    
    def get_model_audit_trail(self, model_id: str) -> Tuple[bool, str, Optional[List[Dict]]]:
        """Get complete audit trail for a model"""
        try:
            trail = audit_logger.get_audit_trail(model_id)
            return True, "Audit trail retrieved", trail
        except Exception as e:
            return False, f"Failed to retrieve audit trail: {str(e)}", None
    
    def get_model_lineage(self, model_id: str) -> Tuple[bool, str, Optional[List[Dict]]]:
        """Get lineage/traceability information for a model"""
        try:
            lineage = audit_logger.get_lineage_trace(model_id)
            return True, "Lineage trace retrieved", lineage
        except Exception as e:
            return False, f"Failed to retrieve lineage: {str(e)}", None
    
    def list_models(self) -> Tuple[bool, str, Optional[List[Dict]]]:
        """List all uploaded and registered models"""
        try:
            # List from MLFlow registry
            models = mlflow_registrar.list_registered_models()
            return True, "Models listed", models
        except Exception as e:
            return False, f"Failed to list models: {str(e)}", None
    
    def predict(
        self,
        model_name: str,
        data: Any,
        stage: str = "Production"
    ) -> Tuple[bool, Any, str]:
        """Make prediction using a registered model"""
        return mlflow_inference.predict(model_name, data, stage)
    
    def batch_predict(
        self,
        model_name: str,
        data_path: Path,
        stage: str = "Production"
    ) -> Tuple[bool, Optional[Path], str]:
        """Batch prediction on data file"""
        return mlflow_inference.batch_predict(model_name, data_path, stage)
    
    def create_inference_endpoint(
        self,
        model_name: str,
        version: int,
        stage: str = "Production"
    ) -> Tuple[bool, str]:
        """Create inference endpoint for a model"""
        return mlflow_inference.create_inference_endpoint(model_name, version, stage)
    
    def transition_model_stage(
        self,
        model_name: str,
        version: int,
        stage: str
    ) -> Tuple[bool, str]:
        """Transition model to different stage"""
        return mlflow_registrar.transition_model_stage(model_name, version, stage)
    
    def get_model_info(self, model_name: str, stage: str = "Production") -> Optional[Dict]:
        """Get detailed model information from registry"""
        return mlflow_registrar.get_model_from_registry(model_name, stage)
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get list of supported model formats"""
        formats = {}
        for model_format in ModelFormat:
            formats[model_format.value] = {
                'extension': config.format_extensions[model_format],
                'framework': config.format_framework_mapping[model_format].value,
            }
        return formats


# Global instance
mlops_orchestrator = MLOpsOrchestrator()
