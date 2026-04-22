"""
Model Format Converters
Converts models between different formats to ONNX (standard interchange format) and MLFlow compatible format.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import uuid
from datetime import datetime
import json as json_lib

from config import ModelFormat, config
from model_handlers import handler_registry
from audit import audit_logger, AuditEntry, AuditEventType, LineageEntry, AuditLogger


class ModelConverter(ABC):
    """Base class for model format converters"""
    
    @abstractmethod
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """Check if conversion is supported"""
        pass
    
    @abstractmethod
    def convert(self, source_path: Path, target_path: Path, model: Any) -> Tuple[bool, str]:
        """Convert model from source to target format"""
        pass


class PyTorchToONNXConverter(ModelConverter):
    """Converts PyTorch models to ONNX"""
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """Check if PyTorch to ONNX conversion is supported"""
        return (source_format in [ModelFormat.PYTORCH_PT, ModelFormat.PYTORCH_PTH] and 
                target_format == ModelFormat.ONNX)
    
    def convert(self, source_path: Path, target_path: Path, model: Any) -> Tuple[bool, str]:
        """Convert PyTorch to ONNX"""
        try:
            import torch
            import onnx
            
            # Ensure model is in eval mode
            if hasattr(model, 'eval'):
                model.eval()
            
            # Create dummy input based on model
            if hasattr(model, 'input_shape'):
                dummy_input = torch.randn(*model.input_shape)
            else:
                # Default batch size 1
                dummy_input = torch.randn(1, 3, 224, 224)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(target_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                verbose=False
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(str(target_path))
            onnx.checker.check_model(onnx_model)
            
            return True, "Successfully converted PyTorch to ONNX"
        except Exception as e:
            return False, f"PyTorch to ONNX conversion failed: {str(e)}"


class TensorFlowToONNXConverter(ModelConverter):
    """Converts TensorFlow models to ONNX"""
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """Check if TensorFlow to ONNX conversion is supported"""
        return (source_format in [ModelFormat.TENSORFLOW_SAVEDMODEL, ModelFormat.KERAS_H5] and 
                target_format == ModelFormat.ONNX)
    
    def convert(self, source_path: Path, target_path: Path, model: Any) -> Tuple[bool, str]:
        """Convert TensorFlow to ONNX"""
        try:
            import tf2onnx
            import tensorflow as tf
            
            # Convert to ONNX
            spec = (tf.TensorSpec(shape=[None] + list(model.input_shape[1:]), dtype=tf.float32, name='input'),)
            
            output_path, _ = tf2onnx.convert.from_keras(model, output_path=str(target_path))
            
            return True, "Successfully converted TensorFlow to ONNX"
        except Exception as e:
            return False, f"TensorFlow to ONNX conversion failed: {str(e)}"


class SklearnPickleToONNXConverter(ModelConverter):
    """Converts scikit-learn Pickle/Joblib models to ONNX"""
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """Check if sklearn to ONNX conversion is supported"""
        return (source_format in [ModelFormat.PICKLE, ModelFormat.JOBLIB] and 
                target_format == ModelFormat.ONNX)
    
    def convert(self, source_path: Path, target_path: Path, model: Any) -> Tuple[bool, str]:
        """Convert sklearn model to ONNX"""
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            import onnx
            
            # Initialize types
            initial_type = [('float_input', FloatTensorType([None, 4]))]
            
            # Convert to ONNX
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            # Save ONNX model
            with open(target_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            # Verify
            onnx.checker.check_model(onnx_model)
            
            return True, "Successfully converted sklearn model to ONNX"
        except Exception as e:
            return False, f"sklearn to ONNX conversion failed: {str(e)}"


class XGBoostToONNXConverter(ModelConverter):
    """Converts XGBoost models to ONNX"""
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """Check if XGBoost to ONNX conversion is supported"""
        return (source_format == ModelFormat.XGBOOST and target_format == ModelFormat.ONNX)
    
    def convert(self, source_path: Path, target_path: Path, model: Any) -> Tuple[bool, str]:
        """Convert XGBoost to ONNX"""
        try:
            from onnxmltools import convert_xgboost
            import onnx
            
            # Convert to ONNX
            onnx_model = convert_xgboost(model)
            
            # Save ONNX model
            onnx.save(onnx_model, str(target_path))
            
            # Verify
            onnx.checker.check_model(onnx_model)
            
            return True, "Successfully converted XGBoost to ONNX"
        except Exception as e:
            return False, f"XGBoost to ONNX conversion failed: {str(e)}"


class LightGBMToONNXConverter(ModelConverter):
    """Converts LightGBM models to ONNX"""
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """Check if LightGBM to ONNX conversion is supported"""
        return (source_format == ModelFormat.LIGHTGBM and target_format == ModelFormat.ONNX)
    
    def convert(self, source_path: Path, target_path: Path, model: Any) -> Tuple[bool, str]:
        """Convert LightGBM to ONNX"""
        try:
            from onnxmltools import convert_lightgbm
            import onnx
            
            # Convert to ONNX
            onnx_model = convert_lightgbm(model)
            
            # Save ONNX model
            onnx.save(onnx_model, str(target_path))
            
            # Verify
            onnx.checker.check_model(onnx_model)
            
            return True, "Successfully converted LightGBM to ONNX"
        except Exception as e:
            return False, f"LightGBM to ONNX conversion failed: {str(e)}"


class CatBoostToONNXConverter(ModelConverter):
    """Converts CatBoost models to ONNX"""
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """Check if CatBoost to ONNX conversion is supported"""
        return (source_format == ModelFormat.CATBOOST and target_format == ModelFormat.ONNX)
    
    def convert(self, source_path: Path, target_path: Path, model: Any) -> Tuple[bool, str]:
        """Convert CatBoost to ONNX"""
        try:
            from onnxmltools import convert_catboost
            import onnx
            
            # Convert to ONNX
            onnx_model = convert_catboost(model)
            
            # Save ONNX model
            onnx.save(onnx_model, str(target_path))
            
            # Verify
            onnx.checker.check_model(onnx_model)
            
            return True, "Successfully converted CatBoost to ONNX"
        except Exception as e:
            return False, f"CatBoost to ONNX conversion failed: {str(e)}"


class DirectCopyConverter(ModelConverter):
    """Direct copy converter for formats already compatible with MLFlow"""
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """Check if direct copy is needed"""
        # ONNX to ONNX, etc.
        return source_format == target_format
    
    def convert(self, source_path: Path, target_path: Path, model: Any) -> Tuple[bool, str]:
        """Direct copy conversion"""
        try:
            import shutil
            shutil.copy2(source_path, target_path)
            return True, "Successfully copied model file"
        except Exception as e:
            return False, f"Copy conversion failed: {str(e)}"


class ModelConverterRegistry:
    """Registry for model format converters"""
    
    def __init__(self):
        self.converters = [
            PyTorchToONNXConverter(),
            TensorFlowToONNXConverter(),
            SklearnPickleToONNXConverter(),
            XGBoostToONNXConverter(),
            LightGBMToONNXConverter(),
            CatBoostToONNXConverter(),
            DirectCopyConverter(),
        ]
    
    def get_converter(self, source_format: ModelFormat, target_format: ModelFormat) -> Optional[ModelConverter]:
        """Get converter for source to target format"""
        for converter in self.converters:
            if converter.can_convert(source_format, target_format):
                return converter
        return None
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """Check if conversion is supported"""
        return self.get_converter(source_format, target_format) is not None


class ConversionService:
    """Service for converting and processing models"""
    
    def __init__(self, converter_registry: ModelConverterRegistry = None):
        self.converter_registry = converter_registry or ModelConverterRegistry()
    
    def convert_model(
        self,
        model_id: str,
        model_name: str,
        source_path: Path,
        source_format: ModelFormat,
        target_format: ModelFormat = ModelFormat.ONNX,
        user: str = "system"
    ) -> Tuple[bool, str, Optional[Path]]:
        """Convert model from source format to target format with full audit trail"""
        
        try:
            event_id = str(uuid.uuid4())
            lineage_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            
            # Load source model
            handler = handler_registry.get_handler(source_path)
            if not handler:
                return False, f"No handler found for {source_format.value}", None
            
            model = handler.load(source_path)
            
            # Validate source model
            is_valid, validation_msg = handler.validate(model)
            if not is_valid:
                return False, f"Model validation failed: {validation_msg}", None
            
            # Get source metadata
            source_metadata = handler.get_metadata(model, source_path)
            
            # Get converter
            converter = self.converter_registry.get_converter(source_format, target_format)
            if not converter:
                return False, f"No converter available from {source_format.value} to {target_format.value}", None
            
            # Create target path
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_dir = config.storage.converted_models_path / model_id / version
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"{model_name}{config.format_extensions[target_format]}"
            
            # Perform conversion
            success, conversion_msg = converter.convert(source_path, target_path, model)
            
            if not success:
                return False, f"Conversion failed: {conversion_msg}", None
            
            # Load converted model for metadata
            target_handler = handler_registry.get_handler(target_path)
            if target_handler:
                converted_model = target_handler.load(target_path)
                target_metadata = target_handler.get_metadata(converted_model, target_path)
            else:
                target_metadata = {}
            
            # Compute file hashes
            source_hash = AuditLogger.compute_file_hash(source_path)
            target_hash = AuditLogger.compute_file_hash(target_path)
            environment_info = AuditLogger.capture_environment_info()
            
            # Calculate data loss metrics
            data_loss_metrics = self._calculate_data_loss_metrics(
                source_metadata, target_metadata
            )
            
            # Log audit event
            audit_entry = AuditEntry(
                event_id=event_id,
                event_type=AuditEventType.MODEL_CONVERTED,
                model_id=model_id,
                model_name=model_name,
                model_version=version,
                source_format=source_format.value,
                target_format=target_format.value,
                timestamp=timestamp,
                user=user,
                status="success",
                details={
                    'source_metadata': source_metadata,
                    'target_metadata': target_metadata,
                    'conversion_message': conversion_msg,
                },
                model_hash=target_hash,
                environment_info=environment_info,
            )
            audit_logger.log_audit_event(audit_entry)
            
            # Log lineage
            lineage_entry = LineageEntry(
                lineage_id=lineage_id,
                model_id=model_id,
                parent_model_id=None,
                parent_version=None,
                source_format=source_format,
                target_format=target_format,
                conversion_method=converter.__class__.__name__,
                timestamp=timestamp,
                parameters={},
                data_loss_metrics=data_loss_metrics,
                validation_results={'source_validation': validation_msg}
            )
            audit_logger.log_lineage(lineage_entry)
            
            return True, "Conversion completed successfully", target_path
        
        except Exception as e:
            return False, f"Conversion error: {str(e)}", None
    
    @staticmethod
    def _calculate_data_loss_metrics(source_meta: Dict, target_meta: Dict) -> Dict[str, float]:
        """Calculate data loss metrics during conversion"""
        metrics = {}
        
        # Compare file sizes
        if 'file_size' in source_meta and 'file_size' in target_meta:
            size_ratio = target_meta['file_size'] / source_meta['file_size']
            metrics['size_compression_ratio'] = size_ratio
        
        # Compare features
        if 'num_features' in source_meta and 'num_features' in target_meta:
            if source_meta['num_features'] == target_meta['num_features']:
                metrics['feature_preservation_ratio'] = 1.0
            else:
                metrics['feature_preservation_ratio'] = (
                    target_meta['num_features'] / source_meta['num_features']
                )
        
        return metrics


# Global instances
converter_registry = ModelConverterRegistry()
conversion_service = ConversionService(converter_registry)
