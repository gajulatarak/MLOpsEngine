"""
Model Format Handlers
Detects and handles different model formats.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pickle
import json
import numpy as np

from config import ModelFormat, ModelFramework, config


class ModelHandler(ABC):
    """Base class for model format handlers"""
    
    @abstractmethod
    def detect(self, file_path: Path) -> bool:
        """Check if file is this format"""
        pass
    
    @abstractmethod
    def load(self, file_path: Path) -> Any:
        """Load model from file"""
        pass
    
    @abstractmethod
    def validate(self, model: Any) -> Tuple[bool, str]:
        """Validate model integrity"""
        pass
    
    @abstractmethod
    def get_metadata(self, model: Any, file_path: Path) -> Dict[str, Any]:
        """Extract model metadata"""
        pass


class PickleHandler(ModelHandler):
    """Handles Pickle (.pkl) format"""
    
    def detect(self, file_path: Path) -> bool:
        """Detect if file is pickle format"""
        return file_path.suffix.lower() in ['.pkl', '.pickle']
    
    def load(self, file_path: Path) -> Any:
        """Load pickle model"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def validate(self, model: Any) -> Tuple[bool, str]:
        """Validate pickle model"""
        try:
            # Basic validation - check if model has common sklearn methods
            if hasattr(model, 'predict'):
                return True, "Valid sklearn model"
            else:
                return True, "Pickle file loaded but may not be ML model"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"
    
    def get_metadata(self, model: Any, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from pickle model"""
        metadata = {
            'framework': 'sklearn',
            'format': 'pickle',
            'file_size': file_path.stat().st_size,
            'has_predict': hasattr(model, 'predict'),
            'has_predict_proba': hasattr(model, 'predict_proba'),
            'has_fit': hasattr(model, 'fit'),
        }
        
        # Extract model type if sklearn
        if hasattr(model, '__class__'):
            metadata['model_type'] = model.__class__.__name__
        
        return metadata


class JoblibHandler(ModelHandler):
    """Handles Joblib format"""
    
    def detect(self, file_path: Path) -> bool:
        """Detect if file is joblib format"""
        return file_path.suffix.lower() == '.joblib'
    
    def load(self, file_path: Path) -> Any:
        """Load joblib model"""
        try:
            import joblib
            return joblib.load(file_path)
        except ImportError:
            raise ImportError("joblib not installed. Install with: pip install joblib")
    
    def validate(self, model: Any) -> Tuple[bool, str]:
        """Validate joblib model"""
        try:
            if hasattr(model, 'predict'):
                return True, "Valid joblib model"
            else:
                return True, "Joblib file loaded but may not be ML model"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"
    
    def get_metadata(self, model: Any, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from joblib model"""
        metadata = {
            'framework': 'sklearn',
            'format': 'joblib',
            'file_size': file_path.stat().st_size,
            'has_predict': hasattr(model, 'predict'),
            'has_predict_proba': hasattr(model, 'predict_proba'),
            'has_fit': hasattr(model, 'fit'),
        }
        
        if hasattr(model, '__class__'):
            metadata['model_type'] = model.__class__.__name__
        
        return metadata


class XGBoostHandler(ModelHandler):
    """Handles XGBoost binary format"""
    
    def detect(self, file_path: Path) -> bool:
        """Detect if file is XGBoost format"""
        return file_path.suffix.lower() in ['.xgb', '.xgboost', '.bst', '.ubj', '.json']
    
    def load(self, file_path: Path) -> Any:
        """Load XGBoost model"""
        try:
            import xgboost as xgb
            return xgb.Booster(model_file=str(file_path))
        except ImportError:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")
    
    def validate(self, model: Any) -> Tuple[bool, str]:
        """Validate XGBoost model"""
        try:
            if hasattr(model, 'predict'):
                return True, "Valid XGBoost model"
            else:
                return False, "XGBoost object missing predict method"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"
    
    def get_metadata(self, model: Any, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from XGBoost model"""
        metadata = {
            'framework': 'xgboost',
            'format': 'xgboost',
            'file_size': file_path.stat().st_size,
            'num_features': model.num_features(),
        }
        
        try:
            # Get model config
            config_dict = json.loads(model.get_config())
            metadata['config'] = config_dict
        except:
            pass
        
        return metadata


class LightGBMHandler(ModelHandler):
    """Handles LightGBM format"""
    
    def detect(self, file_path: Path) -> bool:
        """Detect if file is LightGBM format"""
        return file_path.suffix.lower() in ['.lgb', '.model']
    
    def load(self, file_path: Path) -> Any:
        """Load LightGBM model"""
        try:
            import lightgbm as lgb
            return lgb.Booster(model_file=str(file_path))
        except ImportError:
            raise ImportError("lightgbm not installed. Install with: pip install lightgbm")
    
    def validate(self, model: Any) -> Tuple[bool, str]:
        """Validate LightGBM model"""
        try:
            if hasattr(model, 'predict'):
                return True, "Valid LightGBM model"
            else:
                return False, "LightGBM object missing predict method"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"
    
    def get_metadata(self, model: Any, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from LightGBM model"""
        metadata = {
            'framework': 'lightgbm',
            'format': 'lightgbm',
            'file_size': file_path.stat().st_size,
            'num_features': model.num_feature(),
        }
        
        return metadata


class CatBoostHandler(ModelHandler):
    """Handles CatBoost format"""
    
    def detect(self, file_path: Path) -> bool:
        """Detect if file is CatBoost format"""
        return file_path.suffix.lower() in ['.cb', '.cbm', '.model']
    
    def load(self, file_path: Path) -> Any:
        """Load CatBoost model"""
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            # Try loading as either classifier or regressor
            try:
                return CatBoostClassifier().load_model(str(file_path))
            except:
                return CatBoostRegressor().load_model(str(file_path))
        except ImportError:
            raise ImportError("catboost not installed. Install with: pip install catboost")
    
    def validate(self, model: Any) -> Tuple[bool, str]:
        """Validate CatBoost model"""
        try:
            if hasattr(model, 'predict'):
                return True, "Valid CatBoost model"
            else:
                return False, "CatBoost object missing predict method"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"
    
    def get_metadata(self, model: Any, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from CatBoost model"""
        metadata = {
            'framework': 'catboost',
            'format': 'catboost',
            'file_size': file_path.stat().st_size,
        }
        
        return metadata


class PyTorchHandler(ModelHandler):
    """Handles PyTorch (.pt/.pth) format"""
    
    def detect(self, file_path: Path) -> bool:
        """Detect if file is PyTorch format"""
        return file_path.suffix.lower() in ['.pt', '.pth']
    
    def load(self, file_path: Path) -> Any:
        """Load PyTorch model"""
        try:
            import torch
            return torch.load(file_path, map_location='cpu')
        except ImportError:
            raise ImportError("torch not installed. Install with: pip install torch")
    
    def validate(self, model: Any) -> Tuple[bool, str]:
        """Validate PyTorch model"""
        try:
            # Check if it's a state dict or model
            if isinstance(model, dict):
                return True, "Valid PyTorch state dict"
            elif hasattr(model, 'forward'):
                return True, "Valid PyTorch model"
            else:
                return True, "PyTorch object loaded but type unclear"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"
    
    def get_metadata(self, model: Any, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from PyTorch model"""
        metadata = {
            'framework': 'pytorch',
            'format': 'pytorch',
            'file_size': file_path.stat().st_size,
            'is_state_dict': isinstance(model, dict),
            'has_forward': hasattr(model, 'forward'),
        }
        
        if isinstance(model, dict):
            metadata['num_layers'] = len(model)
            metadata['layer_names'] = list(model.keys())[:5]  # First 5 layers
        
        return metadata


class TensorFlowHandler(ModelHandler):
    """Handles TensorFlow SavedModel format"""
    
    def detect(self, file_path: Path) -> bool:
        """Detect if file is TensorFlow SavedModel format"""
        # SavedModel is a directory with specific structure
        if file_path.is_dir():
            return (file_path / 'saved_model.pb').exists() or (file_path / 'assets').exists()
        return file_path.suffix.lower() == '.pb'
    
    def load(self, file_path: Path) -> Any:
        """Load TensorFlow SavedModel"""
        try:
            import tensorflow as tf
            return tf.saved_model.load(str(file_path))
        except ImportError:
            raise ImportError("tensorflow not installed. Install with: pip install tensorflow")
    
    def validate(self, model: Any) -> Tuple[bool, str]:
        """Validate TensorFlow model"""
        try:
            if hasattr(model, '__call__') or hasattr(model, 'signatures'):
                return True, "Valid TensorFlow SavedModel"
            else:
                return False, "TensorFlow object invalid"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"
    
    def get_metadata(self, model: Any, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from TensorFlow model"""
        metadata = {
            'framework': 'tensorflow',
            'format': 'tensorflow_savedmodel',
            'file_size': file_path.stat().st_size if file_path.is_file() else None,
            'has_signatures': hasattr(model, 'signatures'),
        }
        
        if hasattr(model, 'signatures'):
            metadata['signatures'] = list(model.signatures.keys())
        
        return metadata


class KerasH5Handler(ModelHandler):
    """Handles Keras H5 format"""
    
    def detect(self, file_path: Path) -> bool:
        """Detect if file is Keras H5 format"""
        return file_path.suffix.lower() == '.h5'
    
    def load(self, file_path: Path) -> Any:
        """Load Keras H5 model"""
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(str(file_path))
        except ImportError:
            raise ImportError("tensorflow not installed. Install with: pip install tensorflow")
    
    def validate(self, model: Any) -> Tuple[bool, str]:
        """Validate Keras H5 model"""
        try:
            if hasattr(model, 'predict'):
                return True, "Valid Keras H5 model"
            else:
                return False, "Keras model invalid"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"
    
    def get_metadata(self, model: Any, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from Keras H5 model"""
        metadata = {
            'framework': 'keras',
            'format': 'keras_h5',
            'file_size': file_path.stat().st_size,
        }
        
        if hasattr(model, 'summary'):
            try:
                metadata['num_layers'] = len(model.layers)
                metadata['input_shape'] = str(model.input_shape)
                metadata['output_shape'] = str(model.output_shape)
            except:
                pass
        
        return metadata


class ONNXHandler(ModelHandler):
    """Handles ONNX format"""
    
    def detect(self, file_path: Path) -> bool:
        """Detect if file is ONNX format"""
        return file_path.suffix.lower() == '.onnx'
    
    def load(self, file_path: Path) -> Any:
        """Load ONNX model"""
        try:
            import onnx
            return onnx.load(str(file_path))
        except ImportError:
            raise ImportError("onnx not installed. Install with: pip install onnx")
    
    def validate(self, model: Any) -> Tuple[bool, str]:
        """Validate ONNX model"""
        try:
            if not hasattr(model, 'graph'):
                return False, "ONNX model missing graph"
            if len(model.graph.node) == 0:
                return False, "ONNX graph has no nodes"
            if len(model.graph.input) == 0:
                return False, "ONNX graph has no inputs"
            if len(model.graph.output) == 0:
                return False, "ONNX graph has no outputs"
            return True, "ONNX graph structure is valid"
        except Exception as e:
            return False, f"ONNX validation failed: {str(e)}"
    
    def get_metadata(self, model: Any, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from ONNX model"""
        metadata = {
            'framework': 'onnx',
            'format': 'onnx',
            'file_size': file_path.stat().st_size,
            'ir_version': model.ir_version,
            'producer_name': model.producer_name,
            'num_inputs': len(model.graph.input),
            'num_outputs': len(model.graph.output),
        }
        
        # Get input/output shapes
        inputs = []
        for inp in model.graph.input:
            inputs.append({
                'name': inp.name,
                'shape': [d.dim_value for d in inp.type.tensor_type.shape.dim]
            })
        metadata['inputs'] = inputs
        
        outputs = []
        for outp in model.graph.output:
            outputs.append({
                'name': outp.name,
                'shape': [d.dim_value for d in outp.type.tensor_type.shape.dim] if outp.type.tensor_type.shape else []
            })
        metadata['outputs'] = outputs
        
        return metadata


class ModelHandlerRegistry:
    """Registry for model format handlers"""
    
    def __init__(self):
        self.handlers = [
            ONNXHandler(),
            TensorFlowHandler(),
            KerasH5Handler(),
            PyTorchHandler(),
            XGBoostHandler(),
            LightGBMHandler(),
            CatBoostHandler(),
            JoblibHandler(),
            PickleHandler(),
        ]
    
    def detect_format(self, file_path: Path) -> Optional[ModelFormat]:
        """Detect format of model file"""
        for handler in self.handlers:
            if handler.detect(file_path):
                # Map handler to ModelFormat
                if isinstance(handler, ONNXHandler):
                    return ModelFormat.ONNX
                elif isinstance(handler, TensorFlowHandler):
                    return ModelFormat.TENSORFLOW_SAVEDMODEL
                elif isinstance(handler, KerasH5Handler):
                    return ModelFormat.KERAS_H5
                elif isinstance(handler, PyTorchHandler):
                    return ModelFormat.PYTORCH_PT if file_path.suffix == '.pt' else ModelFormat.PYTORCH_PTH
                elif isinstance(handler, XGBoostHandler):
                    return ModelFormat.XGBOOST
                elif isinstance(handler, LightGBMHandler):
                    return ModelFormat.LIGHTGBM
                elif isinstance(handler, CatBoostHandler):
                    return ModelFormat.CATBOOST
                elif isinstance(handler, JoblibHandler):
                    return ModelFormat.JOBLIB
                elif isinstance(handler, PickleHandler):
                    return ModelFormat.PICKLE
        
        return None
    
    def get_handler(self, file_path: Path) -> Optional[ModelHandler]:
        """Get handler for model file"""
        for handler in self.handlers:
            if handler.detect(file_path):
                return handler
        return None
    
    def get_handler_for_format(self, model_format: ModelFormat) -> Optional[ModelHandler]:
        """Get handler for specific format"""
        for handler in self.handlers:
            format_detected = handler.detect(Path(f"dummy{config.format_extensions[model_format]}"))
            if format_detected:
                return handler
        return None


# Global instance
handler_registry = ModelHandlerRegistry()
