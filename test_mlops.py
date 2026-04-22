"""
Unit Tests for Unified MLOps Layer
Tests for model handlers, converters, orchestrator, and API.
"""

import pytest
import tempfile
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd

from config import config, ModelFormat, ModelFramework
from model_handlers import handler_registry, PickleHandler, JoblibHandler
from audit import AuditLogger, AuditEntry, AuditEventType, LineageEntry
from orchestrator import mlops_orchestrator


class TestAuditLogger:
    """Test audit logging functionality"""
    
    def test_file_hash_computation(self):
        """Test file hash computation"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test data")
            f.flush()
            
            hash_info = AuditLogger.compute_file_hash(Path(f.name))
            
            assert hash_info.md5 is not None
            assert hash_info.sha256 is not None
            assert hash_info.file_size == 9
            
            Path(f.name).unlink()
    
    def test_environment_info_capture(self):
        """Test environment information capture"""
        env_info = AuditLogger.capture_environment_info()
        
        assert env_info.python_version is not None
        assert env_info.platform is not None
        assert env_info.framework_versions is not None
    
    def test_audit_entry_creation(self):
        """Test audit entry creation"""
        entry = AuditEntry(
            event_id="test-event",
            event_type=AuditEventType.MODEL_UPLOAD,
            model_id="model-123",
            model_name="test_model",
            model_version="1.0",
            source_format="pickle",
            target_format="pickle",
            timestamp="2024-04-21T00:00:00Z",
            user="test_user",
            status="success",
            details={"test": True}
        )
        
        assert entry.event_id == "test-event"
        assert entry.event_type == AuditEventType.MODEL_UPLOAD
        assert entry.status == "success"
        
        # Test dict conversion
        entry_dict = entry.to_dict()
        assert entry_dict['event_type'] == 'model_upload'


class TestModelHandlers:
    """Test model format handlers"""
    
    def test_pickle_handler_detection(self):
        """Test pickle format detection"""
        handler = PickleHandler()
        
        # Test with pickle extension
        assert handler.detect(Path("model.pkl"))
        assert handler.detect(Path("model.pickle"))
        assert not handler.detect(Path("model.joblib"))
    
    def test_pickle_model_save_and_load(self):
        """Test saving and loading pickle models"""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            # Create a simple model
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=42)
            
            # Fit on sample data
            X = np.array([[1, 2], [3, 4], [5, 6]])
            y = np.array([0, 1, 0])
            model.fit(X, y)
            
            # Save model
            pickle.dump(model, open(f.name, 'wb'))
            
            # Load with handler
            handler = PickleHandler()
            loaded_model = handler.load(Path(f.name))
            
            assert loaded_model is not None
            assert hasattr(loaded_model, 'predict')
            
            # Get metadata
            metadata = handler.get_metadata(loaded_model, Path(f.name))
            assert metadata['framework'] == 'sklearn'
            assert metadata['has_predict']
            
            Path(f.name).unlink()
    
    def test_format_detection_registry(self):
        """Test format detection through registry"""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()
            pickle.dump(model, open(f.name, 'wb'))
            
            # Detect format
            detected_format = handler_registry.detect_format(Path(f.name))
            assert detected_format == ModelFormat.PICKLE
            
            # Get handler
            handler = handler_registry.get_handler(Path(f.name))
            assert isinstance(handler, PickleHandler)
            
            Path(f.name).unlink()


class TestAuditAndLineage:
    """Test audit and lineage tracking"""
    
    def test_audit_trail_recording(self, tmp_path):
        """Test recording audit trail"""
        # Create temporary audit database
        db_path = tmp_path / "audit.db"
        
        from audit import AuditLogger
        logger = AuditLogger()
        
        # Create and log audit entry
        entry = AuditEntry(
            event_id="audit-test-1",
            event_type=AuditEventType.MODEL_UPLOAD,
            model_id="model-456",
            model_name="test_audit",
            model_version="1.0",
            source_format="pickle",
            target_format="pickle",
            timestamp="2024-04-21T10:00:00Z",
            user="test_user",
            status="success",
            details={"test": "data"}
        )
        
        logger.log_audit_event(entry)
        
        # Retrieve audit trail
        trail = logger.get_audit_trail("model-456")
        assert len(trail) > 0
        assert trail[0]['model_name'] == 'test_audit'
    
    def test_lineage_tracking(self):
        """Test lineage tracking"""
        from audit import audit_logger
        
        entry = LineageEntry(
            lineage_id="lineage-test-1",
            model_id="model-789",
            parent_model_id=None,
            parent_version=None,
            source_format=ModelFormat.PICKLE,
            target_format=ModelFormat.ONNX,
            conversion_method="PickleToONNX",
            timestamp="2024-04-21T11:00:00Z",
            parameters={"batch_size": 32},
            data_loss_metrics={"compression": 0.85}
        )
        
        audit_logger.log_lineage(entry)
        
        # Retrieve lineage
        lineage = audit_logger.get_lineage_trace("model-789")
        assert len(lineage) > 0


class TestOrchestrator:
    """Test MLOps orchestrator"""
    
    def test_get_supported_formats(self):
        """Test getting supported formats"""
        formats = mlops_orchestrator.get_supported_formats()
        
        assert 'pickle' in formats
        assert 'onnx' in formats
        assert 'pytorch_pt' in formats
        
        assert 'extension' in formats['pickle']
        assert 'framework' in formats['pickle']
    
    def test_model_upload(self):
        """Test model upload functionality"""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            # Create and save a simple model
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier()
            X = np.array([[1, 2], [3, 4], [5, 6]])
            y = np.array([0, 1, 0])
            model.fit(X, y)
            
            pickle.dump(model, open(f.name, 'wb'))
            
            # Upload through orchestrator
            success, msg, result = mlops_orchestrator.upload_model(
                source_file_path=Path(f.name),
                model_name="test_tree",
                user="test_scientist"
            )
            
            assert success
            assert result is not None
            assert 'model_id' in result
            assert 'model_hash' in result
            
            Path(f.name).unlink()


class TestModelFormatSupport:
    """Test support for various model formats"""
    
    def test_supported_format_list(self):
        """Test that all expected formats are supported"""
        expected_formats = [
            ModelFormat.PICKLE,
            ModelFormat.JOBLIB,
            ModelFormat.ONNX,
            ModelFormat.PYTORCH_PT,
            ModelFormat.PYTORCH_PTH,
            ModelFormat.KERAS_H5,
            ModelFormat.TENSORFLOW_SAVEDMODEL,
            ModelFormat.XGBOOST,
            ModelFormat.LIGHTGBM,
            ModelFormat.CATBOOST,
        ]
        
        for fmt in expected_formats:
            assert fmt in config.format_extensions
            assert fmt in config.format_framework_mapping


class TestConfigManagement:
    """Test configuration management"""
    
    def test_storage_config_creation(self):
        """Test storage configuration"""
        assert config.storage is not None
        assert config.storage.base_path is not None
        assert config.storage.raw_models_path is not None
        assert config.storage.converted_models_path is not None
    
    def test_mlflow_config(self):
        """Test MLFlow configuration"""
        assert config.mlflow is not None
        assert config.mlflow.tracking_uri is not None
        assert config.mlflow.experiment_name is not None
    
    def test_audit_config(self):
        """Test audit configuration"""
        assert config.audit is not None
        assert config.audit.enable_audit
        assert config.audit.log_file_path is not None


class TestDataTypes:
    """Test data type handling"""
    
    def test_framework_enum(self):
        """Test framework enum"""
        assert ModelFramework.TENSORFLOW is not None
        assert ModelFramework.PYTORCH is not None
        assert ModelFramework.SKLEARN is not None
    
    def test_model_format_enum(self):
        """Test model format enum"""
        assert ModelFormat.ONNX is not None
        assert ModelFormat.PICKLE is not None
        pickle_ext = config.format_extensions[ModelFormat.PICKLE]
        assert pickle_ext == '.pkl'


@pytest.fixture
def temp_model_file():
    """Fixture to create temporary model file"""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        pickle.dump(model, open(f.name, 'wb'))
        
        yield Path(f.name)
        
        Path(f.name).unlink()


def test_end_to_end_workflow(temp_model_file):
    """Test end-to-end workflow"""
    # Upload
    success, msg, result = mlops_orchestrator.upload_model(
        source_file_path=temp_model_file,
        model_name="e2e_test",
        user="test_user"
    )
    assert success
    
    model_id = result['model_id']
    
    # Get audit trail
    success, msg, trail = mlops_orchestrator.get_model_audit_trail(model_id)
    assert success
    assert len(trail) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
