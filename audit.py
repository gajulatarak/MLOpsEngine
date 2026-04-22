"""
Audit and Lineage Tracking System
Maintains comprehensive audit trail, lineage, and traceability for all models.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import sqlite3
import platform
import json as json_lib

from config import config, ModelFormat, ModelFramework


class AuditEventType(Enum):
    """Types of audit events"""
    MODEL_UPLOAD = "model_upload"
    MODEL_CONVERTED = "model_converted"
    MODEL_REGISTERED = "model_registered"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_INFERENCE = "model_inference"
    MODEL_UPDATED = "model_updated"
    MODEL_DELETED = "model_deleted"
    LINEAGE_CREATED = "lineage_created"


@dataclass
class ModelHash:
    """Model file hash information"""
    md5: str
    sha256: str
    file_size: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EnvironmentInfo:
    """Environment information at model registration time"""
    python_version: str
    platform: str
    framework_versions: Dict[str, str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class DataProfile:
    """Data profiling information"""
    sample_count: int
    feature_count: int
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    statistics: Dict[str, Dict[str, float]]


@dataclass
class AuditEntry:
    """Single audit entry"""
    event_id: str
    event_type: AuditEventType
    model_id: str
    model_name: str
    model_version: str
    source_format: str
    target_format: str
    timestamp: str
    user: str
    status: str  # success, failure, pending
    details: Dict[str, Any]
    model_hash: Optional[ModelHash] = None
    environment_info: Optional[EnvironmentInfo] = None
    data_profile: Optional[DataProfile] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        if self.model_hash:
            data['model_hash'] = asdict(self.model_hash)
        if self.environment_info:
            data['environment_info'] = asdict(self.environment_info)
        if self.data_profile:
            data['data_profile'] = asdict(self.data_profile)
        return data


@dataclass
class LineageEntry:
    """Lineage/traceability entry tracking model transformations"""
    lineage_id: str
    model_id: str
    parent_model_id: Optional[str]
    parent_version: Optional[str]
    source_format: ModelFormat
    target_format: ModelFormat
    conversion_method: str
    timestamp: str
    parameters: Dict[str, Any]
    data_loss_metrics: Dict[str, float] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """Manages audit logging and traceability"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.db_path = config.audit.audit_db_path.replace("sqlite:///", "")
        self._init_db()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup audit logger"""
        logger = logging.getLogger("mlops_audit")
        logger.setLevel(logging.INFO)
        
        # File handler
        handler = logging.FileHandler(config.audit.log_file_path)
        handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _init_db(self):
        """Initialize audit database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Audit events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT,
                model_id TEXT,
                model_name TEXT,
                model_version TEXT,
                source_format TEXT,
                target_format TEXT,
                timestamp TEXT,
                user TEXT,
                status TEXT,
                details TEXT,
                model_hash TEXT,
                environment_info TEXT,
                data_profile TEXT
            )
        """)
        
        # Lineage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lineage (
                lineage_id TEXT PRIMARY KEY,
                model_id TEXT,
                parent_model_id TEXT,
                parent_version TEXT,
                source_format TEXT,
                target_format TEXT,
                conversion_method TEXT,
                timestamp TEXT,
                parameters TEXT,
                data_loss_metrics TEXT,
                validation_results TEXT,
                FOREIGN KEY (model_id) REFERENCES audit_events(model_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_audit_event(self, entry: AuditEntry) -> None:
        """Log an audit event"""
        try:
            # Log to file
            self.logger.info(f"Event: {entry.event_type.value} | Model: {entry.model_name} "
                           f"| Version: {entry.model_version} | Status: {entry.status}")
            
            # Log to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO audit_events VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                entry.event_id,
                entry.event_type.value,
                entry.model_id,
                entry.model_name,
                entry.model_version,
                entry.source_format,
                entry.target_format,
                entry.timestamp,
                entry.user,
                entry.status,
                json_lib.dumps(entry.details),
                json_lib.dumps(asdict(entry.model_hash)) if entry.model_hash else None,
                json_lib.dumps(asdict(entry.environment_info)) if entry.environment_info else None,
                json_lib.dumps(asdict(entry.data_profile)) if entry.data_profile else None,
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Error logging audit event: {str(e)}")
            raise
    
    def log_lineage(self, entry: LineageEntry) -> None:
        """Log lineage/traceability information"""
        try:
            self.logger.info(f"Lineage: {entry.source_format.value} -> {entry.target_format.value} "
                           f"| Method: {entry.conversion_method} | Model: {entry.model_id}")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO lineage VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                entry.lineage_id,
                entry.model_id,
                entry.parent_model_id,
                entry.parent_version,
                entry.source_format.value,
                entry.target_format.value,
                entry.conversion_method,
                entry.timestamp,
                json_lib.dumps(entry.parameters),
                json_lib.dumps(entry.data_loss_metrics),
                json_lib.dumps(entry.validation_results),
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Error logging lineage: {str(e)}")
            raise
    
    def get_audit_trail(self, model_id: str) -> List[Dict[str, Any]]:
        """Get complete audit trail for a model"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM audit_events WHERE model_id = ? ORDER BY timestamp
        """, (model_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        result = []
        for row in rows:
            entry = dict(row)
            # Parse JSON fields
            for key in ['details', 'model_hash', 'environment_info', 'data_profile']:
                if entry[key]:
                    entry[key] = json_lib.loads(entry[key])
            result.append(entry)
        
        return result
    
    def get_lineage_trace(self, model_id: str) -> List[Dict[str, Any]]:
        """Get complete lineage trace for a model"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM lineage WHERE model_id = ? ORDER BY timestamp
        """, (model_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        result = []
        for row in rows:
            entry = dict(row)
            # Parse JSON fields
            for key in ['parameters', 'data_loss_metrics', 'validation_results']:
                if entry[key]:
                    entry[key] = json_lib.loads(entry[key])
            result.append(entry)
        
        return result
    
    @staticmethod
    def compute_file_hash(file_path: Path) -> ModelHash:
        """Compute MD5 and SHA256 hash of a model file"""
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        file_size = 0
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
                file_size += len(chunk)
        
        return ModelHash(
            md5=md5_hash.hexdigest(),
            sha256=sha256_hash.hexdigest(),
            file_size=file_size
        )
    
    @staticmethod
    def capture_environment_info() -> EnvironmentInfo:
        """Capture current environment information"""
        import sys
        
        framework_versions = {}
        frameworks = ['tensorflow', 'torch', 'sklearn', 'xgboost', 'lightgbm', 'catboost']
        
        for fw in frameworks:
            try:
                module = __import__(fw)
                framework_versions[fw] = module.__version__
            except (ImportError, AttributeError):
                pass
        
        return EnvironmentInfo(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=f"{platform.system()} {platform.release()}",
            framework_versions=framework_versions
        )


# Global instance
audit_logger = AuditLogger()
