"""
MLOps REST API
FastAPI/Flask-based REST API for the unified MLOps layer.
"""

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from pathlib import Path
from typing import Dict, Any
import os
from datetime import datetime

from config import config
from orchestrator import mlops_orchestrator

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.api.max_upload_size_mb * 1024 * 1024
app.config['UPLOAD_FOLDER'] = str(config.storage.raw_models_path)

# CORS support
from flask_cors import CORS
CORS(app, origins=config.api.allowed_origins)


# ============================================================================
# Health and Status Endpoints
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
    })


@app.route('/info', methods=['GET'])
def info():
    """Get API information"""
    return jsonify({
        'name': 'Unified MLOps Layer',
        'version': '1.0.0',
        'max_upload_size_mb': config.api.max_upload_size_mb,
        'supported_formats': mlops_orchestrator.get_supported_formats(),
    })


# ============================================================================
# Model Management Endpoints
# ============================================================================

@app.route('/api/models/upload', methods=['POST'])
def upload_model():
    """
    Upload a model file.
    
    Request:
        - file: Model file (multipart/form-data)
        - model_name: Model name (form field)
        - metadata: Optional JSON metadata (form field)
    
    Returns:
        Model upload result with model_id and metadata
    """
    try:
        # Validate file
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get model name
        model_name = request.form.get('model_name', 'unnamed_model')
        user = request.form.get('user', 'api_user')
        
        # Parse metadata if provided
        metadata = None
        if 'metadata' in request.form:
            import json
            try:
                metadata = json.loads(request.form['metadata'])
            except:
                pass
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = Path(app.config['UPLOAD_FOLDER']) / f"temp_{filename}"
        file.save(str(temp_path))
        
        # Upload model
        success, message, result = mlops_orchestrator.upload_model(
            source_file_path=temp_path,
            model_name=model_name,
            user=user,
            metadata=metadata
        )
        
        if not success:
            temp_path.unlink()
            return jsonify({'error': message}), 400
        
        # Clean up temp file
        temp_path.unlink()
        
        return jsonify({
            'success': True,
            'message': message,
            'data': result,
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/models/<model_id>/convert-register', methods=['POST'])
def convert_and_register(model_id):
    """
    Convert and register a model with MLFlow.
    
    Request:
        - target_format: Target format (optional, defaults to ONNX)
    
    Returns:
        Conversion and registration result
    """
    try:
        model_name = request.json.get('model_name', model_id)
        target_format_str = request.json.get('target_format', 'onnx')
        user = request.json.get('user', 'api_user')
        
        # Convert format string to enum
        from config import ModelFormat
        try:
            target_format = ModelFormat[target_format_str.upper()]
        except KeyError:
            return jsonify({'error': f'Invalid format: {target_format_str}'}), 400
        
        success, message, result = mlops_orchestrator.convert_and_register(
            model_id=model_id,
            model_name=model_name,
            target_format=target_format,
            user=user
        )
        
        if not success:
            return jsonify({'error': message}), 400
        
        return jsonify({
            'success': True,
            'message': message,
            'data': result,
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Operation failed: {str(e)}'}), 500


@app.route('/api/models', methods=['GET'])
def list_models():
    """List all registered models"""
    try:
        success, message, models = mlops_orchestrator.list_models()
        
        if not success:
            return jsonify({'error': message}), 500
        
        return jsonify({
            'success': True,
            'message': message,
            'data': models,
            'count': len(models) if models else 0,
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to list models: {str(e)}'}), 500


@app.route('/api/models/<model_name>/info', methods=['GET'])
def get_model_info(model_name):
    """Get model information from registry"""
    try:
        stage = request.args.get('stage', 'Production')
        model_info = mlops_orchestrator.get_model_info(model_name, stage)
        
        if not model_info:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({
            'success': True,
            'data': model_info,
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500


# ============================================================================
# Audit and Lineage Endpoints
# ============================================================================

@app.route('/api/models/<model_id>/audit-trail', methods=['GET'])
def get_audit_trail(model_id):
    """Get audit trail for a model"""
    try:
        success, message, trail = mlops_orchestrator.get_model_audit_trail(model_id)
        
        if not success:
            return jsonify({'error': message}), 500
        
        return jsonify({
            'success': True,
            'message': message,
            'data': trail,
            'count': len(trail) if trail else 0,
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve audit trail: {str(e)}'}), 500


@app.route('/api/models/<model_id>/lineage', methods=['GET'])
def get_lineage(model_id):
    """Get lineage/traceability for a model"""
    try:
        success, message, lineage = mlops_orchestrator.get_model_lineage(model_id)
        
        if not success:
            return jsonify({'error': message}), 500
        
        return jsonify({
            'success': True,
            'message': message,
            'data': lineage,
            'count': len(lineage) if lineage else 0,
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve lineage: {str(e)}'}), 500


# ============================================================================
# Inference Endpoints
# ============================================================================

@app.route('/api/inference/<model_name>/predict', methods=['POST'])
def predict(model_name):
    """
    Make prediction using a registered model.
    
    Request:
        - data: Input data (JSON array or pandas-compatible format)
        - stage: Model stage (optional, defaults to Production)
    
    Returns:
        Predictions
    """
    try:
        import json
        import numpy as np
        import pandas as pd
        
        stage = request.json.get('stage', 'Production')
        data_input = request.json.get('data')
        
        if not data_input:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert input to appropriate format
        try:
            data = pd.DataFrame(data_input) if isinstance(data_input, list) else data_input
        except:
            data = np.array(data_input)
        
        success, predictions, message = mlops_orchestrator.predict(
            model_name=model_name,
            data=data,
            stage=stage
        )
        
        if not success:
            return jsonify({'error': message}), 400
        
        # Convert predictions to JSON-serializable format
        if hasattr(predictions, 'tolist'):
            predictions_data = predictions.tolist()
        else:
            predictions_data = predictions
        
        return jsonify({
            'success': True,
            'message': message,
            'model_name': model_name,
            'stage': stage,
            'predictions': predictions_data,
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/inference/<model_name>/batch-predict', methods=['POST'])
def batch_predict(model_name):
    """
    Batch prediction on uploaded data file.
    
    Request:
        - file: Data file (CSV or Excel)
        - stage: Model stage (optional)
    
    Returns:
        Path to predictions file
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        stage = request.form.get('stage', 'Production')
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = Path(app.config['UPLOAD_FOLDER']) / f"temp_batch_{filename}"
        file.save(str(temp_path))
        
        success, output_path, message = mlops_orchestrator.batch_predict(
            model_name=model_name,
            data_path=temp_path,
            stage=stage
        )
        
        temp_path.unlink()
        
        if not success:
            return jsonify({'error': message}), 400
        
        return jsonify({
            'success': True,
            'message': message,
            'output_file': str(output_path),
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500


# ============================================================================
# Model Lifecycle Endpoints
# ============================================================================

@app.route('/api/models/<model_name>/transition-stage', methods=['POST'])
def transition_stage(model_name):
    """
    Transition model to different stage.
    
    Request:
        - version: Model version
        - stage: Target stage (Staging, Production, Archived)
    """
    try:
        version = request.json.get('version')
        stage = request.json.get('stage')
        
        if not version or not stage:
            return jsonify({'error': 'version and stage are required'}), 400
        
        success, message = mlops_orchestrator.transition_model_stage(
            model_name=model_name,
            version=int(version),
            stage=stage
        )
        
        if not success:
            return jsonify({'error': message}), 400
        
        return jsonify({
            'success': True,
            'message': message,
            'model_name': model_name,
            'version': version,
            'stage': stage,
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Stage transition failed: {str(e)}'}), 500


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': f'File too large. Maximum size: {config.api.max_upload_size_mb}MB'
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# Static Files and UI
# ============================================================================

@app.route('/')
def index():
    """Serve main page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Unified MLOps Layer</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .container { 
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                padding: 40px;
                max-width: 600px;
                width: 100%;
            }
            h1 { 
                color: #333;
                margin-bottom: 20px;
                text-align: center;
            }
            .section {
                margin-bottom: 30px;
            }
            .section h2 {
                color: #667eea;
                font-size: 18px;
                margin-bottom: 15px;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }
            .api-link {
                display: block;
                margin: 10px 0;
                padding: 12px 15px;
                background: #f5f5f5;
                border-left: 4px solid #667eea;
                text-decoration: none;
                color: #333;
                border-radius: 4px;
                transition: all 0.3s;
            }
            .api-link:hover {
                background: #667eea;
                color: white;
            }
            .code {
                background: #f5f5f5;
                padding: 12px;
                border-radius: 4px;
                font-family: monospace;
                font-size: 12px;
                overflow-x: auto;
                margin: 10px 0;
            }
            .footer {
                text-align: center;
                color: #999;
                margin-top: 40px;
                font-size: 12px;
            }
            button {
                background: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.3s;
            }
            button:hover {
                background: #764ba2;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Unified MLOps Layer</h1>
            
            <div class="section">
                <h2>📊 Quick Links</h2>
                <a href="/api/models" class="api-link">View All Models</a>
                <a href="/health" class="api-link">Health Check</a>
                <a href="/info" class="api-link">API Info</a>
            </div>
            
            <div class="section">
                <h2>📤 Model Upload</h2>
                <form id="uploadForm" enctype="multipart/form-data">
                    <div style="margin-bottom: 15px;">
                        <input type="text" id="modelName" placeholder="Model Name" 
                               style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                    </div>
                    <div style="margin-bottom: 15px;">
                        <input type="file" id="modelFile" accept=".pkl,.joblib,.onnx,.pt,.pth,.h5,.xgb,.lgb,.cb"
                               style="width: 100%; padding: 8px;">
                    </div>
                    <button type="submit">Upload Model</button>
                </form>
                <div id="uploadResult"></div>
            </div>
            
            <div class="section">
                <h2>🔗 Supported Formats</h2>
                <p style="font-size: 12px; color: #666;">
                    Deep Learning: ONNX, TensorFlow, Keras H5, PyTorch<br>
                    Classical ML: Pickle, Joblib, PMML<br>
                    Tree Models: XGBoost, LightGBM, CatBoost
                </p>
            </div>
            
            <div class="footer">
                Version 1.0 | MLOps Orchestrator with MLFlow Integration
            </div>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData();
                formData.append('file', document.getElementById('modelFile').files[0]);
                formData.append('model_name', document.getElementById('modelName').value);
                
                try {
                    const response = await fetch('/api/models/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    document.getElementById('uploadResult').innerHTML = 
                        '<div class="code">' + JSON.stringify(result, null, 2) + '</div>';
                } catch (error) {
                    document.getElementById('uploadResult').innerHTML = 
                        '<div class="code" style="color: red;">Error: ' + error + '</div>';
                }
            });
        </script>
    </body>
    </html>
    '''


def create_app():
    """Factory function to create Flask app"""
    return app


if __name__ == '__main__':
    app.run(
        host=config.api.host,
        port=config.api.port,
        debug=config.api.debug
    )
