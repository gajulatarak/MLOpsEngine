"""
Example Usage Scripts for Unified MLOps Layer

This file demonstrates various use cases:
1. Training and uploading different model types
2. Format conversion
3. Model registration and inference
4. Audit trail and lineage tracking
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from orchestrator import mlops_orchestrator
from config import ModelFormat


# ============================================================================
# Example 1: Scikit-learn Model Upload and Inference
# ============================================================================

def example_sklearn_workflow():
    """Example: Train and deploy scikit-learn model"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Scikit-learn Model Workflow")
    print("="*70)
    
    try:
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        
        # 1. Train model
        iris = load_iris()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(iris.data, iris.target)
        
        # 2. Save model locally
        model_path = Path("models/iris_classifier.pkl")
        model_path.parent.mkdir(exist_ok=True)
        pickle.dump(model, open(model_path, 'wb'))
        print(f"✓ Model saved to {model_path}")
        
        # 3. Upload to MLOps platform
        success, msg, result = mlops_orchestrator.upload_model(
            source_file_path=model_path,
            model_name="iris_classifier",
            user="scientist@company.com",
            metadata={
                "accuracy": 0.97,
                "dataset": "iris",
                "train_date": datetime.now().isoformat()
            }
        )
        
        if success:
            model_id = result['model_id']
            print(f"✓ Model uploaded: {model_id}")
            print(f"  Format: {result['format']}")
            print(f"  Hash (SHA256): {result['model_hash']['sha256'][:16]}...")
            
            # 4. Convert to ONNX and register with MLFlow
            success, msg, reg_result = mlops_orchestrator.convert_and_register(
                model_id=model_id,
                model_name="iris_classifier",
                target_format=ModelFormat.ONNX,
                user="scientist@company.com"
            )
            
            if success:
                print(f"✓ Model registered with MLFlow")
                print(f"  Model URI: {reg_result['model_uri']}")
                
                # 5. Make predictions
                test_data = pd.DataFrame([
                    [5.1, 3.5, 1.4, 0.2],
                    [6.2, 2.9, 4.3, 1.3],
                    [7.1, 3.0, 5.9, 2.1]
                ])
                
                success, predictions, msg = mlops_orchestrator.predict(
                    model_name="iris_classifier_onnx",
                    data=test_data,
                    stage="Production"
                )
                
                if success:
                    print(f"✓ Predictions made successfully")
                    print(f"  Output: {predictions[:3]}")
                
                # 6. View audit trail
                success, msg, trail = mlops_orchestrator.get_model_audit_trail(model_id)
                if success:
                    print(f"✓ Audit trail ({len(trail)} events):")
                    for event in trail:
                        print(f"  - {event['event_type']}: {event['status']} at {event['timestamp']}")
        
    except Exception as e:
        print(f"✗ Error in sklearn workflow: {str(e)}")


# ============================================================================
# Example 2: PyTorch Model Upload and Conversion
# ============================================================================

def example_pytorch_workflow():
    """Example: Train and deploy PyTorch model"""
    print("\n" + "="*70)
    print("EXAMPLE 2: PyTorch Model Workflow")
    print("="*70)
    
    try:
        import torch
        import torch.nn as nn
        
        # 1. Create a simple neural network
        class SimpleNet(nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 3)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
        
        # 2. Train model
        model = SimpleNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        print("✓ PyTorch model trained")
        
        # 3. Save model
        model_path = Path("models/pytorch_model.pt")
        model_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"✓ Model saved to {model_path}")
        
        # 4. Upload to MLOps platform
        success, msg, result = mlops_orchestrator.upload_model(
            source_file_path=model_path,
            model_name="classification_net",
            user="ml_engineer@company.com",
            metadata={
                "framework": "pytorch",
                "architecture": "SimpleNet",
                "parameters": 80
            }
        )
        
        if success:
            model_id = result['model_id']
            print(f"✓ PyTorch model uploaded: {model_id}")
            
            # 5. Convert to ONNX
            success, msg, conv_result = mlops_orchestrator.convert_and_register(
                model_id=model_id,
                model_name="classification_net",
                target_format=ModelFormat.ONNX,
                user="ml_engineer@company.com"
            )
            
            if success:
                print(f"✓ PyTorch model converted to ONNX and registered")
    
    except ImportError:
        print("  PyTorch not installed. Install with: pip install torch")
    except Exception as e:
        print(f"✗ Error in PyTorch workflow: {str(e)}")


# ============================================================================
# Example 3: XGBoost Model with Lineage Tracking
# ============================================================================

def example_xgboost_workflow():
    """Example: Deploy XGBoost model with lineage tracking"""
    print("\n" + "="*70)
    print("EXAMPLE 3: XGBoost Model with Lineage Tracking")
    print("="*70)
    
    try:
        import xgboost as xgb
        from sklearn.datasets import load_breast_cancer
        
        # 1. Train XGBoost model
        data = load_breast_cancer()
        X_train = data.data[:400]
        y_train = data.target[:400]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            'max_depth': 5,
            'eta': 0.1,
            'objective': 'binary:logistic',
        }
        
        model = xgb.train(params, dtrain, num_boost_round=10)
        
        # 2. Save model
        model_path = Path("models/xgboost_model.xgb")
        model_path.parent.mkdir(exist_ok=True)
        model.save_model(str(model_path))
        print(f"✓ XGBoost model saved to {model_path}")
        
        # 3. Upload
        success, msg, result = mlops_orchestrator.upload_model(
            source_file_path=model_path,
            model_name="cancer_classifier_xgb",
            user="data_scientist@company.com",
            metadata={"dataset": "breast_cancer", "auc": 0.96}
        )
        
        if success:
            model_id = result['model_id']
            print(f"✓ XGBoost model uploaded: {model_id}")
            
            # 4. Get lineage trace
            success, msg, lineage = mlops_orchestrator.get_model_lineage(model_id)
            if success:
                print(f"✓ Lineage trace retrieved ({len(lineage)} conversions)")
                for entry in lineage:
                    print(f"  - {entry['source_format']} → {entry['target_format']} "
                          f"via {entry['conversion_method']}")
    
    except ImportError:
        print("  XGBoost not installed. Install with: pip install xgboost")
    except Exception as e:
        print(f"✗ Error in XGBoost workflow: {str(e)}")


# ============================================================================
# Example 4: Batch Prediction and Results
# ============================================================================

def example_batch_prediction():
    """Example: Perform batch predictions"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Prediction")
    print("="*70)
    
    try:
        # Create sample data
        data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100),
            'feature4': np.random.rand(100),
        })
        
        # Save to CSV
        data_path = Path("data/batch_input.csv")
        data_path.parent.mkdir(exist_ok=True)
        data.to_csv(data_path, index=False)
        print(f"✓ Batch data saved to {data_path}")
        
        # Run batch prediction
        # Note: This would work with an actual registered model
        print("  (Batch prediction would be executed with a registered model)")
    
    except Exception as e:
        print(f"✗ Error in batch prediction: {str(e)}")


# ============================================================================
# Example 5: Audit and Compliance Reporting
# ============================================================================

def example_audit_reporting():
    """Example: Generate audit and compliance reports"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Audit and Compliance Reporting")
    print("="*70)
    
    try:
        # List all models
        success, msg, models = mlops_orchestrator.list_models()
        
        if success:
            print(f"✓ Total models in registry: {len(models) if models else 0}")
            if models:
                for model in models[:3]:  # Show first 3
                    print(f"  - {model['name']}: {model['latest_versions']} versions")
        
        print("\n✓ Sample Audit Report:")
        print("  For detailed audit trails, use:")
        print("  - GET /api/models/{model_id}/audit-trail")
        print("  - GET /api/models/{model_id}/lineage")
    
    except Exception as e:
        print(f"✗ Error in audit reporting: {str(e)}")


# ============================================================================
# Example 6: Supported Formats Overview
# ============================================================================

def example_supported_formats():
    """Example: Show supported model formats"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Supported Model Formats")
    print("="*70)
    
    formats = mlops_orchestrator.get_supported_formats()
    
    print("\n📦 Deep Learning Formats:")
    dl_formats = ['onnx', 'tensorflow_savedmodel', 'keras_h5', 'pytorch_pt', 'pytorch_pth']
    for fmt in dl_formats:
        if fmt in formats:
            ext = formats[fmt].get('extension', 'N/A')
            fw = formats[fmt].get('framework', 'N/A')
            print(f"  ✓ {fmt:20} ({ext:10}) - {fw}")
    
    print("\n📦 Classical ML Formats:")
    ml_formats = ['pickle', 'joblib', 'pmml']
    for fmt in ml_formats:
        if fmt in formats:
            ext = formats[fmt].get('extension', 'N/A')
            fw = formats[fmt].get('framework', 'N/A')
            print(f"  ✓ {fmt:20} ({ext:10}) - {fw}")
    
    print("\n📦 Tree-based & Gradient Boosting:")
    tree_formats = ['xgboost', 'lightgbm', 'catboost']
    for fmt in tree_formats:
        if fmt in formats:
            ext = formats[fmt].get('extension', 'N/A')
            fw = formats[fmt].get('framework', 'N/A')
            print(f"  ✓ {fmt:20} ({ext:10}) - {fw}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("UNIFIED MLOps LAYER - USAGE EXAMPLES")
    print("="*70)
    
    # Run examples
    example_supported_formats()
    example_sklearn_workflow()
    example_pytorch_workflow()
    example_xgboost_workflow()
    example_batch_prediction()
    example_audit_reporting()
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETED")
    print("="*70)
    print("\nFor more information, see README.md or visit:")
    print("  - API: http://localhost:5001")
    print("  - MLFlow: http://localhost:5000")


if __name__ == '__main__':
    main()
