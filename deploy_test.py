"""
End-to-End Deployment Test Script
Tests the complete MLOps workflow from model upload to inference with audit trails.
"""

import sys
import json
import pickle
import pathlib
import time
import requests
import traceback
from datetime import datetime

API_BASE = "http://127.0.0.1:5001"

PASSED = []
FAILED = []


def step(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def ok(msg):
    print(f"  [OK] {msg}")
    PASSED.append(msg)


def fail(msg, detail=""):
    print(f"  [FAIL] {msg}")
    if detail:
        print(f"    {detail}")
    FAILED.append(msg)


def check_api():
    step("1. API Health Check")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        d = r.json()
        ok(f"API is healthy (status={d.get('status', 'unknown')})")
        r = requests.get(f"{API_BASE}/info", timeout=5)
        d = r.json()
        ok(f"Supported formats: {len(d['supported_formats'])} formats")
        for fmt_name in list(d['supported_formats'].keys())[:5]:
            ext = d['supported_formats'][fmt_name]['extension']
            fw = d['supported_formats'][fmt_name]['framework']
            print(f"         * {fmt_name:25} ({ext:8}) -> {fw}")
        return True
    except Exception as e:
        fail("API not reachable", str(e))
        return False


def train_and_save_models():
    step("2. Training & Saving Test Models")

    pathlib.Path("test_models").mkdir(exist_ok=True)

    # sklearn RandomForest
    try:
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        iris = load_iris()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(iris.data, iris.target)
        with open("test_models/iris_rf.pkl", "wb") as f:
            pickle.dump(model, f)
        ok("sklearn RandomForest saved -> test_models/iris_rf.pkl")
    except Exception as e:
        fail("sklearn model creation failed", str(e))

    # XGBoost
    try:
        import xgboost as xgb
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        dtrain = xgb.DMatrix(data.data[:400], label=data.target[:400])
        bst = xgb.train({"max_depth": 3, "eta": 0.1, "objective": "binary:logistic"},
                         dtrain, num_boost_round=5, verbose_eval=False)
        bst.save_model("test_models/breast_cancer.xgb")
        ok("XGBoost model saved -> test_models/breast_cancer.xgb")
    except Exception as e:
        fail("XGBoost model creation failed", str(e))

    # LightGBM
    try:
        import lightgbm as lgb
        from sklearn.datasets import load_wine
        data = load_wine()
        dtrain = lgb.Dataset(data.data, label=data.target)
        params = {"objective": "multiclass", "num_class": 3,
                  "n_estimators": 10, "verbose": -1}
        model = lgb.train(params, dtrain, num_boost_round=5)
        model.save_model("test_models/wine_classifier.lgb")
        ok("LightGBM model saved -> test_models/wine_classifier.lgb")
    except Exception as e:
        fail("LightGBM model creation failed", str(e))

    # Joblib
    try:
        import joblib
        from sklearn.svm import SVC
        iris = load_iris()
        svc = SVC(probability=True)
        svc.fit(iris.data[:100], iris.target[:100])
        joblib.dump(svc, "test_models/iris_svc.joblib")
        ok("SVC (joblib) saved -> test_models/iris_svc.joblib")
    except Exception as e:
        fail("Joblib model creation failed", str(e))


def upload_model(filepath, model_name, user="takeda_scientist"):
    try:
        with open(filepath, "rb") as f:
            files = {"file": (pathlib.Path(filepath).name, f, "application/octet-stream")}
            data = {"model_name": model_name, "user": user,
                    "metadata": json.dumps({"test": True, "created": datetime.now().isoformat()})}
            r = requests.post(f"{API_BASE}/api/models/upload", files=files, data=data, timeout=30)

        if r.status_code == 200:
            d = r.json()
            return d["data"]["model_id"], d["data"]
        else:
            return None, r.json()
    except Exception as e:
        return None, {"error": str(e)}


def test_model_uploads():
    step("3. Model Upload Tests")
    results = {}

    models = [
        ("test_models/iris_rf.pkl",         "iris_classifier",     "pickle"),
        ("test_models/breast_cancer.xgb",   "breast_cancer_xgb",   "xgboost"),
        ("test_models/wine_classifier.lgb", "wine_lgb",            "lightgbm"),
        ("test_models/iris_svc.joblib",     "iris_svc",            "joblib"),
    ]

    for path, name, expected_fmt in models:
        if not pathlib.Path(path).exists():
            fail(f"{name} skipped (file not found)")
            continue
        model_id, data = upload_model(path, name)
        if model_id:
            actual_fmt = data.get("format", "?")
            md5 = data.get("model_hash", {}).get("md5", "?")[:12]
            size = data.get("model_hash", {}).get("file_size", 0)
            ok(f"{name:35} fmt={actual_fmt:10} hash={md5}... size={size:,}B")
            results[name] = model_id
        else:
            fail(f"{name} upload failed", str(data))

    return results


def test_convert_register(model_uploads):
    step("4. Format Conversion & MLFlow Registration")
    registered = {}

    for name, model_id in model_uploads.items():
        try:
            r = requests.post(
                f"{API_BASE}/api/models/{model_id}/convert-register",
                json={"model_name": name, "target_format": "onnx", "user": "takeda_scientist"},
                timeout=120
            )
            if r.status_code == 200:
                d = r.json()["data"]
                ok(f"{name:35} {d['source_format']:10} -> {d['target_format']:6} | uri={d.get('model_uri','N/A')[:40]}")
                registered[name] = {"model_id": model_id, "uri": d.get("model_uri")}
            else:
                fail(f"{name} convert-register failed", str(r.json()))
        except Exception as e:
            fail(f"{name} convert-register error", str(e))

    return registered


def test_list_models():
    step("5. List Registered Models")
    try:
        r = requests.get(f"{API_BASE}/api/models", timeout=10)
        d = r.json()
        ok(f"Models in registry: {d.get('count', 0)}")
        if d.get("data"):
            for m in d["data"][:5]:
                print(f"    * {m.get('name', '?'):40} versions={m.get('latest_versions', 0)}")
        return True
    except Exception as e:
        fail("List models failed", str(e))
        return False


def test_audit_trails(model_uploads):
    step("6. Audit Trail & Lineage Verification")
    for name, model_id in list(model_uploads.items())[:2]:
        try:
            # Audit trail
            r = requests.get(f"{API_BASE}/api/models/{model_id}/audit-trail", timeout=10)
            trail = r.json().get("data", [])
            ok(f"{name:30} audit events: {len(trail)}")
            for evt in trail:
                print(f"    [{evt.get('timestamp','?')[:19]}] {evt.get('event_type','?'):25} {evt.get('status','?')}")

            # Lineage
            r2 = requests.get(f"{API_BASE}/api/models/{model_id}/lineage", timeout=10)
            lineage = r2.json().get("data", [])
            ok(f"{name:30} lineage entries: {len(lineage)}")
            for ln in lineage:
                compression = ln.get("data_loss_metrics", {}).get("size_compression_ratio")
                cr_str = f"compression={compression:.2f}" if compression else ""
                print(f"    {ln.get('source_format','?'):12} -> {ln.get('target_format','?'):8} via {ln.get('conversion_method','?')} {cr_str}")
        except Exception as e:
            fail(f"{name} audit/lineage failed", str(e))


def test_mlflow_ui():
    step("7. MLFlow Server Connectivity")
    try:
        r = requests.get("http://127.0.0.1:5000/health", timeout=5)
        ok(f"MLFlow server running (status={r.status_code})")
        r2 = requests.get("http://127.0.0.1:5000/api/2.0/mlflow/experiments/list", timeout=5)
        experiments = r2.json().get("experiments", [])
        ok(f"MLFlow experiments: {len(experiments)}")
        for exp in experiments[:3]:
            print(f"    * {exp.get('name', '?'):30} id={exp.get('experiment_id', '?')}")
        return True
    except Exception as e:
        fail("MLFlow connectivity failed", str(e))
        return False


def print_summary():
    print(f"\n{'='*60}")
    print(f"  DEPLOYMENT TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  [OK] PASSED: {len(PASSED)}")
    print(f"  [FAIL] FAILED: {len(FAILED)}")
    if FAILED:
        print(f"\n  Failed tests:")
        for f_item in FAILED:
            print(f"    * {f_item}")
    print(f"\n  {'='*60}")
    print(f"  ACCESS POINTS:")
    print(f"  * Web UI:   http://127.0.0.1:5001")
    print(f"  * REST API: http://127.0.0.1:5001/api/")
    print(f"  * MLFlow:   http://127.0.0.1:5000")
    print(f"  {'='*60}\n")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  MLOps Unified Layer - Deployment Test")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    if not check_api():
        print("\n[WARNING] API not running. Start it first: python api.py")
        sys.exit(1)

    train_and_save_models()
    uploads = test_model_uploads()
    registered = test_convert_register(uploads) if uploads else {}
    test_list_models()
    if uploads:
        test_audit_trails(uploads)
    test_mlflow_ui()
    print_summary()
