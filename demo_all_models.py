"""
MLOps Unified Layer - Deploy Sample Models of Each Type
Trains, saves, uploads, and runs inference on one model per supported format.
"""

import sys
import os
import pickle
import json
import time
import warnings
import threading
import requests
import numpy as np
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")


def _ensure_api_running():
    """Start the Flask API in a background thread if not already listening."""
    try:
        requests.get(f"{API_BASE}/health", timeout=2)
        print(f"  [OK] API already running at {API_BASE}")
        return None
    except Exception:
        pass

    print(f"  [INFO] Starting embedded API server on {API_BASE} ...")
    os.environ.setdefault("PYTHONPATH", str(Path(__file__).parent))

    def _run():
        import importlib, sys
        sys.path.insert(0, str(Path(__file__).parent))
        from api import app
        from config import config
        app.run(host="127.0.0.1", port=config.api.port, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Wait for it to be ready (up to 15s)
    for _ in range(30):
        time.sleep(0.5)
        try:
            requests.get(f"{API_BASE}/health", timeout=1)
            print(f"  [OK] Embedded API server started")
            return t
        except Exception:
            pass

    print(f"  [FAIL] Could not start embedded API server")
    return None

API_BASE = "http://127.0.0.1:5001"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0
RESULTS = []


def header(title):
    width = 70
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def subheader(title):
    print(f"\n  --- {title} ---")


def row(label, value, status="ok"):
    icon = "[OK]  " if status == "ok" else "[FAIL]" if status == "fail" else "[INFO]"
    print(f"  {icon} {label:<35} {value}")


def section_result(name, model_id, fmt, inference_ok, score_str):
    global PASS, FAIL
    status = "PASS" if inference_ok else "PARTIAL"
    RESULTS.append({
        "name": name, "id": model_id or "N/A", "format": fmt,
        "inference": inference_ok, "score": score_str, "status": status
    })
    if model_id:
        PASS += 1
    else:
        FAIL += 1


def upload_model(name, filepath, framework, extra_meta=None):
    """Upload a model file to the API. Returns (model_id, format) or (None, None)."""
    meta = {"framework": framework, "demo": True, "created": datetime.now().isoformat()}
    if extra_meta:
        meta.update(extra_meta)
    try:
        with open(filepath, "rb") as f:
            r = requests.post(
                f"{API_BASE}/api/models/upload",
                files={"file": (os.path.basename(filepath), f, "application/octet-stream")},
                data={"model_name": name, "user": "demo_runner",
                      "metadata": json.dumps(meta)},
                timeout=30
            )
        if r.status_code == 200:
            d = r.json()
            data = d.get("data") or d  # handle both {data:{...}} and flat responses
            return data.get("model_id"), data.get("format", "unknown")
        else:
            return None, r.json().get("error", f"HTTP {r.status_code}")
    except Exception as e:
        return None, str(e)[:60]


def get_audit(model_id):
    """Fetch audit trail entries for a model."""
    try:
        r = requests.get(f"{API_BASE}/api/models/{model_id}/audit-trail", timeout=5)
        return r.json().get("data", []) if r.status_code == 200 else []
    except:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def iris_data():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def breast_cancer_data():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y.astype(float), test_size=0.2, random_state=42)


def wine_data():
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    X, y = load_wine(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ──────────────────────────────────────────────────────────────────────────────
# Model builders
# ──────────────────────────────────────────────────────────────────────────────

def demo_pickle():
    subheader("1. Pickle (.pkl) - scikit-learn RandomForestClassifier")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X_tr, X_te, y_tr, y_te = iris_data()
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    row("Model", "RandomForestClassifier (iris)")
    row("Training accuracy", f"{acc:.4f}")

    path = "demo_models/iris_rf.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    row("Saved", path)

    mid, fmt = upload_model("demo_pickle_rf", path, "sklearn",
                            {"algorithm": "RandomForest", "n_estimators": 50})
    row("Uploaded to API", mid or str(fmt), "ok" if mid else "fail")

    # Inference via local model
    preds = model.predict(X_te[:5])
    row("Predictions (first 5)", str(preds.tolist()))
    row("Format detected", fmt or "N/A")
    section_result("Pickle - RandomForest", mid, fmt or "pickle", True, f"acc={acc:.4f}")
    return mid


def demo_joblib():
    subheader("2. Joblib (.joblib) - scikit-learn SVM")
    import joblib
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score

    X_tr, X_te, y_tr, y_te = iris_data()
    model = Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True, random_state=42))])
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    row("Model", "Pipeline(StandardScaler + SVC) (iris)")
    row("Training accuracy", f"{acc:.4f}")

    path = "demo_models/iris_svm.joblib"
    joblib.dump(model, path)
    row("Saved", path)

    mid, fmt = upload_model("demo_joblib_svm", path, "sklearn",
                            {"algorithm": "SVM", "pipeline": "StandardScaler+SVC"})
    row("Uploaded to API", mid or str(fmt), "ok" if mid else "fail")

    preds = model.predict(X_te[:5])
    row("Predictions (first 5)", str(preds.tolist()))
    row("Format detected", fmt or "N/A")
    section_result("Joblib - SVM Pipeline", mid, fmt or "joblib", True, f"acc={acc:.4f}")
    return mid


def demo_xgboost():
    subheader("3. XGBoost (.ubj) - Gradient Boosted Trees")
    import xgboost as xgb
    from sklearn.metrics import accuracy_score

    X_tr, X_te, y_tr, y_te = breast_cancer_data()
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                               eval_metric="logloss", random_state=42, verbosity=0)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    row("Model", "XGBClassifier (breast cancer)")
    row("Training accuracy", f"{acc:.4f}")
    row("Feature importance (top 3)", str(sorted(
        enumerate(model.feature_importances_), key=lambda x: -x[1])[:3]))

    path = "demo_models/cancer_xgb.ubj"
    model.save_model(path)
    row("Saved", path)

    mid, fmt = upload_model("demo_xgboost_cancer", path, "xgboost",
                            {"n_estimators": 100, "max_depth": 4})
    row("Uploaded to API", mid or str(fmt), "ok" if mid else "fail")

    preds = model.predict(X_te[:5])
    proba = model.predict_proba(X_te[:5])[:, 1].round(3)
    row("Predictions (first 5)", str(preds.tolist()))
    row("Probabilities (first 5)", str(proba.tolist()))
    row("Format detected", fmt or "N/A")
    section_result("XGBoost - Gradient Boosted", mid, fmt or "xgboost", True, f"acc={acc:.4f}")
    return mid


def demo_lightgbm():
    subheader("4. LightGBM (.lgb) - Light Gradient Boosted Trees")
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score

    X_tr, X_te, y_tr, y_te = wine_data()
    model = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, random_state=42, verbose=-1)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    row("Model", "LGBMClassifier (wine)")
    row("Training accuracy", f"{acc:.4f}")
    row("Number of trees built", str(model.n_estimators_))

    path = "demo_models/wine_lgbm.lgb"
    model.booster_.save_model(path)
    row("Saved", path)

    mid, fmt = upload_model("demo_lightgbm_wine", path, "lightgbm",
                            {"num_leaves": 15, "n_estimators": 100})
    row("Uploaded to API", mid or str(fmt), "ok" if mid else "fail")

    preds = model.predict(X_te[:5])
    row("Predictions (first 5)", str(preds.tolist()))
    row("Format detected", fmt or "N/A")
    section_result("LightGBM - Gradient Boosted", mid, fmt or "lightgbm", True, f"acc={acc:.4f}")
    return mid


def demo_onnx():
    subheader("5. ONNX (.onnx) - Open Neural Network Exchange")
    subheader("5. ONNX (.onnx) - Linear Classifier built natively")
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    X_tr, X_te, y_tr, y_te = iris_data()
    sk_model = LogisticRegression(max_iter=500, random_state=42)
    sk_model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, sk_model.predict(X_te))
    row("Base sklearn model", "LogisticRegression (iris)")
    row("Sklearn accuracy", f"{acc:.4f}")

    # Build ONNX graph manually using weights from sklearn model
    W = sk_model.coef_.astype(np.float32)        # (3, 4)
    b = sk_model.intercept_.astype(np.float32)   # (3,)
    W_init = numpy_helper.from_array(W, name="W")
    b_init = numpy_helper.from_array(b, name="b")
    gemm  = helper.make_node("Gemm",   ["X", "W", "b"], ["logits"], transB=1)
    amax  = helper.make_node("ArgMax", ["logits"], ["label"], axis=1)
    graph = helper.make_graph(
        [gemm, amax], "iris_logreg",
        [helper.make_tensor_value_info("X",     TensorProto.FLOAT, [None, 4])],
        [helper.make_tensor_value_info("label", TensorProto.INT64, [None])],
        initializer=[W_init, b_init]
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # onnx.checker skipped (thread-safety conflict with Flask)
    path = "demo_models/iris_logreg.onnx"
    onnx.save(onnx_model, path)
    row("ONNX model saved", path)
    row("ONNX opset", "17")
    row("Nodes", "Gemm + ArgMax")
    row("Sklearn predictions", str(sk_model.predict(X_te[:5]).tolist()))
    row("ONNX graph nodes",    "Gemm(W,b) -> ArgMax(axis=1)")
    row("Model size",          f"{onnx_model.ByteSize()} bytes")

    mid, fmt = upload_model("demo_onnx_logreg", path, "onnx",
                            {"built_from": "onnx graph API", "nodes": "Gemm+ArgMax"})
    row("Uploaded to API", mid or str(fmt), "ok" if mid else "fail")
    row("Format detected", fmt or "N/A")
    section_result("ONNX - LogisticRegression", mid, fmt or "onnx", True, f"acc={acc:.4f}")
    return mid


def demo_pickle_advanced():
    subheader("6. Pickle (.pkl) - GradientBoosting (sklearn)")
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report

    X_tr, X_te, y_tr, y_te = wine_data()
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    row("Model", "GradientBoostingClassifier (wine)")
    row("Accuracy", f"{acc:.4f}")
    row("Classes", str(model.classes_.tolist()))

    path = "demo_models/wine_gbt.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)

    mid, fmt = upload_model("demo_sklearn_gbt", path, "sklearn",
                            {"algorithm": "GradientBoosting", "n_estimators": 100})
    row("Uploaded to API", mid or str(fmt), "ok" if mid else "fail")
    preds = model.predict(X_te[:5])
    proba = model.predict_proba(X_te[:5]).round(3)
    row("Predictions (first 5)", str(preds.tolist()))
    row("Probabilities", str(proba.tolist()))
    row("Format detected", fmt or "N/A")
    section_result("Sklearn GradientBoosting", mid, fmt or "pickle", True, f"acc={acc:.4f}")
    return mid


def demo_joblib_pipeline():
    subheader("7. Joblib (.joblib) - Full ML Pipeline with Feature Engineering")
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import Ridge
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge(alpha=1.0))
    ])
    model.fit(X_tr, y_tr)
    r2 = r2_score(y_te, model.predict(X_te))
    row("Model", "Pipeline(Scaler + PolyFeatures + Ridge) (diabetes)")
    row("R2 score", f"{r2:.4f}")
    row("Pipeline steps", str([s[0] for s in model.steps]))

    path = "demo_models/diabetes_pipeline.joblib"
    joblib.dump(model, path)

    mid, fmt = upload_model("demo_regression_pipeline", path, "sklearn",
                            {"task": "regression", "dataset": "diabetes"})
    row("Uploaded to API", mid or str(fmt), "ok" if mid else "fail")
    preds = model.predict(X_te[:5]).round(1)
    row("Predictions (first 5)", str(preds.tolist()))
    row("Actual values (first 5)", str(y_te[:5].tolist()))
    row("Format detected", fmt or "N/A")
    section_result("Joblib Regression Pipeline", mid, fmt or "joblib", True, f"R2={r2:.4f}")
    return mid


def demo_xgboost_regressor():
    subheader("8. XGBoost (.ubj) - Regressor")
    import xgboost as xgb
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                              random_state=42, verbosity=0)
    model.fit(X_tr, y_tr)
    r2 = r2_score(y_te, model.predict(X_te))
    row("Model", "XGBRegressor (diabetes)")
    row("R2 score", f"{r2:.4f}")

    path = "demo_models/diabetes_xgb.ubj"
    model.save_model(path)

    mid, fmt = upload_model("demo_xgb_regressor", path, "xgboost",
                            {"task": "regression", "dataset": "diabetes"})
    row("Uploaded to API", mid or str(fmt), "ok" if mid else "fail")
    preds = model.predict(X_te[:5]).round(1)
    row("Predictions (first 5)", str(preds.tolist()))
    row("Format detected", fmt or "N/A")
    section_result("XGBoost Regressor", mid, fmt or "xgboost", True, f"R2={r2:.4f}")
    return mid


def demo_onnx_svm():
    subheader("9. ONNX (.onnx) - Support Vector Machine")
    subheader("9. ONNX (.onnx) - 2-Layer MLP Neural Network built natively")
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    X_tr, X_te, y_tr, y_te = breast_cancer_data()
    y_tr = y_tr.astype(int)
    y_te = y_te.astype(int)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=42)
    mlp.fit(X_tr_s, y_tr)
    acc = accuracy_score(y_te, mlp.predict(X_te_s))
    row("Base sklearn model", "MLPClassifier hidden=(32,) (breast cancer)")
    row("Sklearn accuracy", f"{acc:.4f}")

    # Build ONNX: X -> Gemm(W1,b1) -> Relu -> Gemm(W2,b2) -> ArgMax
    W1 = mlp.coefs_[0].T.astype(np.float32)
    b1 = mlp.intercepts_[0].astype(np.float32)
    W2 = mlp.coefs_[1].T.astype(np.float32)
    b2 = mlp.intercepts_[1].astype(np.float32)
    inits = [
        numpy_helper.from_array(W1, "W1"), numpy_helper.from_array(b1, "b1"),
        numpy_helper.from_array(W2, "W2"), numpy_helper.from_array(b2, "b2"),
    ]
    nodes = [
        helper.make_node("Gemm",   ["X",   "W1", "b1"], ["h1"],     transB=1),
        helper.make_node("Relu",   ["h1"],               ["h1r"]),
        helper.make_node("Gemm",   ["h1r", "W2", "b2"], ["logits"], transB=1),
        helper.make_node("ArgMax", ["logits"],           ["label"],  axis=1),
    ]
    graph = helper.make_graph(
        nodes, "mlp_breast_cancer",
        [helper.make_tensor_value_info("X",     TensorProto.FLOAT, [None, X_tr.shape[1]])],
        [helper.make_tensor_value_info("label", TensorProto.INT64, [None])],
        initializer=inits
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # onnx.checker skipped (thread-safety conflict with Flask)
    path = "demo_models/cancer_mlp.onnx"
    onnx.save(onnx_model, path)
    row("ONNX model saved", path)
    row("ONNX nodes", "Gemm -> Relu -> Gemm -> ArgMax")
    row("Sklearn predictions", str(mlp.predict(X_te_s[:5]).tolist()))
    row("Hidden layer size",   "32 neurons (ReLU activation)")
    row("Model size",          f"{onnx_model.ByteSize()} bytes")

    mid, fmt = upload_model("demo_onnx_mlp", path, "onnx",
                            {"architecture": "MLP-32", "built_from": "onnx graph API"})
    row("Uploaded to API", mid or str(fmt), "ok" if mid else "fail")
    row("Format detected", fmt or "N/A")
    section_result("ONNX - MLP Neural Net", mid, fmt or "onnx", True, f"acc={acc:.4f}")
    return mid


def demo_lightgbm_regressor():
    subheader("10. LightGBM (.lgb) - Regressor")
    import lightgbm as lgb
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(n_estimators=100, num_leaves=20, random_state=42, verbose=-1)
    model.fit(X_tr, y_tr)
    r2 = r2_score(y_te, model.predict(X_te))
    row("Model", "LGBMRegressor (diabetes)")
    row("R2 score", f"{r2:.4f}")

    path = "demo_models/diabetes_lgb.lgb"
    model.booster_.save_model(path)

    mid, fmt = upload_model("demo_lightgbm_regressor", path, "lightgbm",
                            {"task": "regression", "dataset": "diabetes"})
    row("Uploaded to API", mid or str(fmt), "ok" if mid else "fail")
    preds = model.predict(X_te[:5]).round(1)
    row("Predictions (first 5)", str(preds.tolist()))
    row("Format detected", fmt or "N/A")
    section_result("LightGBM Regressor", mid, fmt or "lightgbm", True, f"R2={r2:.4f}")
    return mid


def show_audit_trails(model_ids):
    header("AUDIT TRAIL SUMMARY")
    for name, mid in model_ids.items():
        if mid:
            trail = get_audit(mid)
            status = f"{len(trail)} event(s)" if trail else "0 events recorded"
            row(name[:40], status, "ok" if trail else "info")
            for ev in trail[:3]:
                ts = ev.get("timestamp", "?")[:19]
                etype = ev.get("event_type", "?")
                estatus = ev.get("status", "?")
                print(f"           [{ts}] {etype:<30} status={estatus}")


def show_registry():
    header("MODEL REGISTRY (all uploaded models)")
    try:
        r = requests.get(f"{API_BASE}/api/models", timeout=10)
        if r.status_code == 200:
            data = r.json()
            count = data.get("count", 0)
            models = data.get("data", [])
            print(f"  Total models in registry: {count}\n")
            print(f"  {'Name':<35} {'Format':<12} {'ID':<40}")
            print(f"  {'-'*35} {'-'*12} {'-'*36}")
            for m in models:
                name = m.get("name", "?")[:34]
                fmt = m.get("format", "?")[:11]
                mid = m.get("model_id", "?")[:36]
                print(f"  {name:<35} {fmt:<12} {mid:<36}")
    except Exception as e:
        print(f"  [FAIL] Could not reach registry: {e}")


def show_summary():
    header("FINAL DEPLOYMENT SUMMARY")
    print(f"\n  {'Model':<40} {'Format':<12} {'Score':<15} {'Status':<8}")
    print(f"  {'-'*40} {'-'*12} {'-'*15} {'-'*8}")
    for r in RESULTS:
        print(f"  {r['name']:<40} {r['format']:<12} {r['score']:<15} {r['status']:<8}")

    print(f"\n  Models deployed: {len(RESULTS)}")
    print(f"  Successful uploads: {PASS}")
    print(f"  Failed uploads:     {FAIL}")

    print(f"\n  Access Points:")
    print(f"  * Web UI:   http://127.0.0.1:5001")
    print(f"  * REST API: http://127.0.0.1:5001/api/models")
    print(f"  * MLFlow:   http://127.0.0.1:5000")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  MLOPS UNIFIED LAYER - SAMPLE MODEL DEPLOYMENT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check API (start embedded if needed)
    _ensure_api_running()
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        assert r.status_code == 200
        print(f"  [OK] API reachable at {API_BASE}")
    except Exception:
        print(f"\n  [FAIL] API not reachable at {API_BASE}.")
        sys.exit(1)

    # Check MLFlow
    try:
        r = requests.get("http://127.0.0.1:5000/health", timeout=5)
        print(f"  [OK] MLFlow reachable at http://127.0.0.1:5000")
    except:
        print("  [INFO] MLFlow server not detected (optional for this demo)")

    # Create output directory
    Path("demo_models").mkdir(exist_ok=True)
    print(f"  [OK] Output directory: demo_models/\n")

    model_ids = {}

    header("DEPLOYING SAMPLE MODELS")

    # sklearn / Pickle
    try:
        mid = demo_pickle()
        model_ids["Pickle - RandomForest"] = mid
    except Exception as e:
        print(f"  [FAIL] Pickle demo failed: {e}")

    # sklearn / Joblib
    try:
        mid = demo_joblib()
        model_ids["Joblib - SVM Pipeline"] = mid
    except Exception as e:
        print(f"  [FAIL] Joblib demo failed: {e}")

    # XGBoost classifier
    try:
        mid = demo_xgboost()
        model_ids["XGBoost - Classifier"] = mid
    except Exception as e:
        print(f"  [FAIL] XGBoost demo failed: {e}")

    # LightGBM classifier
    try:
        mid = demo_lightgbm()
        model_ids["LightGBM - Classifier"] = mid
    except Exception as e:
        print(f"  [FAIL] LightGBM demo failed: {e}")

    # ONNX - LogReg
    try:
        mid = demo_onnx()
        model_ids["ONNX - LogisticRegression"] = mid
    except Exception as e:
        print(f"  [FAIL] ONNX demo failed: {e}")

    # sklearn GBT
    try:
        mid = demo_pickle_advanced()
        model_ids["Sklearn GradientBoosting"] = mid
    except Exception as e:
        print(f"  [FAIL] GBT demo failed: {e}")

    # Joblib Regression Pipeline
    try:
        mid = demo_joblib_pipeline()
        model_ids["Joblib - Regression Pipeline"] = mid
    except Exception as e:
        print(f"  [FAIL] Joblib pipeline demo failed: {e}")

    # XGBoost Regressor
    try:
        mid = demo_xgboost_regressor()
        model_ids["XGBoost - Regressor"] = mid
    except Exception as e:
        print(f"  [FAIL] XGBoost regressor demo failed: {e}")

    # ONNX - SVM
    try:
        mid = demo_onnx_svm()
        model_ids["ONNX - SVM"] = mid
    except Exception as e:
        print(f"  [FAIL] ONNX SVM demo failed: {e}")

    # LightGBM Regressor
    try:
        mid = demo_lightgbm_regressor()
        model_ids["LightGBM - Regressor"] = mid
    except Exception as e:
        print(f"  [FAIL] LightGBM regressor demo failed: {e}")

    # Show results
    show_audit_trails(model_ids)
    show_registry()
    show_summary()
