import streamlit as st
import joblib, json, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report
)
from typing import Dict, Any, Tuple
import io

MODELS_DIR = Path(__file__).parent / "models"

EXPECTED_COLS = [
    'Time', 'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14',
    'V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount'
]

st.set_page_config(page_title="Fraud Models Dashboard", layout="wide")

# ----------------------- Helpers ----------------------- #
def safe_load(path: Path):
    """Load a model catching missing dependency errors."""
    try:
        return joblib.load(path)
    except ModuleNotFoundError as e:
        st.warning(f"Skip {path.name}: {e}")
        return None
    except Exception as e:
        st.warning(f"Failed {path.name}: {e}")
        return None

def ensure_feature_names(X, reference_model):
    """Return DataFrame with feature names if available."""
    if isinstance(X, pd.DataFrame):
        return X
    cols = None
    for attr in ["feature_names_in_", "feature_name_", "columns"]:
        if hasattr(reference_model, attr):
            candidate = getattr(reference_model, attr)
            if isinstance(candidate, (list, tuple, np.ndarray)):
                cols = list(candidate)
                break
    if cols is None:
        # Fallback to generic names
        cols = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols)

@st.cache_resource
def load_all() -> Tuple[Dict[str, Any], pd.DataFrame, pd.Series, Dict[str, Any], pd.DataFrame]:
    models: Dict[str, Any] = {}
    mapping = [
        ("logreg", "fraud_logreg_baseline.pkl"),
        ("logreg_smote", "fraud_logreg_smote.pkl"),
        ("rf", "fraud_rf.pkl"),
        ("xgb", "fraud_xgb.pkl"),
        ("lgbm", "fraud_lgbm.pkl"),
        ("lgbm_pipeline", "fraud_lgbm_pipeline.pkl"),
        ("cat", "fraud_cat.pkl"),
        ("stack", "fraud_stack.pkl"),
    ]
    for name, fname in mapping:
        fp = MODELS_DIR / fname
        if fp.exists():
            obj = safe_load(fp)
            if obj is not None:
                models[name] = obj
    data = joblib.load(MODELS_DIR / "test_data.pkl")
    X_test, y_test = data["X_test"], data["y_test"]
    # Normalize to DataFrame with feature names
    if models:
        ref_model = next(iter(models.values()))
        X_test = ensure_feature_names(X_test, ref_model)
    metrics: Dict[str, Any] = {}
    mpath = MODELS_DIR / "metrics.json"
    if mpath.exists():
        with open(mpath) as f:
            metrics = json.load(f)
    rf_imp_path = MODELS_DIR / "rf_feature_importance.csv"
    rf_imp = pd.read_csv(rf_imp_path) if rf_imp_path.exists() else pd.DataFrame()
    return models, X_test, y_test, metrics, rf_imp

models, X_test, y_test, saved_metrics, rf_imp = load_all()

st.title("Fraud Detection Models")

if not models:
    st.error("No models found in /models")
    st.stop()

# -------------------- SIDEBAR CONTROLS -------------------- #
st.sidebar.header("⚙️ Model Settings")
model_name = st.sidebar.selectbox("Select Model", list(models.keys()), index=0)
thr = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01)

with st.sidebar.expander("🎯 Advanced Options"):
    auto_f2 = st.button("Find F2 threshold")
    scan_cost = st.button("Find min cost threshold")
    cost_fp = st.number_input("FP cost", 0, 5000, 5)
    cost_fn = st.number_input("FN cost", 0, 50000, 500)

# -------------------- USER DATA UPLOAD -------------------- #
st.sidebar.markdown("---")
st.sidebar.markdown("### 📤 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (30 features)", type=['csv'], help="Upload a CSV file with transaction data")

if uploaded_file is not None:
    with st.spinner("Processing uploaded data..."):
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✓ Loaded {len(df_upload)} rows")
            
            # Check and prepare columns
            missing_cols = [c for c in EXPECTED_COLS if c not in df_upload.columns]
            
            # Prepare data
            df_processed = df_upload[[c for c in EXPECTED_COLS if c in df_upload.columns]].copy()
            for col in missing_cols:
                df_processed[col] = 0.0
            df_processed = df_processed[EXPECTED_COLS]
            
            # Apply scaling if scaler exists and not using pipeline
            if 'pipeline' not in model_name:
                scaler_path = MODELS_DIR / 'scaler.pkl'
                if scaler_path.exists():
                    scaler_obj = safe_load(scaler_path)
                    if scaler_obj is not None:
                        df_processed[['Amount', 'Time']] = scaler_obj.transform(df_processed[['Amount', 'Time']])
            
            # Predict
            selected_model = models[model_name]
            upload_probs = selected_model.predict_proba(df_processed)[:, 1]
            upload_preds = (upload_probs >= thr).astype(int)
            
            fraud_count = upload_preds.sum()
            
            st.success(f"✅ **Analysis Complete** | Frauds: {fraud_count}/{len(df_processed)} ({fraud_count/len(df_processed)*100:.1f}%)")
            
            result_df = pd.DataFrame({
                'probability': upload_probs,
                'prediction': upload_preds,
                'status': upload_preds.map({0: '✓ Normal', 1: '⚠️ Fraud'})
            })
            
            # Show only first 100 rows for performance
            if len(result_df) > 100:
                st.info(f"Showing first 100 of {len(result_df)} rows")
                st.dataframe(result_df.head(100), width='stretch')
            else:
                st.dataframe(result_df, width='stretch')
            
            # Download button
            csv_output = result_df.to_csv(index=False)
            st.download_button(
                label="💾 Download All Predictions",
                data=csv_output,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# -------------------- SINGLE TRANSACTION PREDICTION -------------------- #
with st.expander("🔍 Single Transaction Prediction"):
    st.markdown("Enter transaction details for instant fraud prediction")
    
    col_a, col_b = st.columns(2)
    input_time = col_a.number_input('Time (seconds)', value=0.0, step=1.0, key='single_time')
    input_amount = col_b.number_input('Amount ($)', value=100.0, min_value=0.0, step=10.0, key='single_amount')
    
    show_all_features = st.checkbox("Show all V-features", value=False, key='show_v')
    
    transaction_data = {'Time': input_time, 'Amount': input_amount}
    
    if show_all_features:
        st.markdown("**V-features (PCA components):**")
        v_features = [c for c in EXPECTED_COLS if c.startswith('V')]
        v_cols = st.columns(4)
        for i, v_feat in enumerate(v_features):
            col_idx = i % 4
            transaction_data[v_feat] = v_cols[col_idx].number_input(
                v_feat, value=0.0, step=0.1, format="%.4f", key=f'v_{v_feat}'
            )
    else:
        for v_feat in [c for c in EXPECTED_COLS if c.startswith('V')]:
            transaction_data[v_feat] = 0.0
    
    if st.button("🎯 Predict Transaction", type="primary", key='predict_btn'):
        with st.spinner("Analyzing transaction..."):
            # Create DataFrame
            single_df = pd.DataFrame([transaction_data])[EXPECTED_COLS]
            
            # Apply scaling if needed
            if 'pipeline' not in model_name:
                scaler_path = MODELS_DIR / 'scaler.pkl'
                if scaler_path.exists():
                    scaler_obj = safe_load(scaler_path)
                    if scaler_obj is not None:
                        single_df[['Amount', 'Time']] = scaler_obj.transform(single_df[['Amount', 'Time']])
            
            # Predict
            single_model = models[model_name]
            single_prob = single_model.predict_proba(single_df)[:, 1][0]
            single_pred = int(single_prob >= thr)
            
            # Display result
            col1, col2 = st.columns(2)
            with col1:
                if single_pred == 1:
                    st.error(f"🚨 **FRAUD DETECTED**")
                else:
                    st.success(f"✅ **NORMAL TRANSACTION**")
            with col2:
                st.metric("Fraud Probability", f"{single_prob*100:.2f}%")
            
            st.progress(single_prob, text=f"Confidence: {single_prob:.4f}")

st.markdown("---")

# Get model once
model = models[model_name]

# Compute predictions with spinner
with st.spinner(f"Computing metrics for {model_name}..."):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= thr).astype(int)

    roc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    cost_current = fp * cost_fp + fn * cost_fn

# Display metrics
st.subheader(f"📈 Test Set Performance: {model_name.upper()}")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("ROC AUC", f"{roc:.3f}")
c2.metric("PR AUC", f"{pr_auc:.3f}")
c3.metric("Precision", f"{precision:.3f}")
c4.metric("Recall", f"{recall:.3f}")
c5.metric("F1", f"{f1:.3f}")
c6.metric("Cost", f"{int(cost_current)}")

# Threshold optimization buttons
if auto_f2 or scan_cost:
    with st.spinner("Optimizing threshold..."):
        if auto_f2:
            prec_curve, rec_curve, thr_curve = precision_recall_curve(y_test, probs)
            thr_curve = np.append(thr_curve, 1.0)
            beta = 2
            f2_vals = (1 + beta**2) * prec_curve * rec_curve / (beta**2 * prec_curve + rec_curve + 1e-9)
            idx = int(np.argmax(f2_vals))
            st.success(f"✓ F2 threshold: {thr_curve[idx]:.4f} | P={prec_curve[idx]:.3f} R={rec_curve[idx]:.3f} F2={f2_vals[idx]:.3f}")

        if scan_cost:
            def cost_for_thr(p, t):
                pred_local = (p >= t).astype(int)
                cm_local = confusion_matrix(y_test, pred_local)
                tn_l, fp_l, fn_l, tp_l = cm_local.ravel()
                return fp_l * cost_fp + fn_l * cost_fn
            grid = np.linspace(0.01, 0.99, 50)
            costs = [(t, cost_for_thr(probs, t)) for t in grid]
            best_cost_thr, min_cost_val = min(costs, key=lambda x: x[1])
            st.info(f"✓ Min cost threshold: {best_cost_thr:.4f} | Cost={int(min_cost_val)}")

st.subheader("Confusion Matrix")
cm_df = pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"])
st.table(cm_df)

# Visualization tabs
with st.expander("📊 Performance Curves", expanded=False):
    roc_tab, pr_tab = st.tabs(["ROC Curve", "Precision-Recall"])
    
    with roc_tab:
        fpr, tpr, _ = roc_curve(y_test, probs)
        st.line_chart(pd.DataFrame({"FPR": fpr, "TPR": tpr}), x="FPR", y="TPR")
    
    with pr_tab:
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs)
        st.line_chart(pd.DataFrame({"Recall": rec_curve, "Precision": prec_curve}), x="Recall", y="Precision")

# Model comparison - lazy load
with st.expander("📋 All Models Comparison", expanded=False):
    if st.button("🔄 Compute Summary", key='compute_summary'):
        with st.spinner("Computing metrics for all models..."):
            summary_rows = []
            for m, mdl in models.items():
                p = mdl.predict_proba(X_test)[:, 1]
                fixed_pred = (p >= 0.5).astype(int)
                cmf = confusion_matrix(y_test, fixed_pred)
                tn2, fp2, fn2, tp2 = cmf.ravel()
                prec2 = tp2 / (tp2 + fp2 + 1e-9)
                rec2 = tp2 / (tp2 + fn2 + 1e-9)
                f12 = 2 * prec2 * rec2 / (prec2 + rec2 + 1e-9)
                summary_rows.append({
                    "model": m,
                    "roc_auc": roc_auc_score(y_test, p),
                    "pr_auc": average_precision_score(y_test, p),
                    "precision": prec2,
                    "recall": rec2,
                    "f1": f12
                })
            st.dataframe(pd.DataFrame(summary_rows).sort_values("pr_auc", ascending=False), width='stretch')

if model_name == "rf" and not rf_imp.empty:
    st.subheader("RF Feature Importance (Top 20)")
    if "feature" in rf_imp.columns and "importance" in rf_imp.columns:
        st.bar_chart(rf_imp.head(20).set_index("feature"))
    else:
        st.write("Feature importance file format mismatch.")

if st.sidebar.button("Export test predictions.csv"):
    pd.DataFrame({"prob": probs, "pred": preds, "actual": y_test}).to_csv(MODELS_DIR / "predictions_export.csv", index=False)
    st.sidebar.success("Saved to models/predictions_export.csv")

st.markdown("---")
st.caption("⚠️ Version warnings appear because models were trained with sklearn 1.6.1 but loaded with 1.7.2. To remove: retrain models in current environment.")