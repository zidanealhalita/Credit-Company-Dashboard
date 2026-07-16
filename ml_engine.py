"""
ml_engine.py
------------
Lightweight ML diagnostics for the loan process dashboard:

1. Driver model — a RandomForestRegressor explaining *what* predicts a long
   Credit-Analysis-to-Field-Survey duration (the operational bottleneck),
   using only features a process owner actually controls (region, asset
   type, customer type, loan size, timing) — no other-stage leakage.
2. Outcome model — a RandomForestClassifier testing whether Approved /
   Rejected / Cancelled is predictable from process/segment features at all.
   A low score here is itself an insight (decision is policy/credit driven).
3. Anomaly flagging — percentile-based outlier detection on end-to-end
   cycle time, surfaced as a worklist for operations follow-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLS = ["Customer_Type", "Asset_Type", "Branch_Region", "Loan_Amount_IDR", "Submission_DOW"]
CAT_COLS = ["Customer_Type", "Asset_Type", "Branch_Region", "Submission_DOW"]
NUM_COLS = ["Loan_Amount_IDR"]


def _build_matrix(df: pd.DataFrame):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_matrix = enc.fit_transform(df[CAT_COLS])
    cat_names = enc.get_feature_names_out(CAT_COLS)
    X = np.hstack([cat_matrix, df[NUM_COLS].to_numpy(dtype=float)])
    names = list(cat_names) + NUM_COLS
    return X, names


@st.cache_resource(show_spinner="Melatih model diagnosis penyebab keterlambatan...")
def train_duration_driver_model(df: pd.DataFrame, target_col: str = "D3"):
    """Explain variance in a stage-hop duration using segment/context features."""
    work = df.dropna(subset=[target_col]).copy()
    X, names = _build_matrix(work)
    y = work[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))

    importances = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
    return {"model": model, "importances": importances, "r2": r2, "n": len(work)}


@st.cache_resource(show_spinner="Menguji prediktabilitas keputusan akhir...")
def train_outcome_model(df: pd.DataFrame):
    """Test whether Approved/Rejected/Cancelled is predictable from segment features."""
    work = df.copy()
    X, names = _build_matrix(work)
    y = work["Application_Status"].astype(str).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    # Baseline: majority-class naive accuracy for comparison
    baseline_acc = pd.Series(y_test).value_counts(normalize=True).max()

    importances = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
    return {
        "model": model,
        "importances": importances,
        "accuracy": acc,
        "f1": f1,
        "baseline_acc": baseline_acc,
        "n": len(work),
    }


def flag_anomalies(df: pd.DataFrame, hop_col: str = "D3", z_thresh: float = 1.4) -> pd.DataFrame:
    """Flag applications whose bottleneck-hop duration is a statistical outlier
    *within their own segment* (Asset_Type x Branch_Region), so we catch cases
    that are slow even relative to their expected baseline, not just globally."""
    work = df.dropna(subset=[hop_col]).copy()
    grp = work.groupby(["Asset_Type", "Branch_Region"])[hop_col]
    seg_mean = grp.transform("mean")
    seg_std = grp.transform("std").replace(0, np.nan)
    work["_z"] = (work[hop_col] - seg_mean) / seg_std
    flagged = work[work["_z"] > z_thresh].copy()
    flagged = flagged.sort_values("_z", ascending=False)
    return flagged
