"""
data_pipeline.py
-----------------
Loading, cleaning, and feature engineering for the multifinance loan
process log dashboard. Kept separate from app.py so the Streamlit layer
stays focused on presentation.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Constants — the business process definition.
# This is the single source of truth for how raw timestamp columns map to
# real workflow stages / department owners. Update here if the process
# changes and the rest of the app follows automatically.
# ---------------------------------------------------------------------------

STAGE_COLUMNS = [
    "T1_Submission",
    "T2_Doc_Verification",
    "T3_Credit_Analysis",
    "T4_Field_Survey",
    "T5_Final_Approval",
    "T6_Disbursement",
]

STAGE_META = [
    {"code": "T1", "col": "T1_Submission", "label": "Pengajuan",
     "dept": "Sales / Front Office", "desc": "Aplikasi masuk dari dealer atau nasabah."},
    {"code": "T2", "col": "T2_Doc_Verification", "label": "Verifikasi Dokumen",
     "dept": "Back Office / Admin", "desc": "Kelengkapan & keabsahan dokumen diperiksa."},
    {"code": "T3", "col": "T3_Credit_Analysis", "label": "Analisis Kredit",
     "dept": "Credit Analyst", "desc": "Penilaian kelayakan kredit & skor risiko."},
    {"code": "T4", "col": "T4_Field_Survey", "label": "Survei Lapangan",
     "dept": "Surveyor / Appraisal", "desc": "Verifikasi fisik agunan & kunjungan lapangan."},
    {"code": "T5", "col": "T5_Final_Approval", "label": "Persetujuan Akhir",
     "dept": "Credit Committee", "desc": "Keputusan akhir: disetujui / ditolak."},
    {"code": "T6", "col": "T6_Disbursement", "label": "Pencairan Dana",
     "dept": "Finance / Treasury", "desc": "Dana dicairkan ke rekening nasabah / dealer."},
]

# Hop definitions: (id, from_col, to_col, from_code, to_code, label, owning dept for the *gap*)
HOPS = [
    {"id": "D1", "src": "T1_Submission", "dst": "T2_Doc_Verification",
     "label": "Pengajuan → Verifikasi Dokumen", "dept": "Back Office / Admin"},
    {"id": "D2", "src": "T2_Doc_Verification", "dst": "T3_Credit_Analysis",
     "label": "Verifikasi Dokumen → Analisis Kredit", "dept": "Credit Analyst"},
    {"id": "D3", "src": "T3_Credit_Analysis", "dst": "T4_Field_Survey",
     "label": "Analisis Kredit → Survei Lapangan", "dept": "Surveyor / Appraisal"},
    {"id": "D4", "src": "T4_Field_Survey", "dst": "T5_Final_Approval",
     "label": "Survei Lapangan → Persetujuan Akhir", "dept": "Credit Committee"},
    {"id": "D5", "src": "T5_Final_Approval", "dst": "T6_Disbursement",
     "label": "Persetujuan Akhir → Pencairan Dana", "dept": "Finance / Treasury"},
]

# Illustrative internal SLA target for total submission-to-disbursement time.
# This is an assumption made for the purpose of this analysis (not provided
# by the source data) — flagged clearly wherever it's used in the app.
SLA_TARGET_HOURS = 24.0

DATA_PATH_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "data", "multifinance_loan_process_logs.csv"),
    "data/multifinance_loan_process_logs.csv",
    "multifinance_loan_process_logs.csv",
]


@st.cache_data(show_spinner="Memuat data proses pinjaman...")
def load_raw(uploaded_bytes: bytes | None = None) -> pd.DataFrame:
    """Load the raw CSV either from an uploaded file or the bundled data/ path."""
    if uploaded_bytes is not None:
        import io
        df = pd.read_csv(io.BytesIO(uploaded_bytes))
    else:
        path = None
        for candidate in DATA_PATH_CANDIDATES:
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(
                "File data tidak ditemukan. Pastikan "
                "'data/multifinance_loan_process_logs.csv' ada di repo, "
                "atau unggah file melalui sidebar."
            )
        df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner="Memproses & menghitung durasi tiap tahap...")
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps and derive all duration / calendar features used app-wide."""
    df = df.copy()
    for col in STAGE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Hop durations (hours)
    for hop in HOPS:
        df[hop["id"]] = (df[hop["dst"]] - df[hop["src"]]).dt.total_seconds() / 3600

    # End-to-end cycle time (only meaningful for disbursed / Approved rows)
    df["E2E_Hours"] = (df["T6_Disbursement"] - df["T1_Submission"]).dt.total_seconds() / 3600

    # Elapsed time up to the final decision point (T5) — meaningful for ALL rows,
    # including Rejected/Cancelled, since T1-T5 are always populated.
    df["Time_To_Decision_Hours"] = (
        df["T5_Final_Approval"] - df["T1_Submission"]
    ).dt.total_seconds() / 3600

    # Calendar features
    df["Submission_Date"] = df["T1_Submission"].dt.date
    df["Submission_Month"] = df["T1_Submission"].dt.to_period("M").astype(str)
    df["Submission_Week"] = df["T1_Submission"].dt.to_period("W").astype(str)
    df["Submission_DOW"] = df["T1_Submission"].dt.day_name()
    df["Submission_Hour"] = df["T1_Submission"].dt.hour

    # Convenience flags
    df["Is_Approved"] = df["Application_Status"].eq("Approved")
    df["Is_Rejected"] = df["Application_Status"].eq("Rejected")
    df["Is_Cancelled"] = df["Application_Status"].eq("Cancelled")
    df["SLA_Breach"] = df["E2E_Hours"] > SLA_TARGET_HOURS

    return df


def apply_filters(
    df: pd.DataFrame,
    date_range: tuple,
    regions: list,
    asset_types: list,
    customer_types: list,
    statuses: list,
) -> pd.DataFrame:
    """Apply the global sidebar filter selection to the engineered dataframe."""
    mask = (
        (df["Submission_Date"] >= date_range[0])
        & (df["Submission_Date"] <= date_range[1])
        & (df["Branch_Region"].isin(regions))
        & (df["Asset_Type"].isin(asset_types))
        & (df["Customer_Type"].isin(customer_types))
        & (df["Application_Status"].isin(statuses))
    )
    return df.loc[mask].copy()


def fmt_idr(value: float, compact: bool = True) -> str:
    """Format a rupiah value in a readable Indonesian-finance convention."""
    if pd.isna(value):
        return "-"
    if not compact:
        return f"Rp {value:,.0f}"
    if abs(value) >= 1e12:
        return f"Rp {value / 1e12:,.2f} T"
    if abs(value) >= 1e9:
        return f"Rp {value / 1e9:,.2f} M"
    if abs(value) >= 1e6:
        return f"Rp {value / 1e6:,.1f} Jt"
    return f"Rp {value:,.0f}"


def fmt_hours(value: float) -> str:
    if pd.isna(value):
        return "-"
    if value >= 24:
        days = value / 24
        return f"{value:,.1f} jam (~{days:,.1f} hari)"
    return f"{value:,.1f} jam"
