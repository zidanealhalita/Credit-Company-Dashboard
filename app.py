"""
Multifinance Loan Process Intelligence Dashboard
=================================================
Streamlit app for mapping the end-to-end auto-loan approval process and
surfacing department-level improvement opportunities from operational
timestamp logs.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_pipeline import (
    STAGE_META, HOPS, SLA_TARGET_HOURS,
    load_raw, engineer_features, apply_filters, fmt_idr, fmt_hours,
)
from ml_engine import train_duration_driver_model, train_outcome_model, flag_anomalies
import ui_kit as ui

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multifinance Loan Process Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()
ui.set_plotly_theme()

# ---------------------------------------------------------------------------
# Sidebar — data source + global filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚘 Process Intelligence")
    st.caption("Dashboard analitik proses pembiayaan kendaraan")

    with st.expander("📁 Sumber data", expanded=False):
        uploaded = st.file_uploader("Ganti dataset (opsional, .csv)", type=["csv"])
        st.caption("Jika kosong, dashboard memakai `data/multifinance_loan_process_logs.csv` bawaan repo.")

    try:
        raw_df = load_raw(uploaded.getvalue() if uploaded else None)
        df = engineer_features(raw_df)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.markdown("---")
    st.markdown("**Filter Data**")

    min_date, max_date = df["Submission_Date"].min(), df["Submission_Date"].max()
    date_range = st.slider(
        "Rentang tanggal pengajuan",
        min_value=min_date, max_value=max_date, value=(min_date, max_date),
    )

    regions = st.multiselect(
        "Wilayah Cabang", sorted(df["Branch_Region"].unique()),
        default=sorted(df["Branch_Region"].unique()),
    )
    asset_types = st.multiselect(
        "Jenis Aset", sorted(df["Asset_Type"].unique()),
        default=sorted(df["Asset_Type"].unique()),
    )
    customer_types = st.multiselect(
        "Jenis Nasabah", sorted(df["Customer_Type"].unique()),
        default=sorted(df["Customer_Type"].unique()),
    )
    statuses = st.multiselect(
        "Status Aplikasi", sorted(df["Application_Status"].unique()),
        default=sorted(df["Application_Status"].unique()),
    )

    st.markdown("---")
    st.caption(
        f"SLA target siklus penuh (asumsi analisis): **{SLA_TARGET_HOURS:.0f} jam**. "
        "Dapat disesuaikan sesuai kebijakan internal perusahaan."
    )
    st.caption("Dibuat dengan Streamlit · Data operasional Jan–Mar 2026")

fdf = apply_filters(df, date_range, regions, asset_types, customer_types, statuses)

if fdf.empty:
    st.warning("Tidak ada data pada kombinasi filter ini. Silakan ubah filter di sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
        <div class="kicker">Loan Operations · T1 → T6 Workflow</div>
        <div class="app-title">Peta Proses & Peluang Perbaikan Pembiayaan Kendaraan</div>
        <div class="app-subtitle">
            Menelusuri {len(fdf):,} aplikasi pembiayaan dari pengajuan hingga pencairan untuk
            memetakan alur bisnis end-to-end, menemukan titik penyumbatan (bottleneck) proses,
            dan merekomendasikan perbaikan yang terukur per departemen.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_flow, tab_bottleneck, tab_ml, tab_reco, tab_explore = st.tabs(
    ["📊 Ringkasan", "🔄 Peta Alur Proses", "🔍 Analisis Bottleneck", "🤖 Insight ML", "🎯 Rekomendasi", "🗂️ Data Explorer"]
)

# =============================================================================
# TAB 1 — RINGKASAN EKSEKUTIF
# =============================================================================
with tab_overview:
    approved = fdf[fdf["Is_Approved"]]
    n_total = len(fdf)
    approval_rate = fdf["Is_Approved"].mean() * 100
    rejection_rate = fdf["Is_Rejected"].mean() * 100
    cancellation_rate = fdf["Is_Cancelled"].mean() * 100
    avg_e2e = approved["E2E_Hours"].mean()
    total_disbursed = approved["Loan_Amount_IDR"].sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        ui.kpi_card("Total Aplikasi", f"{n_total:,}")
    with c2:
        ui.kpi_card("Tingkat Persetujuan", f"{approval_rate:.1f}%", "dari seluruh aplikasi", "good")
    with c3:
        ui.kpi_card("Tingkat Penolakan", f"{rejection_rate:.1f}%", None, "bad")
    with c4:
        ui.kpi_card("Tingkat Pembatalan", f"{cancellation_rate:.1f}%", None, "neutral")
    with c5:
        ui.kpi_card("Rata² Siklus Penuh", f"{avg_e2e:.1f} jam", f"vs SLA {SLA_TARGET_HOURS:.0f} jam",
                    "bad" if avg_e2e > SLA_TARGET_HOURS else "good")
    with c6:
        ui.kpi_card("Total Dana Tersalurkan", fmt_idr(total_disbursed))

    st.write("")
    st.markdown("##### Alur Proses Sekilas — rata-rata waktu antar tahap")
    hop_means = [fdf[h["id"]].mean() for h in HOPS]
    ui.process_rail(STAGE_META, hop_means)
    st.caption(
        "🟢 cepat · 🟠 perlu perhatian · 🔴 bottleneck — lebar & warna batang proporsional terhadap rata-rata durasi jam."
    )

    st.markdown("---")
    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("##### Tren Volume Harian & Tingkat Persetujuan")
        daily = fdf.groupby("Submission_Date").agg(
            volume=("Application_ID", "count"),
            approval_rate=("Is_Approved", "mean"),
        ).reset_index()
        daily["ma7_volume"] = daily["volume"].rolling(7, min_periods=1).mean()
        daily["approval_rate_pct"] = daily["approval_rate"] * 100

        fig = go.Figure()
        fig.add_bar(x=daily["Submission_Date"], y=daily["volume"], name="Volume harian",
                    marker_color=ui.BRAND, opacity=0.35)
        fig.add_scatter(x=daily["Submission_Date"], y=daily["ma7_volume"], name="Rata² bergerak 7 hari",
                        line=dict(color=ui.BRAND_DARK, width=2.5))
        fig.add_scatter(x=daily["Submission_Date"], y=daily["approval_rate_pct"], name="Tingkat persetujuan (%)",
                        yaxis="y2", line=dict(color=ui.AMBER, width=2, dash="dot"))
        fig.update_layout(
            height=380,
            yaxis=dict(title="Jumlah aplikasi"),
            yaxis2=dict(title="Tingkat persetujuan (%)", overlaying="y", side="right", range=[0, 100], showgrid=False),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, width='stretch')

    with right:
        st.markdown("##### Distribusi Status Aplikasi")
        status_counts = fdf["Application_Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Jumlah"]
        color_map = {"Approved": ui.FAST, "Rejected": ui.SLOW, "Cancelled": ui.AMBER}
        fig = px.pie(status_counts, names="Status", values="Jumlah", hole=0.55,
                     color="Status", color_discrete_map=color_map)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("##### Volume per Wilayah")
        reg = fdf["Branch_Region"].value_counts().reset_index()
        reg.columns = ["Wilayah", "Jumlah"]
        fig = px.bar(reg, x="Jumlah", y="Wilayah", orientation="h", color_discrete_sequence=[ui.BRAND])
        fig.update_layout(height=300, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, width='stretch')
    with b2:
        st.markdown("##### Volume per Jenis Aset")
        asset = fdf["Asset_Type"].value_counts().reset_index()
        asset.columns = ["Jenis Aset", "Jumlah"]
        fig = px.bar(asset, x="Jumlah", y="Jenis Aset", orientation="h", color_discrete_sequence=[ui.BRAND])
        fig.update_layout(height=300, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, width='stretch')
    with b3:
        st.markdown("##### Nilai Pinjaman Rata-rata per Jenis Nasabah")
        cust = fdf.groupby("Customer_Type")["Loan_Amount_IDR"].mean().reset_index()
        cust.columns = ["Jenis Nasabah", "Rata² Pinjaman"]
        fig = px.bar(cust, x="Jenis Nasabah", y="Rata² Pinjaman", color_discrete_sequence=[ui.BRAND])
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch')

# =============================================================================
# TAB 2 — PEMETAAN ALUR PROSES BISNIS
# =============================================================================
with tab_flow:
    st.markdown("### Pemetaan Alur Proses Bisnis End-to-End")
    st.markdown(
        "Setiap aplikasi yang masuk melewati **6 titik waktu (T1–T6)** yang dipetakan ke "
        "**5 departemen** berbeda. Diagram di bawah menskalakan lebar & warna tiap ruas "
        "sesuai rata-rata durasi aktualnya — sehingga bottleneck langsung terlihat tanpa membaca angka."
    )

    hop_means_full = [fdf[h["id"]].mean() for h in HOPS]
    ui.process_rail(STAGE_META, hop_means_full)

    st.markdown("")
    st.markdown("##### Tabel Ringkasan per Tahap Proses")
    rows = []
    for hop in HOPS:
        s = fdf[hop["id"]].dropna()
        rows.append({
            "Tahap": hop["label"],
            "Departemen Pemilik": hop["dept"],
            "Rata-rata (jam)": round(s.mean(), 2),
            "Median (jam)": round(s.median(), 2),
            "P90 (jam)": round(s.quantile(0.9), 2),
            "Maks (jam)": round(s.max(), 2),
        })
    summary_tbl = pd.DataFrame(rows)
    st.dataframe(summary_tbl, width='stretch', hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("##### Sankey: Alur Volume Aplikasi dari Pengajuan hingga Keputusan Akhir")
        n = len(fdf)
        n_app = int(fdf["Is_Approved"].sum())
        n_rej = int(fdf["Is_Rejected"].sum())
        n_can = int(fdf["Is_Cancelled"].sum())

        labels = [
            "T1 Pengajuan", "T2 Verifikasi Dokumen", "T3 Analisis Kredit",
            "T4 Survei Lapangan", "T5 Persetujuan Akhir",
            "✅ Disetujui & Dicairkan", "❌ Ditolak", "⏸️ Dibatalkan",
        ]
        node_colors = [ui.BRAND, ui.BRAND, ui.BRAND, ui.BRAND, ui.BRAND, ui.FAST, ui.SLOW, ui.AMBER]
        fig = go.Figure(go.Sankey(
            node=dict(label=labels, color=node_colors, pad=18, thickness=16,
                      line=dict(color="white", width=0.5)),
            link=dict(
                source=[0, 1, 2, 3, 4, 4, 4],
                target=[1, 2, 3, 4, 5, 6, 7],
                value=[n, n, n, n, n_app, n_rej, n_can],
                color=["rgba(20,92,100,0.25)"] * 4 + [
                    "rgba(76,140,74,0.35)", "rgba(193,68,45,0.35)", "rgba(216,161,58,0.35)",
                ],
            ),
        ))
        fig.update_layout(height=420, font_size=12)
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Catatan: dalam data ini, seluruh aplikasi melewati kelima tahap proses secara penuh — "
            "tidak ada jalur keluar (short-circuit) di tengah alur. Percabangan hasil (disetujui / "
            "ditolak / dibatalkan) baru terjadi di gerbang akhir setelah T5."
        )

    with col2:
        st.markdown("##### Sebaran Durasi per Ruas Proses (jam)")
        melt_cols = [h["id"] for h in HOPS]
        melt_labels = {h["id"]: h["label"] for h in HOPS}
        long_df = fdf[melt_cols].melt(var_name="Ruas", value_name="Jam").dropna()
        long_df["Ruas"] = long_df["Ruas"].map(melt_labels)
        fig = px.box(long_df, x="Jam", y="Ruas", color="Ruas", points=False,
                     color_discrete_sequence=ui.CHART_COLORWAY)
        fig.update_layout(height=420, showlegend=False, yaxis=dict(categoryorder="array",
                          categoryarray=[melt_labels[h["id"]] for h in HOPS][::-1]))
        st.plotly_chart(fig, width='stretch')

    st.info(
        "💡 **Insight utama**: ruas **Analisis Kredit → Survei Lapangan** jauh lebih lambat dan lebih "
        "bervariasi dibanding ruas lainnya (lihat kotak/whisker yang jauh lebih panjang). Ini adalah "
        "target perbaikan proses dengan dampak terbesar — dibahas lebih lanjut di tab **Analisis Bottleneck**."
    )

# =============================================================================
# TAB 3 — ANALISIS BOTTLENECK & DEPARTEMEN
# =============================================================================
with tab_bottleneck:
    st.markdown("### Analisis Bottleneck & Cakupan Perbaikan Departemen")

    hop_avg = {h["id"]: fdf[h["id"]].mean() for h in HOPS}
    bottleneck_hop = max(hop_avg, key=hop_avg.get)
    bottleneck_label = next(h["label"] for h in HOPS if h["id"] == bottleneck_hop)
    bottleneck_dept = next(h["dept"] for h in HOPS if h["id"] == bottleneck_hop)
    baseline_other = np.mean([v for k, v in hop_avg.items() if k != bottleneck_hop])
    multiple = hop_avg[bottleneck_hop] / baseline_other if baseline_other else float("nan")

    st.markdown(
        f"""
        <div class="section-card">
        <span class="pill slow">BOTTLENECK UTAMA</span>
        <span class="rec-title" style="display:inline;">{bottleneck_label}</span>
        <p style="margin-top:0.6rem; color:{ui.MUTED};">
        Rata-rata durasi ruas ini adalah <b>{hop_avg[bottleneck_hop]:.1f} jam</b> — sekitar
        <b>{multiple:.1f}x lebih lama</b> dibanding rata-rata ruas proses lainnya
        ({baseline_other:.1f} jam). Ruas ini berada di bawah tanggung jawab
        <b>{bottleneck_dept}</b>.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Rata-rata Durasi per Ruas Proses")
    hop_df = pd.DataFrame([
        {"Ruas": h["label"], "Jam": fdf[h["id"]].mean(), "id": h["id"]} for h in HOPS
    ])
    hop_df["Warna"] = hop_df["id"].apply(lambda x: ui.SLOW if x == bottleneck_hop else ui.BRAND)
    fig = px.bar(hop_df, x="Ruas", y="Jam", text_auto=".1f")
    fig.update_traces(marker_color=hop_df["Warna"])
    fig.update_layout(height=340)
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.markdown(f"##### Peta Panas: Durasi *{bottleneck_label}* per Wilayah × Jenis Aset")
    pivot = fdf.pivot_table(index="Branch_Region", columns="Asset_Type", values=bottleneck_hop, aggfunc="mean")
    fig = px.imshow(
        pivot, text_auto=".1f", aspect="auto",
        color_continuous_scale=[[0, "#E7F0E6"], [0.5, ui.AMBER], [1, ui.SLOW]],
        labels=dict(color="Jam"),
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Berdasarkan Jenis Aset")
        seg = fdf.groupby("Asset_Type")[bottleneck_hop].mean().sort_values(ascending=False).reset_index()
        seg.columns = ["Jenis Aset", "Jam"]
        fig = px.bar(seg, x="Jam", y="Jenis Aset", orientation="h", text_auto=".1f",
                     color_discrete_sequence=[ui.SLOW])
        fig.update_layout(height=280, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.markdown("##### Berdasarkan Wilayah Cabang")
        seg2 = fdf.groupby("Branch_Region")[bottleneck_hop].mean().sort_values(ascending=False).reset_index()
        seg2.columns = ["Wilayah", "Jam"]
        fig = px.bar(seg2, x="Jam", y="Wilayah", orientation="h", text_auto=".1f",
                     color_discrete_sequence=[ui.SLOW])
        fig.update_layout(height=280, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.markdown("##### Kepatuhan SLA Siklus Penuh (Submission → Disbursement)")
    approved_fdf = fdf[fdf["Is_Approved"]]
    if len(approved_fdf) > 0:
        breach_rate = approved_fdf["SLA_Breach"].mean() * 100
        c1, c2 = st.columns([1, 2])
        with c1:
            ui.kpi_card("Aplikasi Melebihi SLA", f"{breach_rate:.1f}%",
                       f"dari {len(approved_fdf):,} aplikasi disetujui", "bad" if breach_rate > 40 else "neutral")
        with c2:
            breach_by_asset = approved_fdf.groupby("Asset_Type")["SLA_Breach"].mean().sort_values(ascending=False) * 100
            fig = px.bar(breach_by_asset.reset_index(), x="Asset_Type", y="SLA_Breach",
                        labels={"SLA_Breach": "% Melebihi SLA", "Asset_Type": "Jenis Aset"},
                        text_auto=".1f", color_discrete_sequence=[ui.SLOW])
            fig.update_layout(height=260)
            st.plotly_chart(fig, width='stretch')

    st.success(
        "💡 **Temuan kunci**: keterlambatan terkonsentrasi pada dua pola yang independen — "
        "(1) **appraisal mobil bekas** yang secara konsisten lebih lama di **semua wilayah**, dan "
        "(2) **kapasitas tim survei di Jawa Timur** yang lambat untuk **semua jenis aset**, bukan hanya "
        "mobil bekas. Keduanya perlu solusi berbeda — lihat tab **Rekomendasi**."
    )

# =============================================================================
# TAB 4 — INSIGHT MACHINE LEARNING
# =============================================================================
with tab_ml:
    st.markdown("### Insight Machine Learning")
    st.caption(
        "Model dilatih pada data terfilter saat ini untuk memvalidasi temuan secara statistik "
        "dan mendeteksi anomali operasional yang butuh tindak lanjut."
    )

    if len(fdf) < 300:
        st.warning("Data terfilter terlalu sedikit untuk melatih model yang andal. Perluas filter di sidebar.")
    else:
        st.markdown("#### 1️⃣ Model Diagnosis Penyebab Keterlambatan")
        st.caption(
            f"RandomForestRegressor memprediksi durasi *{bottleneck_label}* (jam) dari "
            "segmen aplikasi (jenis nasabah, jenis aset, wilayah, nilai pinjaman, hari pengajuan) — "
            "tanpa memakai durasi tahap lain, supaya tidak ada kebocoran informasi."
        )
        driver_result = train_duration_driver_model(fdf, target_col=bottleneck_hop)
        colA, colB = st.columns([1, 2])
        with colA:
            ui.kpi_card("R² Model (data uji)", f"{driver_result['r2']:.2f}",
                       f"n = {driver_result['n']:,} aplikasi", "good" if driver_result["r2"] > 0.5 else "neutral")
            st.caption(
                "R² tinggi berarti segmen aplikasi (bukan faktor acak harian) memang menjelaskan "
                "sebagian besar variasi lamanya proses ini."
            )
        with colB:
            top_feat = driver_result["importances"].head(8).sort_values()
            fig = px.bar(top_feat, orientation="h", labels={"value": "Tingkat Kepentingan", "index": ""},
                        color_discrete_sequence=[ui.SLOW])
            fig.update_layout(height=320, showlegend=False, title="Fitur Paling Berpengaruh")
            st.plotly_chart(fig, width='stretch')

        st.markdown("---")
        st.markdown("#### 2️⃣ Apakah Keputusan Akhir Bisa Diprediksi dari Segmen/Proses?")
        st.caption(
            "RandomForestClassifier mencoba memprediksi status akhir (Disetujui/Ditolak/Dibatalkan) "
            "hanya dari fitur segmen — untuk menguji apakah keputusan dipengaruhi proses/segmen, "
            "atau murni faktor kelayakan kredit individual di luar data ini."
        )
        outcome_result = train_outcome_model(fdf)
        colC, colD = st.columns([1, 2])
        with colC:
            ui.kpi_card("Akurasi Model", f"{outcome_result['accuracy']*100:.1f}%",
                       f"vs baseline tebak mayoritas {outcome_result['baseline_acc']*100:.1f}%",
                       "neutral")
            gap = (outcome_result["accuracy"] - outcome_result["baseline_acc"]) * 100
            if gap < 5:
                st.info(
                    "📌 Model **hampir tidak lebih baik** dari sekadar menebak kelas mayoritas. "
                    "Ini mengindikasikan keputusan akhir **tidak dijelaskan oleh segmen atau kecepatan "
                    "proses** — kemungkinan besar ditentukan oleh faktor kelayakan kredit individual "
                    "(riwayat, penghasilan, dsb.) yang tidak tercatat di log operasional ini."
                )
            else:
                st.info(f"Model mengungguli baseline sebesar {gap:.1f} poin persentase akurasi.")
        with colD:
            top_feat2 = outcome_result["importances"].head(8).sort_values()
            fig = px.bar(top_feat2, orientation="h", labels={"value": "Tingkat Kepentingan", "index": ""},
                        color_discrete_sequence=[ui.BRAND])
            fig.update_layout(height=320, showlegend=False, title="Fitur Paling Berpengaruh")
            st.plotly_chart(fig, width='stretch')

        st.markdown("---")
        st.markdown("#### 3️⃣ Daftar Anomali: Aplikasi Lebih Lambat dari Ekspektasi Segmennya")
        st.caption(
            "Deteksi outlier berbasis z-score *relatif terhadap segmen* (kombinasi wilayah × jenis aset), "
            "sehingga yang tertangkap adalah kasus yang lambat **melebihi baseline segmennya sendiri** — "
            "kandidat prioritas untuk audit operasional, bukan hanya yang lambat secara umum."
        )
        z_thresh = st.slider("Ambang z-score anomali", 1.0, 1.8, 1.4, 0.05)
        flagged = flag_anomalies(fdf, hop_col=bottleneck_hop, z_thresh=z_thresh)
        st.write(f"**{len(flagged):,} aplikasi** terdeteksi sebagai anomali dari {len(fdf):,} total.")
        show_cols = ["Application_ID", "Customer_Type", "Asset_Type", "Branch_Region",
                     bottleneck_hop, "_z", "Application_Status"]
        display_flagged = flagged[show_cols].rename(columns={
            bottleneck_hop: "Durasi (jam)", "_z": "Z-score", "Application_ID": "ID Aplikasi",
            "Customer_Type": "Jenis Nasabah", "Asset_Type": "Jenis Aset", "Branch_Region": "Wilayah",
            "Application_Status": "Status",
        }).round(2)
        st.dataframe(display_flagged.head(50), width='stretch', hide_index=True)
        st.download_button(
            "⬇️ Unduh daftar anomali lengkap (CSV)",
            display_flagged.to_csv(index=False).encode("utf-8"),
            file_name="anomali_proses_survei.csv", mime="text/csv",
        )

# =============================================================================
# TAB 5 — REKOMENDASI PERBAIKAN DEPARTEMEN
# =============================================================================
with tab_reco:
    st.markdown("### Rekomendasi Perbaikan Departemen")
    st.caption(
        "Estimasi dampak dihitung dari data terfilter saat ini, membandingkan rata-rata durasi "
        "segmen bermasalah terhadap baseline segmen yang sehat (mobil baru/truk di luar Jawa Timur)."
    )

    used_car = fdf[fdf["Asset_Type"] == "Used Passenger Car"]
    non_used = fdf[fdf["Asset_Type"] != "Used Passenger Car"]
    jatim = fdf[fdf["Branch_Region"] == "Jawa Timur"]
    healthy = fdf[(fdf["Asset_Type"] != "Used Passenger Car") & (fdf["Branch_Region"] != "Jawa Timur")]

    baseline = healthy[bottleneck_hop].mean() if len(healthy) else fdf[bottleneck_hop].mean()

    used_other_region = fdf[(fdf["Asset_Type"] == "Used Passenger Car") & (fdf["Branch_Region"] != "Jawa Timur")]
    jatim_non_used = fdf[(fdf["Asset_Type"] != "Used Passenger Car") & (fdf["Branch_Region"] == "Jawa Timur")]
    jatim_used = fdf[(fdf["Asset_Type"] == "Used Passenger Car") & (fdf["Branch_Region"] == "Jawa Timur")]

    def extra_hours(segment_df):
        if len(segment_df) == 0 or pd.isna(baseline):
            return 0.0, 0
        seg_mean = segment_df[bottleneck_hop].mean()
        return max(0, (seg_mean - baseline)) * len(segment_df), len(segment_df)

    extra_used, n_used = extra_hours(used_other_region)
    extra_jatim, n_jatim = extra_hours(jatim_non_used)
    extra_both, n_both = extra_hours(jatim_used)
    total_extra_hours = extra_used + extra_jatim + extra_both

    st.markdown(
        f"""
        <div class="section-card">
        <span class="kicker">Dampak Agregat Bottleneck (periode terfilter)</span>
        <p style="font-family:{ui.FONT_DISPLAY}; font-size:1.6rem; font-weight:700; margin:0.3rem 0;">
        ≈ {total_extra_hours:,.0f} jam kerja tambahan ({total_extra_hours/24:,.0f} hari) tersita
        akibat dua pola bottleneck di atas.
        </p>
        <p style="color:{ui.MUTED};">Setara waktu tunggu ekstra bagi {n_used + n_jatim + n_both:,} nasabah
        yang bisa dipangkas jika kedua isu ditangani.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Rencana Aksi per Departemen")

    st.markdown(
        f"""
        <div class="rec-card urgent">
            <span class="rec-dept">SURVEYOR / APPRAISAL — PRIORITAS TINGGI</span>
            <div class="rec-title">Standardisasi SOP Appraisal Mobil Bekas</div>
            <span class="pill slow">+{used_car[bottleneck_hop].mean()-baseline:.1f} jam/aplikasi</span>
            <span class="pill amber">{n_used:,} aplikasi terdampak</span>
            <p style="margin-top:0.6rem;">
            Survei lapangan untuk <b>mobil bekas</b> berlangsung jauh lebih lama dibanding mobil baru/truk
            di <b>semua wilayah</b> — pola yang konsisten dan sistemik, bukan kebetulan regional.
            Kemungkinan penyebab: proses inspeksi kondisi fisik & estimasi nilai pasar yang belum
            distandarkan/didigitalkan.
            </p>
            <p><b>Aksi yang disarankan:</b></p>
            <ul>
                <li>Buat <i>checklist</i> inspeksi digital terstandar (foto wajib per titik, kondisi mesin,
                riwayat servis) agar surveyor tidak menunggu proses manual berulang.</li>
                <li>Sediakan alat bantu estimasi harga pasar mobil bekas (basis data harga acuan) agar
                penilaian tidak bergantung negosiasi/verifikasi manual berkepanjangan.</li>
                <li>Pertimbangkan surveyor khusus/terlatih untuk kategori mobil bekas dengan target waktu
                selayaknya mobil baru.</li>
            </ul>
            <p><b>Potensi dampak:</b> memangkas ~{extra_used:,.0f} jam ({extra_used/24:,.0f} hari) waktu
            tunggu kumulatif pada periode ini.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="rec-card urgent">
            <span class="rec-dept">OPERASIONAL CABANG JAWA TIMUR — PRIORITAS TINGGI</span>
            <div class="rec-title">Tinjau Kapasitas Tim Survei Regional</div>
            <span class="pill slow">+{jatim[bottleneck_hop].mean()-baseline:.1f} jam/aplikasi</span>
            <span class="pill amber">{n_jatim + n_both:,} aplikasi terdampak</span>
            <p style="margin-top:0.6rem;">
            Berbeda dari isu mobil bekas, di wilayah <b>Jawa Timur</b> keterlambatan terjadi pada
            <b>seluruh jenis aset</b> — termasuk mobil baru & truk yang biasanya cepat diproses di
            wilayah lain. Pola ini menunjuk pada <b>keterbatasan kapasitas/jumlah surveyor</b> atau
            <b>backlog penjadwalan</b> di cabang tersebut, bukan kompleksitas appraisal.
            </p>
            <p><b>Aksi yang disarankan:</b></p>
            <ul>
                <li>Audit rasio jumlah surveyor aktif terhadap volume aplikasi masuk di Jawa Timur
                dibanding wilayah lain.</li>
                <li>Evaluasi opsi penambahan personel sementara atau realokasi dari wilayah dengan
                kapasitas lebih longgar.</li>
                <li>Terapkan sistem antrian/penjadwalan survei berbasis prioritas agar aplikasi
                tidak menumpuk tanpa visibilitas.</li>
            </ul>
            <p><b>Potensi dampak:</b> memangkas ~{extra_jatim + extra_both:,.0f} jam
            ({(extra_jatim + extra_both)/24:,.0f} hari) waktu tunggu kumulatif pada periode ini.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    d5_mean = fdf["D5"].mean()
    st.markdown(
        f"""
        <div class="rec-card">
            <span class="rec-dept">FINANCE / TREASURY — PRIORITAS MENENGAH</span>
            <div class="rec-title">Percepat Gerbang Persetujuan → Pencairan Dana</div>
            <span class="pill amber">rata-rata {d5_mean:.1f} jam</span>
            <p style="margin-top:0.6rem;">
            Tahap ini relatif seragam di semua segmen (bukan bottleneck struktural), namun tetap
            merupakan ruas kedua terlama secara keseluruhan (~{d5_mean:.0f} jam). Ada peluang efisiensi
            inkremental tanpa perlu perombakan proses besar.
            </p>
            <p><b>Aksi yang disarankan:</b></p>
            <ul>
                <li>Evaluasi apakah pencairan bisa diproses same-day (T+0) untuk aplikasi yang disetujui
                sebelum jam potong tertentu, memakai kanal transfer instan.</li>
                <li>Otomatisasi rekonsiliasi dokumen pencairan agar tidak menunggu proses batch harian.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="rec-card">
            <span class="rec-dept">CREDIT ANALYST & CREDIT COMMITTEE — PERTAHANKAN</span>
            <div class="rec-title">Tahap Ini Sudah Efisien — Jadikan Acuan Benchmark Internal</div>
            <span class="pill fast">konsisten di semua segmen</span>
            <p style="margin-top:0.6rem;">
            Verifikasi dokumen, analisis kredit, dan persetujuan akhir menunjukkan durasi yang stabil
            dan seragam di semua wilayah/jenis aset/jenis nasabah — tidak ada indikasi bottleneck.
            Praktik kerja di tahap-tahap ini layak dijadikan referensi standar saat mendesain ulang
            SOP tim survei.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### Matriks Prioritas: Dampak vs. Kompleksitas Implementasi")
    priority_data = pd.DataFrame([
        {"Inisiatif": "SOP Digital Appraisal Mobil Bekas", "Dampak (jam dihemat)": extra_used,
         "Kompleksitas": 6, "Departemen": "Surveyor/Appraisal"},
        {"Inisiatif": "Tambah Kapasitas Survei Jawa Timur", "Dampak (jam dihemat)": extra_jatim + extra_both,
         "Kompleksitas": 4, "Departemen": "Ops Regional"},
        {"Inisiatif": "Pencairan Dana Same-Day (T+0)", "Dampak (jam dihemat)": len(fdf) * max(0, d5_mean - 4),
         "Kompleksitas": 7, "Departemen": "Finance/Treasury"},
    ])
    fig = px.scatter(
        priority_data, x="Kompleksitas", y="Dampak (jam dihemat)", size="Dampak (jam dihemat)",
        color="Departemen", text="Inisiatif", size_max=60,
        color_discrete_sequence=[ui.SLOW, ui.AMBER, ui.BRAND],
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        height=420, xaxis=dict(title="Kompleksitas Implementasi (1=mudah, 10=sulit)", range=[0, 10]),
        showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Skor kompleksitas bersifat kualitatif/estimasi awal untuk diskusi prioritas, bukan hasil "
        "perhitungan model. Kuadran kiri-atas (dampak tinggi, kompleksitas rendah) adalah kandidat "
        "*quick win* yang layak dieksekusi lebih dulu."
    )

# =============================================================================
# TAB 6 — DATA EXPLORER
# =============================================================================
with tab_explore:
    st.markdown("### Data Explorer")
    st.caption("Jelajahi data mentah beserta fitur turunan (durasi tiap tahap) sesuai filter aktif di sidebar.")
    display_cols = [
        "Application_ID", "Customer_Type", "Asset_Type", "Branch_Region", "Loan_Amount_IDR",
        "Application_Status", "D1", "D2", "D3", "D4", "D5", "E2E_Hours",
    ]
    st.dataframe(fdf[display_cols].round(2), width='stretch', hide_index=True, height=460)
    st.download_button(
        "⬇️ Unduh data terfilter (CSV)",
        fdf[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="multifinance_filtered_data.csv", mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Dashboard analitik ini dibangun di atas log operasional proses pembiayaan kendaraan "
    "(20.000 aplikasi, Jan–Mar 2026). Ambang SLA & skor kompleksitas bersifat asumsi analitis "
    "untuk mendukung diskusi perbaikan proses, dan dapat disesuaikan dengan kebijakan aktual perusahaan."
)
