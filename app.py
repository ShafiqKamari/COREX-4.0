
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="COREX (COntribution via Removal and EXplained variability) FOR MCDM WEIGHTING METHOD",
    page_icon="🧭",
    layout="wide",
)

APP_TITLE = "COREX (COntribution via Removal and EXplained variability) FOR MCDM WEIGHTING METHOD"
CORE_TYPES = ["Benefit", "Cost", "Target"]

# -----------------------------
# COREX helpers
# -----------------------------
def normalize_column(x: pd.Series, ctype: str, target: float = None) -> pd.Series:
    xmin, xmax = x.min(), x.max()
    if ctype == "Benefit":
        r = (x - xmin) / (xmax - xmin)
    elif ctype == "Cost":
        r = (xmax - x) / (xmax - xmin)
    elif ctype == "Target":
        d = (x - target).abs()
        Dmax = d.max()
        r = 1.0 - d / Dmax
    else:
        raise ValueError("Unknown criterion type")
    return r.astype(float)

def corex_pipeline(df_vals: pd.DataFrame, crit_types: dict, targets: dict, alpha: float = 0.5):
    cols = list(df_vals.columns)
    n = len(cols)

    # Step 1
    R = pd.DataFrame(index=df_vals.index, columns=cols, dtype=float)
    for c in cols:
        t = crit_types[c]
        if t == "Target":
            R[c] = normalize_column(df_vals[c], t, targets.get(c, 0.0))
        else:
            R[c] = normalize_column(df_vals[c], t)

    # Step 2
    P = R.sum(axis=1) / n

    # Step 3
    row_sums = R.sum(axis=1)
    P_minus = pd.DataFrame(index=df_vals.index, columns=cols, dtype=float)
    for c in cols:
        P_minus[c] = (row_sums - R[c]) / n

    # Step 4
    D = pd.DataFrame(index=df_vals.index, columns=cols, dtype=float)
    for c in cols:
        D[c] = (P - P_minus[c]).abs()
    Rj = D.sum(axis=0).rename("RemovalImpact")

    # Step 5
    sigma = R.std(axis=0, ddof=1).rename("Sigma")

    # Step 6
    corr = R.corr(method="pearson").fillna(0.0)
    sum_abs_corr = corr.abs().sum(axis=1).rename("SumAbsCorr")

    # Step 7
    Vj = (sigma / sum_abs_corr).rename("ExplainedVariability")

    # Step 8
    Rbar = (Rj / Rj.sum()).rename("Rbar")
    Vbar = (Vj / Vj.sum()).rename("Vbar")
    W = (alpha * Rbar + (1.0 - alpha) * Vbar).rename("Weight")
    W = W / W.sum()

    summary = pd.concat([Rj, Vj, Rbar, Vbar, W], axis=1)
    summary.index.name = "Criterion"

    return {
        "R": R, "P": P, "P_minus": P_minus, "Rj": Rj, "sigma": sigma,
        "corr": corr, "sum_abs_corr": sum_abs_corr, "Vj": Vj, "W": W,
        "summary": summary
    }

def make_sample_dataset():
    data = {
        "Benefit1": [70, 85, 90, 60, 75],
        "Benefit2": [150, 140, 160, 155, 145],
        "Cost1": [200, 180, 220, 210, 190],
        "Cost2": [15, 12, 18, 14, 13],
        "Target1": [50, 55, 52, 48, 60],
    }
    return pd.DataFrame(data, index=[f"A{i+1}" for i in range(5)])

# -----------------------------
# UI
# -----------------------------
st.markdown(f"<h2 style='color:#0ea5e9;'>{APP_TITLE}</h2>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    alpha = st.slider("Blend parameter α", 0.0, 1.0, 0.5, 0.05)
    use_sample = st.checkbox("Use sample dataset", value=False)

raw_df = None
if use_sample:
    raw_df = make_sample_dataset()
else:
    if file is not None:
        if file.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(file)
        else:
            raw_df = pd.read_excel(file)

if raw_df is None:
    st.info("Upload a file or use the sample dataset.")
    st.stop()

st.subheader("Raw data")
st.dataframe(raw_df, use_container_width=True)

with st.expander("Row identifiers", expanded=False):
    idx_col = st.selectbox("Use this column as alternative names", ["<row number>"] + list(raw_df.columns), index=0)
if idx_col != "<row number>":
    raw_df = raw_df.set_index(idx_col)

num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
if not num_cols:
    st.error("No numeric columns found.")
    st.stop()

with st.expander("Select criteria columns", expanded=False):
    selected_cols = st.multiselect("Criteria", options=num_cols, default=num_cols)
if not selected_cols:
    st.error("Select at least one criterion.")
    st.stop()

df_vals = raw_df[selected_cols].copy()

st.subheader("Criterion types and targets")
if "crit_meta_pretty" not in st.session_state or set(st.session_state["crit_meta_pretty"].index) != set(selected_cols):
    st.session_state["crit_meta_pretty"] = pd.DataFrame(
        {"Type": ["Benefit"]*len(selected_cols), "Target": [0.0]*len(selected_cols)},
        index=selected_cols
    )

meta = st.data_editor(
    st.session_state["crit_meta_pretty"],
    column_config={
        "Type": st.column_config.SelectboxColumn(options=CORE_TYPES),
        "Target": st.column_config.NumberColumn(format="%.6f"),
    },
    use_container_width=True,
)
st.session_state["crit_meta_pretty"] = meta
crit_types = meta["Type"].to_dict()
targets = meta["Target"].astype(float).to_dict()

# Compute button
if st.button("🚀 Compute COREX"):
    A = corex_pipeline(df_vals, crit_types, targets, alpha=alpha)

    tabs = st.tabs([
        "① Normalization of the Decision Matrix",
        "② The Overall Performance Score",
        "③ The Performance Score under Criterion Removal",
        "④ Removal Impact Score",
        "⑤ The Standard Deviation of Each Criterion",
        "⑥ The Sum of Absolute Correlations for Each Criterion",
        "⑦ Explained Variability Score",
        "⑧ COREX Weight Scores",
    ])

    with tabs[0]:
        st.dataframe(A["R"].style.background_gradient(cmap="Blues").format("{:.6f}"), use_container_width=True)

    with tabs[1]:
        st.dataframe(A["P"].to_frame("P").style.format("{:.6f}"), use_container_width=True)

    with tabs[2]:
        st.dataframe(A["P_minus"].style.format("{:.6f}"), use_container_width=True)

    with tabs[3]:
        st.dataframe(A["Rj"].to_frame().style.bar(color="#0ea5e9").format("{:.6f}"), use_container_width=True)

    with tabs[4]:
        st.dataframe(A["sigma"].to_frame().style.bar(color="#22c55e").format("{:.6f}"), use_container_width=True)

    with tabs[5]:
        st.dataframe(A["sum_abs_corr"].to_frame().style.bar(color="#f59e0b").format("{:.6f}"), use_container_width=True)

    with tabs[6]:
        st.dataframe(A["Vj"].to_frame().style.bar(color="#6366f1").format("{:.6f}"), use_container_width=True)

    with tabs[7]:
        st.dataframe(A["W"].to_frame("Weight").style.bar(color="#14b8a6").format("{:.6f}"), use_container_width=True)
        fig = px.bar(A["W"].reset_index(), x="index", y="Weight")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary table")
    st.dataframe(A["summary"].style.format("{:.6f}"), use_container_width=True)
