# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------
# App config
# ---------------------------------------------------------
st.set_page_config(page_title="🚌 U.S. School Buses — Dashboard", page_icon="🚌", layout="wide")
st.title("🚌 U.S. School Buses — Dashboard Overview")

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def to_num(x):
    s = str(x).strip().lower()
    if s in ("unknown", "", "nan", "none"):
        return np.nan
    s = s.replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def fmt_int(n: int) -> str:
    return f"{int(n):,}".replace(",", " ")

def fmt_k(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M".replace(".", ",")
    if n >= 1_000:
        return f"{round(n/1_000):.0f}K"
    return str(n)

# ---------------------------------------------------------
# Tabs (PM2.5 first)
# ---------------------------------------------------------
tab_pm25, tab_gauge, tab_fuel, tab_age = st.tabs([
    "🌫️ PM2.5 KPI",
    "⚡ Electrification Gauge",
    "⛽ Fuel Types by State",
    "🕰️ Bus Age Distribution"
])

# =========================================================
# 🌫️ PM2.5 KPI — using District_level_data.csv
# =========================================================
with tab_pm25:
    st.subheader("🌫️ Air Pollution — PM2.5 Quartiles KPI (District Level)")

    csv_path = Path("District_level_data.csv")
    if not csv_path.exists():
        st.error("CSV not found: District_level_data.csv")
        st.stop()

    @st.cache_data
    def load_kpi_df(path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            path,
            engine="python",
            sep=";",              # semicolon-separated
            quotechar='"',
            escapechar="\\",
            encoding="latin1",
            on_bad_lines="warn"
        )
        col_total = "2a. Total number of buses"
        col_esb   = "3a. Number of ESBs committed "
        col_quart = "5g. Quartile: PM2.5 concentration"
        col_state = "1a. State"

        df["total_buses"]   = df[col_total].map(lambda v: np.nan if str(v).strip().lower() == "unknown" else to_num(v))
        df["esb_committed"] = df[col_esb].map(to_num)
        df["quartile"]      = df[col_quart].astype(str).str.strip().replace({"nan": "Unknown"})
        df["STATE_NORM"]    = (
            df[col_state].astype("string")
            .str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
        )
        return df

    def compute_distribution(df: pd.DataFrame):
        order = ["1.0", "2.0", "3.0", "4.0", "Unknown"]
        ref = df.groupby("quartile")["total_buses"].sum(min_count=1)
        esb = df.groupby("quartile")["esb_committed"].sum(min_count=1)
        ref = ref.reindex(order, fill_value=0).astype(float)
        esb = esb.reindex(order, fill_value=0).astype(float)
        ref_pct = 100 * ref / ref.sum() if ref.sum() > 0 else ref * 0
        esb_pct = 100 * esb / esb.sum() if esb.sum() > 0 else esb * 0
        return order, ref_pct.values, esb_pct.values

    def stacked_plot(order, ref_pct, esb_pct, state_name):
        colors = {
            "1.0": "#F2B233",
            "2.0": "#8E1E23",
            "3.0": "#3B5BA7",
            "4.0": "#4E9861",
            "Unknown": "#BFBFBF"
        }
        x_labels = ["Reference:\nAll school buses", "Committed ESBs"]
        fig = go.Figure()
        for i, q in enumerate(order):
            fig.add_bar(
                x=x_labels,
                y=[ref_pct[i], esb_pct[i]],
                name=q,
                marker=dict(color=colors[q]),
                text=[f"{ref_pct[i]:.0f}%", f"{esb_pct[i]:.0f}%"],
                texttemplate="%{text}",
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="white", size=12),
                hovertemplate=f"Quartile: {q}<br>% = %{{y:.1f}}%<extra></extra>"
            )
        fig.update_layout(
            barmode="stack",
            barnorm="percent",
            title=dict(
                text=f"Air Pollution by Quartile — {state_name.title()}<br>"
                     "<sup>School district average annual PM2.5 concentration</sup>",
                x=0.5
            ),
            yaxis=dict(title=None, showgrid=False, ticks="", range=[0, 100]),
            xaxis=dict(title=None),
            legend_title="PM2.5 quartile",
            margin=dict(l=30, r=30, t=80, b=40),
            height=640
        )
        return fig

    df_kpi = load_kpi_df(csv_path)
    states = sorted(df_kpi["STATE_NORM"].dropna().unique())
    default_idx = states.index("NEW YORK") if "NEW YORK" in states else 0
    selected_state = st.selectbox("Select a state:", states, index=default_idx)

    df_state = df_kpi[df_kpi["STATE_NORM"] == selected_state]
    if df_state.empty:
        st.info("No data for this state. Showing nationwide view.")
        df_state = df_kpi.copy()

    order, ref_pct, esb_pct = compute_distribution(df_state)
    st.plotly_chart(stacked_plot(order, ref_pct, esb_pct, selected_state), use_container_width=True)

    table = pd.DataFrame({
        "Quartile": order,
        "Reference %": [round(v, 1) for v in ref_pct],
        "Committed ESBs %": [round(v, 1) for v in esb_pct]
    })
    st.dataframe(table, use_container_width=True)

# =========================================================
# ⚡ Electrification Gauge (WRI June 2025)
# =========================================================
with tab_gauge:
    st.subheader("⚡ Electrification Progress — WRI June 2025")

    def fmt_int(n: int) -> str:
        return f"{int(n):,}".replace(",", " ")

    def fmt_k(n: int) -> str:
        """Affiche en K (ex: 12 000 → 12K)"""
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M".replace(".", ",")
        elif n >= 1_000:
            return f"{round(n/1_000):.0f}K"
        else:
            return str(n)

    @st.cache_data(show_spinner=False)
    def load_esb_totals(path: str):
        xls = pd.ExcelFile(path)
        df = pd.read_excel(xls, sheet_name="3. State-level data")
        total_col = "2a. Total number of school buses (WRI 2024)"
        electric_col = "3a. Number of committed ESBs"

        row = df.tail(1)[[total_col, electric_col]].iloc[0]
        total = pd.to_numeric(row[total_col], errors="coerce")
        electric = pd.to_numeric(row[electric_col], errors="coerce")

        if pd.isna(total) or total <= 0:
            total = pd.to_numeric(df[total_col], errors="coerce").sum(min_count=1)
        if pd.isna(electric) or electric < 0:
            electric = 0

        pct = float(electric / total) if total else 0.0   # 0..1
        return int(total), int(electric), pct

    excel_path = "ESB_adoption_dataset_v9_update_june_2025.xlsx"
    total, electric, pct = load_esb_totals(excel_path)

    FILL = "#ebebeb"   # gray arc
    TRACK = "#7f2326"   # red slice
    TXT   = "#212121"
    SUB   = "#d6d6d6ff"

    fig = plt.figure(figsize=(9, 4.6), dpi=90)
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))

    inner_r, outer_r = 0.60, 1.00
    theta_start, theta_end = 180, 0  # full top semicircle

    track = Wedge((0, 0), outer_r, theta_start, theta_end,
                width=outer_r - inner_r, facecolor=TRACK, edgecolor="none", zorder=1)
    ax.add_patch(track)

    span = (theta_start - theta_end) * pct
    theta1 = 240 + span
    theta2 = -170
    fill = Wedge((0, 0), outer_r, theta1, theta2,
                width=outer_r - inner_r, facecolor=FILL, edgecolor="none", zorder=2)
    ax.add_patch(fill)

    ax.text(0, -0.02, f"{pct*100:.1f}%".replace(".", ","), ha="center", va="center", fontsize=44, color=TXT)
    ax.text(-1.02, -0.25, fmt_k(electric), ha="left",  va="center", fontsize=13, color=SUB)
    ax.text(-1.02, -0.43, "Electric school buses", ha="left",  va="center", fontsize=11, color=SUB)
    ax.text( 1.02, -0.25, fmt_int(total),   ha="right", va="center", fontsize=13, color=SUB)
    ax.text( 1.02, -0.43, "Total U.S. school buses", ha="right", va="center", fontsize=11, color=SUB)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.20, 1.35)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

# =========================================================
# ⛽ Fuel Types by State
# =========================================================
with tab_fuel:
    st.subheader("⛽ School Bus Fuel Types by State")

    @st.cache_data
    def load_state_fleets():
        df = pd.read_excel(
            "Dataset of U.S. School Bus Fleets_State Summary_v3_2024-05-10.xlsx",
            sheet_name="State summary", header=1
        )

        # Normalize state names
        df["State"] = (
            df["State"].astype("string")
            .str.replace("*", "", regex=False)
            .str.strip().str.upper()
            .str.replace(r"\s+", " ", regex=True)
            .replace("WASHINGTON DC", "DISTRICT OF COLUMBIA")
        )
        df = df[~df["State"].isin(["TOTAL", "NAN", None])]

        # Fuel columns – handle variant names
        col_map = {}
        if "Other fuel" in df.columns and "Other" not in df.columns:
            col_map["Other fuel"] = "Other"
        df = df.rename(columns=col_map)

        fuels = ["Electric", "Diesel", "Gasoline", "Propane", "CNG", "Other"]
        # Keep only fuels that actually exist, drop missing ones from plotting
        fuels = [f for f in fuels if f in df.columns]

        # Coerce numerics and compute totals
        for c in fuels:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df["Total_Bus"] = df[fuels].sum(axis=1)

        # Sort by total buses
        df_sorted = df.sort_values(by="Total_Bus", ascending=False).reset_index(drop=True)
        return df_sorted, fuels

    fpath = "Dataset of U.S. School Bus Fleets_State Summary_v3_2024-05-10.xlsx"
    if not Path(fpath).exists():
        st.error("Excel not found: Dataset of U.S. School Bus Fleets_State Summary_v3_2024-05-10.xlsx")
    else:
        df_sorted, fuels = load_state_fleets()

        cmap = {
            "Electric": "#2ca02c",
            "Diesel": "#ff7f0e",
            "Gasoline": "#1f77b4",
            "Propane": "#d62728",
            "CNG": "#9467bd",
            "Other": "#8c564b"
        }

        st.header("Bus Fleet Distribution Across All States")

        fig_all = go.Figure()
        custom_cols = ["Total_Bus"] + fuels

        hover_all = "<b>State:</b> %{x}<br><b>Total Buses:</b> %{customdata[0]:,}<br>"
        for i, f in enumerate(fuels):
            hover_all += f"<b>{f}:</b> %{{customdata[{i+1}]:,}}<br>"
        hover_all += "<extra></extra>"

        # Ensure fixed category order = sorted by total buses
        category_order = df_sorted["State"].tolist()

        for f in fuels:
            fig_all.add_bar(
                x=df_sorted["State"],
                y=df_sorted[f],
                name=f,
                marker_color=cmap.get(f, "#999999"),
                customdata=df_sorted[custom_cols],
                hovertemplate=hover_all
            )

        fig_all.update_layout(
            barmode="stack",
            title_text="School bus fuel types by State",
            xaxis_title="",
            yaxis_title="Number of Buses",
            height=700,
            legend_title_text="<b>Fuel Type</b>",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_all.update_xaxes(
            categoryorder="array",
            categoryarray=category_order,
            tickangle=-45,
            tickfont=dict(size=10)
        )
        st.plotly_chart(fig_all, use_container_width=True)

        st.header("Detailed View for a Specific State")
        states = df_sorted["State"].tolist()
        idx = states.index("NEW YORK") if "NEW YORK" in states else 0
        pick_state = st.selectbox("Select a state:", states, index=idx)

        df_state = df_sorted[df_sorted["State"] == pick_state]
        fig_s = go.Figure()

        hover_s = "<b>State:</b> %{x}<br><b>Total Buses:</b> %{customdata[0]:,}<br><hr>"
        for i, f in enumerate(fuels):
            hover_s += f"<b>{f}:</b> %{{customdata[{i+1}]:,}}<br>"
        hover_s += "<extra></extra>"

        for f in fuels:
            fig_s.add_bar(
                x=[pick_state],
                y=df_state[f],
                name=f,
                marker_color=cmap.get(f, "#999999"),
                customdata=df_state[custom_cols],
                hovertemplate=hover_s
            )

        fig_s.update_layout(
            barmode="stack",
            title_text=f"School bus fuel types in {pick_state.title()}",
            xaxis_title="",
            yaxis_title="Number of Buses",
            legend_title_text="<b>Fuel Type</b>"
        )
        st.plotly_chart(fig_s, use_container_width=True)

# =========================================================
# 🕰️ Bus Age Distribution by State
# =========================================================
with tab_age:
    st.subheader("🕰️ School Bus Ages by State")


    file_path="Dataset of U.S. School Bus Fleets_State Summary_v3_2024-05-10.xlsx"
    df = pd.read_excel(file_path,header=1)

    age_cols = [
        'Number of buses 2020-newer',
        'Number of buses 2010-2019',
        'Number of buses 2000-2009',
        'Number of buses 1999 and older',
        'Number of buses with age unknown'
    ]

    for col in age_cols + ['Total number of buses']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df[age_cols[:-1]] = df[age_cols[:-1]].fillna(0)
    df['Sum_by_age_known'] = df[age_cols[:-1]].sum(axis=1)
    df['Number of buses with age unknown'] = df['Total number of buses'] - df['Sum_by_age_known']
    df.loc[df['Number of buses with age unknown'] < 0, 'Number of buses with age unknown'] = 0
    for col in age_cols + ['Total number of buses']:
        df[col] = (df[col] / 1000).round() * 1000

    df = df[df['State'].str.lower() != 'total']
    df = df.sort_values(by='Total number of buses', ascending=False).reset_index(drop=True)

    colors = {
        'Number of buses 2020-newer': '#FDB515',
        'Number of buses 2010-2019': '#33A1C9',
        'Number of buses 2000-2009': '#4169E1',
        'Number of buses 1999 and older': '#C21807',
        'Number of buses with age unknown': '#A9A9A9'
    }

    percent_total = [13, 47, 19, 3, 8]
    st.markdown(
        " | ".join([f"<span style='color:{colors[age_cols[i]]}; font-weight:bold'>{age_cols[i]}: {percent_total[i]}%</span>" for i in range(len(age_cols))]),
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(25,12))
    x = np.arange(len(df))
    bottom = np.zeros(len(df))

    for col in age_cols:
        ax.bar(x, df[col], width=0.8, bottom=bottom, color=colors[col], label=col)
        bottom += df[col]

    totals = df['Total number of buses'].values
    for i, total in enumerate(totals):
        ax.text(i, total * 1.01, f"{int(total/1000)}k", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(df['State'], rotation=45, ha='right')
    ax.set_xlabel('State')
    ax.set_ylabel('Number of Buses, by age')
    ax.set_title('School Bus Ages, by State', fontsize=14, fontweight='bold')

    legend_elements = [Patch(facecolor=colors[col], label=col) for col in age_cols]
    ax.legend(handles=legend_elements, title='Age Group', loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, totals.max() * 1.12)
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("📊 Show underlying data"):
        st.dataframe(df)

    df_ny = df[df['State'] == "New York"].reset_index(drop=True)
    percent_ny = [21, 66, 13, 0, 0]
    st.markdown(
        " | ".join([f"<span style='color:{colors[age_cols[i]]}; font-weight:bold'>{age_cols[i]}: {percent_ny[i]}%</span>" for i in range(len(age_cols))]),
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(12,8))
    x = np.arange(len(df_ny))
    bottom = np.zeros(len(df_ny))

    for col in age_cols:
        ax.bar(x, df_ny[col], width=0.8, bottom=bottom, color=colors[col], label=col)
        bottom += df_ny[col]

    totals = df_ny['Total number of buses'].values
    for i, total in enumerate(totals):
        ax.text(i, total * 1.01, f"{int(total/1000)}k", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(df_ny['State'], rotation=0, ha='center')
    ax.set_xlabel('State')
    ax.set_ylabel('Number of Buses, by Age Group')
    ax.set_title('School Bus Ages, New York', fontsize=14, fontweight='bold')
    legend_elements = [Patch(facecolor=colors[col], label=col) for col in age_cols]
    ax.legend(handles=legend_elements, title='Age Group', loc='upper right')
    ax.set_ylim(0, totals.max() * 1.12)
    plt.tight_layout()
    st.pyplot(fig)
