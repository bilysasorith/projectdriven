import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
import plotly.graph_objects as go

st.set_page_config(page_title="U.S. School Buses — Dashboard", page_icon="🚌", layout="wide")

st.title("🚌 U.S. School Buses — Dashboard Overview")


# =========================================================
# 1️⃣  ELECTRIFICATION GAUGE (WRI 2025)
# =========================================================
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
# 3️⃣  FUEL TYPES BY STATE (PLOTLY)
# =========================================================
st.subheader("⛽ School Bus Fuel Types by State")

@st.cache_data
def load_data():
    df = pd.read_excel(
        "Dataset of U.S. School Bus Fleets_State Summary_v3_2024-05-10.xlsx",
        sheet_name="State summary", header=1
    )
    df['State'] = df['State'].astype(str).str.strip().str.upper()
    df = df.rename(columns={'Other fuel': 'Other'})
    df['State'] = df['State'].str.replace('*', '', regex=False)
    df['State'] = df['State'].replace('WASHINGTON DC', 'DISTRICT OF COLUMBIA')
    rows_to_exclude = ['TOTAL', 'NAN']
    df = df[~df['State'].isin(rows_to_exclude)]
    
    all_fuel_types = ['Electric', 'Diesel', 'Gasoline', 'Propane', 'CNG', 'Other']
    df_clean = df[['State'] + all_fuel_types].copy()

    for col in all_fuel_types:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

    df_clean['Total_Bus'] = df_clean[all_fuel_types].sum(axis=1)
    df_sorted = df_clean.sort_values(by='Total_Bus', ascending=False)
    return df_sorted

df_sorted = load_data()
all_fuel_types = ['Electric', 'Diesel', 'Gasoline', 'Propane', 'CNG', 'Other']
color_map = {
    'Electric': '#2ca02c', 'Diesel': '#ff7f0e', 'Propane': '#d62728',
    'Gasoline': '#1f77b4', 'CNG': '#9467bd', 'Other': '#8c564b'
}

st.header("Bus Fleet Distribution Across All States")

fig_all_states = go.Figure()
hover_template_all = "<b>State:</b> %{x}<br><b>Total Buses:</b> %{customdata[0]:,}<br>"
display_names = {'Electric': 'Electric Buses'}
custom_data_cols = ['Total_Bus'] + all_fuel_types

for i, fuel in enumerate(all_fuel_types):
    display_name = display_names.get(fuel, fuel)
    hover_template_all += f"<b>{display_name}:</b> %{{customdata[{i+1}]:,}}<br>"
hover_template_all += "<extra></extra>"

for fuel_type in all_fuel_types:
    display_name = display_names.get(fuel_type, fuel_type)
    fig_all_states.add_trace(go.Bar(
        name=display_name,
        x=df_sorted['State'],
        y=df_sorted[fuel_type],
        marker_color=color_map.get(fuel_type),
        customdata=df_sorted[custom_data_cols],
        hovertemplate=hover_template_all
    ))

fig_all_states.update_layout(
    barmode='stack',
    title_text="School bus fuel types by State",
    xaxis_title="",
    yaxis_title="Number of Buses",
    height=600,
    xaxis={'categoryorder':'total descending'},
    legend_title_text='<b>Fuel Type</b>',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_all_states, use_container_width=True)


st.header("Detailed View for a Specific State")

state_list = df_sorted['State'].unique().tolist()
default_index = state_list.index('NEW YORK') if 'NEW YORK' in state_list else 0
selected_state = st.selectbox('Select a state:', state_list, index=default_index)

df_state = df_sorted[df_sorted['State'] == selected_state].copy()
fig_state = go.Figure()

hover_template_state = "<b>State:</b> %{x}<br><b>Total Buses:</b> %{customdata[0]:,}<br><hr>"
for i, fuel in enumerate(all_fuel_types):
    display_name = display_names.get(fuel, fuel)
    hover_template_state += f"<b>{display_name}:</b> %{{customdata[{i+1}]:,}}<br>"
hover_template_state += "<extra></extra>"

for fuel_type in all_fuel_types:
    display_name = display_names.get(fuel_type, fuel_type)
    fig_state.add_trace(go.Bar(
        name=display_name,
        x=[selected_state],
        y=df_state[fuel_type],
        marker_color=color_map.get(fuel_type),
        customdata=df_state[custom_data_cols],
        hovertemplate=hover_template_state
    ))

fig_state.update_layout(
    barmode='stack',
    title_text=f"School bus fuel types in {selected_state.title()}",
    xaxis_title="",
    yaxis_title="Number of Buses",
    legend_title_text='<b>Fuel Type</b>'
)

st.plotly_chart(fig_state, use_container_width=True)



# =========================================================
# 2️⃣  BUS AGE DISTRIBUTION (BY STATE)
# =========================================================
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



