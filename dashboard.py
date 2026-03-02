import streamlit as st
import pandas as pd
import plotly.express as px
from main import detect_anomalies
from main import train_risk_model
from main import generate_insights

st.set_page_config(layout="wide")

st.markdown("## ESG Risk Intelligence Platform")
st.caption("Real-time ESG risk monitoring and data validation engine")
st.divider()

# Load Data
df = pd.read_csv("classified_esg_data.csv")

# ======================
# SIDEBAR FILTERS
# ======================

st.sidebar.header(" Filters")

facility_filter = st.sidebar.multiselect(
    "Select Facility",
    options=df["Facility"].unique(),
    default=df["Facility"].unique()
)

month_filter = st.sidebar.multiselect(
    "Select Month",
    options=df["Month"].unique(),
    default=df["Month"].unique()
)

risk_filter = st.sidebar.multiselect(
    "Select Risk Level",
    options=df["Risk_Level"].unique(),
    default=df["Risk_Level"].unique()
)

# Apply Filters
filtered_df = df[
    (df["Facility"].isin(facility_filter)) &
    (df["Month"].isin(month_filter)) &
    (df["Risk_Level"].isin(risk_filter))
]

# ======================
# KPI METRICS
# ======================

col1, col2, col3 = st.columns(3)

col1.metric("Total Facilities", len(filtered_df))
col2.metric("High Risk Sites", len(filtered_df[filtered_df["Risk_Level"]=="High"]))
col3.metric("Invalid Records", len(filtered_df[filtered_df["Validation_Status"]!="Valid"]))

st.divider()

# ======================
# VISUALS
# ======================

col4, col5 = st.columns(2)

with col4:
    st.subheader("Risk Distribution")
    fig = px.pie(filtered_df, names="Risk_Level")
    st.plotly_chart(fig, use_container_width=True)

with col5:
    st.subheader("Emissions by Facility")
    fig2 = px.bar(
        filtered_df,
        x="Facility",
        y="Emissions_tCO2",
        color="Risk_Level"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("Filtered Dataset")
    st.dataframe(filtered_df) 

# ======================
# BENCHMARK ENGINE
# ======================

st.subheader("Facility Benchmark Comparison")

avg_score = df["ESG_Score"].mean()

df["Benchmark_Status"] = df["ESG_Score"].apply(
    lambda x: "Above Average" if x > avg_score else "Below Average"
)

benchmark_table = df[["Facility", "ESG_Score", "Benchmark_Status"]]

st.dataframe(benchmark_table)

# ======================
# ALERT ENGINE
# ======================

st.subheader(" System Alerts")

high_risk_count = len(filtered_df[filtered_df["Risk_Level"]=="High"])
invalid_count = len(filtered_df[filtered_df["Validation_Status"]!="Valid"])
max_emission = filtered_df["Emissions_tCO2"].max()

if high_risk_count > 0:
    st.error(f"{high_risk_count} high risk facilities detected!")

if invalid_count > 0:
    st.warning(f"{invalid_count} invalid records need review!")

if max_emission > 65:
    st.info("One or more facilities have unusually high emissions.")

if high_risk_count == 0 and invalid_count == 0:
    st.success("All facilities operating within safe ESG limits.")

# ======================
# AI INSIGHT GENERATOR
# ======================

st.divider()
st.subheader("Automated ESG Insights")

if len(filtered_df) > 0:

    high_risk_count = len(filtered_df[filtered_df["Risk_Level"]=="High"])
    invalid_count = len(filtered_df[filtered_df["Validation_Status"]!="Valid"])

    avg_emission = round(filtered_df["Emissions_tCO2"].mean(), 2)
    highest_emitter = filtered_df.loc[
        filtered_df["Emissions_tCO2"].idxmax()
    ]["Facility"]

    insight_text = f"""
    • The average emissions across selected facilities is **{avg_emission} tCO2**

    • The highest emitting facility is **{highest_emitter}**

    • There are **{high_risk_count} high-risk facilities** requiring immediate attention

    • Data validation flagged **{invalid_count} records**, indicating reporting inconsistencies
    """

    st.markdown(insight_text)

else:
    st.info("No data available for selected filters.")

# ======================
# CUSTOM UI STYLING
# ======================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1 {
    font-weight: 700;
    letter-spacing: 1px;
}

[data-testid="metric-container"] {
    background-color: #1c1f26;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

import plotly.express as px

st.subheader("3D ESG Risk Landscape")

fig = px.scatter_3d(
    df,
    x="Energy_kWh",
    y="Water_m3",
    z="Emissions_tCO2",
    color="Risk_Level",
    size="ESG_Score",
    hover_name="Facility",
    size_max=25
)

st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Predictive Risk Model")

model, acc = train_risk_model(df)

st.write(f"Model Accuracy: {round(acc*100,2)}%")

st.divider()
st.subheader("Executive Insight Summary")

insights = generate_insights(df)

for insight in insights:
    st.write("• " + insight)

