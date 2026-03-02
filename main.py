import pandas as pd


def calculate_esg_score(row):
    score = 0

    # Environmental (Emissions)
    if row["Emissions_tCO2"] < 60:
        score += 40
    elif row["Emissions_tCO2"] < 80:
        score += 25
    else:
        score += 10

    # Energy Efficiency
    if row["Energy_kWh"] < 13000:
        score += 30
    elif row["Energy_kWh"] < 16000:
        score += 20
    else:
        score += 10

    # Water Efficiency
    if row["Water_m3"] < 300:
        score += 30
    elif row["Water_m3"] < 400:
        score += 20
    else:
        score += 10

    return score

# 1. Load Data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df["ESG_Score"] = df.apply(calculate_esg_score, axis=1)
        print("Data loaded successfully")
        return df
    except FileNotFoundError:
        print("File not found. Check file location.")
        exit()

def validate_data(df):

    df["Validation_Status"] = "Valid"

    numeric_columns = ["Energy_kWh", "Water_m3", "Emissions_tCO2"]

    # Check missing values
    df.loc[df.isnull().any(axis=1), "Validation_Status"] = "Missing Data"

    # Check negative values
    for col in numeric_columns:
        df.loc[df[col] < 0, "Validation_Status"] = "Invalid Negative Value"

    # Check unrealistic values
    df.loc[df["Energy_kWh"] > 200000, "Validation_Status"] = "Unrealistic Energy"

    print("Validation check completed")

    return df
# 3. Risk Classification
def classify_metric(value, low, high):
    if value < low:
        return "Low"
    elif low <= value <= high:
        return "Medium"
    else:
        return "High"

def classify_row(row):
    energy_risk = classify_metric(row["Energy_kWh"], 50000, 100000)
    water_risk = classify_metric(row["Water_m3"], 200, 500)
    emission_risk = classify_metric(row["Emissions_tCO2"], 50, 150)

    risks = [energy_risk, water_risk, emission_risk]

    if "High" in risks:
        return "High"
    elif "Medium" in risks:
        return "Medium"
    else:
        return "Low"

# Main Execution
def main():
    file_path = "esg_data.csv"
    df = load_data(file_path)
    df = validate_data(df)

    df["Risk_Level"] = df.apply(classify_row, axis=1)

    df.to_csv("classified_esg_data.csv", index=False)

    print("Risk classification complete")
    print("Output saved as classified_esg_data.csv")

if __name__ == "__main__":
    main()

# ======================
# ESG SCORE ENGINE
# ======================

def calculate_esg_score(row):
    score = 100
    
    # Emission penalty
    if row["Emissions_tCO2"] > 65:
        score -= 30
    elif row["Emissions_tCO2"] > 55:
        score -= 15
    
    # Validation penalty
    if row["Validation_Status"] != "Valid":
        score -= 25
    
    # Risk penalty
    if row["Risk_Level"] == "High":
        score -= 20
    elif row["Risk_Level"] == "Medium":
        score -= 10
        
    return max(score, 0)


def detect_anomalies(df):

    anomalies = []

    # High emissions anomaly
    high_emit = df[df["Emissions_tCO2"] > df["Emissions_tCO2"].mean() * 1.5]
    if not high_emit.empty:
        anomalies.append("Unusually high emissions detected")

    # Water anomaly
    high_water = df[df["Water_m3"] > df["Water_m3"].mean() * 1.5]
    if not high_water.empty:
        anomalies.append("Abnormally high water usage detected")

    # Energy anomaly
    high_energy = df[df["Energy_kWh"] > df["Energy_kWh"].mean() * 1.5]
    if not high_energy.empty:
        anomalies.append("Extreme energy consumption detected")

    return anomalies

# ===============================
# PREDICTIVE RISK MODEL
# ===============================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_risk_model(df):

    df = df.copy()

    df["Risk_Label"] = df["Risk_Level"].map({
        "Low": 0,
        "Medium": 1,
        "High": 2
    })

    features = df[["Energy_kWh", "Water_m3", "Emissions_tCO2", "ESG_Score"]].copy()
    features = features.fillna(features.mean())

    target = df["Risk_Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, accuracy


# ===============================
# INSIGHT NARRATOR
# ===============================

def generate_insights(df):

    insights = []

    avg_emissions = df["Emissions_tCO2"].mean()
    highest_emitter = df.loc[df["Emissions_tCO2"].idxmax(), "Facility"]
    high_risk_count = (df["Risk_Level"] == "High").sum()

    insights.append(
        f"The average emissions across facilities is {round(avg_emissions,2)} tCO2."
    )

    insights.append(
        f"{highest_emitter} is currently the highest emitting facility."
    )

    insights.append(
        f"There are {high_risk_count} facilities classified as high risk."
    )

    return insights