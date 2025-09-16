import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
genai.configure(api_key="AIzaSyA8PTPAJSiY7niZUPMueKnJps5Aa92f1nI")
model = genai.GenerativeModel("gemini-1.5-flash")
df = pd.read_csv("C:/Users/Archit Jagtap/Desktop/mining_sustainability_10years.csv")
X = df[['Energy_kWh', 'Waste_tons']]
y_co2 = df['CO2_emissions']
y_water = df['Water_consumption']
X_train, X_test, y_train_co2, y_test_co2 = train_test_split(X, y_co2, test_size=0.2, random_state=42)
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X, y_water, test_size=0.2, random_state=42)
rf = RandomForestRegressor(random_state=42)
lr = LinearRegression()
rf.fit(X_train, y_train_co2)
lr.fit(X_train, y_train_co2)
st.title("🌱 Mining Sustainability Dashboard (AI + ML)")
tab1, tab2 = st.tabs(["📊 Predictions", "📈 Dataset Insights"])
with tab1:
    st.subheader("Make Predictions + Get AI Recommendations")

    energy = st.number_input("Enter Energy (kWh):", min_value=1000, max_value=10000, value=5000)
    waste = st.number_input("Enter Waste (tons):", min_value=10, max_value=500, value=100)

    co2_pred_rf = rf.predict([[energy, waste]])[0]
    water_pred_rf = rf.predict([[energy, waste]])[0]

    st.write(f"🔹 Predicted CO₂ Emissions (Random Forest): {co2_pred_rf:.2f}")
    st.write(f"🔹 Predicted Water Consumption (Random Forest): {water_pred_rf:.2f}")

    if st.button("💡 Get AI Recommendation"):
        prompt = f"""
        Based on mining sustainability analysis:
        Energy consumption: {energy} kWh
        Waste generated: {waste} tons
        Predicted CO₂ emissions: {co2_pred_rf:.2f}
        Predicted Water consumption: {water_pred_rf:.2f}

        Provide actionable sustainability recommendations (energy, waste, water, emissions).
        """
        response = model.generate_content(prompt)
        st.success(response.text)
with tab2:
    st.subheader("AI Insights from Full Dataset")
    st.dataframe(df.head())

    if st.button("🔍 Analyze Full Dataset with AI"):
        dataset_summary = df.describe().to_string()
        prompt = f"""
        Here is a sustainability dataset summary:
        {dataset_summary}

        Please analyze:
        - Key trends in energy, waste, CO₂, water
        - Potential environmental risks
        - Recommendations to improve sustainability
        """
        response = model.generate_content(prompt)
        st.info(response.text)
