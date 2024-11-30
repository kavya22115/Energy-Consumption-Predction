import pandas as pd
import streamlit as st
import joblib

class EnergyConsumptionApp:
    def __init__(self):
        st.set_page_config(
            page_title="Energy Consumption Prediction",
            page_icon="🔋",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self.setup_page()
        self.load_resources()

    def setup_page(self):
        bg_color = st.sidebar.selectbox("Select Background Color", [
            "Lavender", "Beige", "Mint", 
            "Peach", "Sky Blue", "Lilac", 
            "Light Coral"
        ])
        bg_color_code = {
            "Lavender": "#E6E6FA",
            "Beige": "#F5F5DC",
            "Mint": "#F0FFF0",
            "Peach": "#FFE5B4",
            "Sky Blue": "#87CEEB",
            "Lilac": "#C8A2C8",
            "Light Coral": "#F08080"
        }.get(bg_color, "#FFFFFF")

        st.markdown(f"""
        <style>
        .stApp {{
            background-color: {bg_color_code};
        }}
        .header {{
            text-align: center;
            font-family: 'Arial', sans-serif;
            padding: 15px;
            border-radius: 15px;
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            color: white;
            margin-bottom: 20px;
        }}
        .input-section {{
            font-family: 'Verdana', sans-serif;
            margin: 20px 0;
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .prediction-card {{
            font-family: 'Courier New', monospace;
            margin: 20px 0;
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #ddd;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }}
        </style>
        """, unsafe_allow_html=True)

    def load_resources(self):
        try:
            self.linear_model = joblib.load("linear_model.pkl")
            self.ridge_model = joblib.load("ridge_model.pkl")
            self.feature_names = joblib.load("feature_names.pkl")
            st.sidebar.success("Resources loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading resources: {e}")

    def run(self):
        st.markdown("<div class='header'><h1>🔋 Energy Consumption Prediction</h1></div>", unsafe_allow_html=True)

        st.markdown("<div class='input-section'><h3>Input Features</h3></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            voltage = st.slider("Voltage (V)", 220.0, 255.0, 240.0)
            global_intensity = st.slider("Global Intensity (A)", 0.0, 20.0, 4.63)
            sub_metering_1 = st.slider("Sub Metering 1 (Wh)", 0.0, 50.0, 1.12)
        with col2:
            sub_metering_2 = st.slider("Sub Metering 2 (Wh)", 0.0, 50.0, 1.30)
            sub_metering_3 = st.slider("Sub Metering 3 (Wh)", 0.0, 50.0, 6.46)
            date = st.date_input("Select Date", value=pd.Timestamp("2024-11-28"))
            time = st.time_input("Select Time", value=pd.Timestamp("2024-11-28 12:00:00").time())

        date_time = pd.Timestamp.combine(date, time)
        year = date_time.year
        month = date_time.month
        day = date_time.day
        hour = date_time.hour
        minute = date_time.minute
        is_holiday = 0
        light = 1
        weekday = date_time.weekday()

        # Create input data
        input_data = pd.DataFrame({
            "Global_reactive_power": [0.0],
            "Voltage": [voltage],
            "Global_intensity": [global_intensity],
            "Sub_metering_1": [sub_metering_1],
            "Sub_metering_2": [sub_metering_2],
            "Sub_metering_3": [sub_metering_3],
            "Year": [year],
            "Month": [month],
            "Day": [day],
            "Hour": [hour],
            "Minute": [minute],
            "Is_holiday": [is_holiday],
            "Light": [light],
            "Weekday": [weekday]
        })

        # Drop the 'Global_active_power' column (if present)
        input_data = input_data.drop(columns=['Global_active_power'], errors='ignore')

        # Only keep columns that are in both the model's feature set and the input data
        input_data = input_data[[col for col in self.feature_names if col in input_data.columns]]

        st.markdown("<div class='prediction-card'><h3>Predictions</h3></div>", unsafe_allow_html=True)
        try:
            linear_pred = self.linear_model.predict(input_data)[0]
            ridge_pred = self.ridge_model.predict(input_data)[0]

            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.metric(label="Linear Regression Prediction (kW)", value=f"{linear_pred:.2f}")
            st.metric(label="Ridge Regression Prediction (kW)", value=f"{ridge_pred:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        except ValueError as e:
            st.error(f"Prediction error: {e}")

        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.info("""
        - This tool provides predictions based on historical data.
        - It is not intended for critical energy management.
        """)

def main():
    app = EnergyConsumptionApp()
    app.run()

if __name__ == "__main__":
    main()
