import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

class EnergyConsumptionApp:
    def __init__(self):
        st.set_page_config(
            page_title="Energy Consumption Prediction",
            page_icon="⚡",  # Changed the page icon to a lightning bolt
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self.setup_page()
        self.load_resources()

    def setup_page(self):
        # Set background color options in the sidebar
        bg_color = st.sidebar.selectbox("Select Background Color", [
            "Warm", "Cool", "Minimal", "Bright"
        ])
        bg_color_code = {
            "Warm": "#FFE1A8",
            "Cool": "#A8D5E2",
            "Minimal": "#F0F0F0",
            "Bright": "#FFB6C1"
        }.get(bg_color, "#FFFFFF")

        # Custom background and style
        st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(135deg, {bg_color_code}, #ffffff);
            font-family: 'Arial', sans-serif;
        }}
        .header {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(to right, #f6d365, #fda085);
            color: white;
            font-size: 36px;
            border-radius: 15px;
            box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }}
        .input-section {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 30px;
        }}
        .input-container {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .input-container label {{
            font-weight: bold;
            font-size: 18px;
        }}
        .prediction-card {{
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 30px;
        }}
        .stSidebar {{
            background-color: #f1f1f1;
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
        st.markdown("<div class='header'>⚡ Energy Consumption Prediction</div>", unsafe_allow_html=True)

        # Sidebar content (background color selection) and input features
        st.sidebar.header("Customize Appearance")
        st.sidebar.markdown("Adjust the background color of the webpage to your liking!")

        # Input Section - placed below the background color selection
        st.sidebar.markdown("### Input Features")
        col1, col2 = st.columns([2, 3])

        with st.sidebar:
            with st.form("inputs_form"):
                voltage = st.slider("Voltage (V)", 220.0, 255.0, 240.0, help="Operating voltage in volts")
                global_intensity = st.slider("Global Intensity (A)", 0.0, 20.0, 4.63, help="Total current consumption in amperes")
                sub_metering_1 = st.slider("Sub Metering 1 (Wh)", 0.0, 50.0, 1.12, help="Active energy for kitchen appliances in watt-hours")
                sub_metering_2 = st.slider("Sub Metering 2 (Wh)", 0.0, 50.0, 1.30, help="Active energy for laundry appliances in watt-hours")
                sub_metering_3 = st.slider("Sub Metering 3 (Wh)", 0.0, 50.0, 6.46, help="Active energy for HVAC in watt-hours")
                date = st.date_input("Select Date", value=pd.Timestamp("2024-11-28"))
                time = st.time_input("Select Time", value=pd.Timestamp("2024-11-28 12:00:00").time())
                
                submit_button = st.form_submit_button("Submit")

        if submit_button:
            date_time = pd.Timestamp.combine(date, time)
            year = date_time.year
            month = date_time.month
            day = date_time.day
            hour = date_time.hour
            minute = date_time.minute
            weekday = date_time.weekday()

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
                "Weekday": [weekday]
            })

            input_data = input_data[[col for col in self.feature_names if col in input_data.columns]]

            # Visualize input data using a Heatmap
            st.subheader("Input Data Heatmap")
            heatmap = sns.heatmap(input_data.T, annot=True, cmap="YlGnBu", cbar=True)
            st.pyplot(heatmap.figure)

            # Prediction Section - displaying output in special diagram form
            st.markdown("<div class='prediction-card'><h3>Predictions</h3></div>", unsafe_allow_html=True)
            try:
                linear_pred = self.linear_model.predict(input_data)[0]
                ridge_pred = self.ridge_model.predict(input_data)[0]

                # Displaying predictions using Donut Chart
                fig_linear = go.Figure(go.Pie(
                    labels=["Linear Regression", "Others"],
                    values=[linear_pred, 10 - linear_pred],
                    hole=0.4,
                    textinfo="percent+label",
                    marker=dict(colors=["#00BFFF", "#f0f8ff"])
                ))

                fig_linear.update_layout(title="Linear Regression Prediction (kW)")
                st.plotly_chart(fig_linear)

                fig_ridge = go.Figure(go.Pie(
                    labels=["Ridge Regression", "Others"],
                    values=[ridge_pred, 10 - ridge_pred],
                    hole=0.4,
                    textinfo="percent+label",
                    marker=dict(colors=["#FF6347", "#f0f8ff"])
                ))

                fig_ridge.update_layout(title="Ridge Regression Prediction (kW)")
                st.plotly_chart(fig_ridge)

                # Displaying the predictions as text
                st.subheader("Prediction Results")
                st.write(f"Linear Regression Prediction: {linear_pred:.2f} kW")
                st.write(f"Ridge Regression Prediction: {ridge_pred:.2f} kW")

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
