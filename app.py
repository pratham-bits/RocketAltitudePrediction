import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("rocket_altitude_model.pkl")

st.set_page_config(page_title="Rocket Altitude Predictor", layout="centered")
# Custom Rocket Theme CSS
st.markdown("""
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e3a8a, #312e81);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title styling */
    .title {
        text-align: center;
        color: #ffffff;
        font-size: 40px;
        font-weight: 700;
        padding: 20px 0px;
    }

    /* Section headers */
    h2, h3, h4 {
        color: #93c5fd !important;
    }

    /* Input box styling */
    .stNumberInput > div > div input {
        background-color: #111827;
        color: white;
        border-radius: 8px;
        border: 1px solid #1e40af;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #1e40af;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: 2px solid #3b82f6;
    }

    div.stButton > button:hover {
        background-color: #2563eb;
        border-color: #60a5fa;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🚀 Rocket Altitude Prediction App</div>', unsafe_allow_html=True)

with st.container():
    st.subheader("🔥 Rocket Parameters")

st.write("Select a mode. Simple mode asks for only 3 parameters, Advanced mode gives full control.")

# Mode selector
mode = st.radio("Choose Mode:", ["Simple (3 inputs)", "Advanced (all inputs)"])

# Default values for hidden features (used in Simple Mode)
DEFAULTS = {
    "initial_velocity_m_s": 0.0,
    "drag_coeff": 0.3,
    "air_density": 1.225,
    "cross_section_m2": 0.1
}

# Inputs for Simple or Advanced mode
if mode == "Simple (3 inputs)":
    st.subheader("Simple Mode Inputs (Beginner Friendly)")
    
    thrust = st.number_input("Thrust (N)", min_value=0.0, step=10.0, value=2500.0)
    mass = st.number_input("Mass (kg)", min_value=0.1, step=0.1, value=150.0)

    diameter_cm = st.number_input("Rocket Diameter (cm)", min_value=1.0, step=0.1, value=10.0)

    shape = st.selectbox("Rocket Shape", ["Aerodynamic", "Normal", "High Drag"])

    # Convert diameter → area
    radius_m = (diameter_cm / 100) / 2
    cross_section = np.pi * (radius_m ** 2)

    # Map drag coefficient
    if shape == "Aerodynamic":
        drag = 0.25
    elif shape == "Normal":
        drag = 0.35
    else:
        drag = 0.50

    # Default values
    density = 1.225
    initial_velocity = 0.0

else:
    st.subheader("Advanced Mode Inputs")
    initial_velocity = st.number_input("Initial Velocity (m/s)", min_value=0.0, step=0.1, value=0.0)
    thrust = st.number_input("Thrust (N)", min_value=0.0, step=10.0, value=2500.0)
    mass = st.number_input("Mass (kg)", min_value=0.1, step=0.1, value=150.0)
    drag = st.number_input("Drag Coefficient", min_value=0.0, step=0.01, value=0.3)
    density = st.number_input("Air Density (kg/m³)", min_value=0.0, step=0.01, value=1.225)
    cross_section = st.number_input("Cross Section Area (m²)", min_value=0.0, step=0.001, value=0.1)

# Helper to build the proper DataFrame
def make_df(time_value):
    return pd.DataFrame([{
        "time_s": time_value,
        "initial_velocity_m_s": initial_velocity,
        "thrust_N": thrust,
        "mass_kg": mass,
        "drag_coeff": drag,
        "air_density": density,
        "cross_section_m2": cross_section
    }])

# Single prediction
st.subheader("🧮 Single Time Prediction")


time_single = st.number_input("Enter Time (s)", min_value=0.0, step=0.1, value=5.0)

if st.button("Predict Altitude"):
    X = make_df(time_single)
    altitude = model.predict(X)[0]
    st.success(f"Predicted Altitude at {time_single:.1f} seconds: {altitude:.2f} m")

# Altitude-Time graph
st.subheader("📈 Altitude-Time Graph")

if st.button("Generate Flight Curve"):
    time_values = np.linspace(0, 60, 200)   # simulate from 0 to 60 seconds
    
    # Build DataFrame for full curve
    df_curve = pd.DataFrame({
        "time_s": time_values,
        "initial_velocity_m_s": [initial_velocity] * len(time_values),
        "thrust_N": [thrust] * len(time_values),
        "mass_kg": [mass] * len(time_values),
        "drag_coeff": [drag] * len(time_values),
        "air_density": [density] * len(time_values),
        "cross_section_m2": [cross_section] * len(time_values)
    })

    altitude_predictions = model.predict(df_curve)

    # Plot the curve
    fig, ax = plt.subplots()
    ax.plot(time_values, altitude_predictions, linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Predicted Rocket Altitude Over Time")
    ax.grid(True)
    st.pyplot(fig)

    # Show max altitude
    max_alt = float(np.max(altitude_predictions))
    t_max = float(time_values[np.argmax(altitude_predictions)])
    st.info(f"**Max Altitude: {max_alt:.2f} m at {t_max:.2f} seconds**")
