# app.py
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib
import json
import gzip

# ------------------
# LOAD MODEL AND FILES
# ------------------

@st.cache_resource
def load_model():
    return joblib.load("modelo_ridge_disponibles.pkl")

@st.cache_data
def load_data():
    with gzip.open("datos_plazas_disponibles_sin_prediccion.csv.gz", "rt") as f:
        df = pd.read_csv(f)
    return df

@st.cache_data
def load_geojson():
    with open("geometria_barrios_valido.geojson", "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
df = load_data()
geojson_data = load_geojson()

# ------------------
# USER INTERFACE
# ------------------
st.title("🅿️ Parking Availability Prediction by Neighborhood")

day = st.selectbox("Day of the week", sorted(df["dia_semana"].unique()))
hour = st.slider("Hour of the day", int(df["hora"].min()), int(df["hora"].max()), 12)
selected_neighborhoods = st.multiselect("Select neighborhood(s)", sorted(df["barrio"].unique()))

if not selected_neighborhoods:
    st.warning("Please select at least one neighborhood.")
    st.stop()

# ------------------
# PREPARE INPUT FOR PREDICTION
# ------------------
# Use average number of total spots per neighborhood (fixed value)
plazas_reference = df.groupby("barrio")["numero_plazas"].mean().round().astype(int).to_dict()

df_input = pd.DataFrame({
    "barrio": selected_neighborhoods,
    "dia_semana": [day] * len(selected_neighborhoods),
    "tramo_horario": [f"{hour:02d}:00-{hour+1:02d}:00"] * len(selected_neighborhoods),
    "numero_plazas": [plazas_reference[b] for b in selected_neighborhoods]
})

# Predict
predicted_available = model.predict(df_input).round().astype(int)
df_input["predicted_available"] = np.clip(predicted_available, 0, None)

# ------------------
# DISPLAY TABLE
# ------------------
st.subheader("📋 Predicted Available Spots")
st.dataframe(df_input[["barrio", "numero_plazas", "predicted_available"]].rename(columns={
    "barrio": "Neighborhood",
    "numero_plazas": "Total Spots",
    "predicted_available": "Available Spots"
}))

# ------------------
# MAP DISPLAY
# ------------------
st.subheader("🗺️ Availability Map")

m = folium.Map(location=[40.4168, -3.7038], zoom_start=12)

folium.Choropleth(
    geo_data=geojson_data,
    data=df_input,
    columns=["barrio", "predicted_available"],
    key_on="feature.properties.barrio",
    fill_color="YlGnBu",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Predicted Available Spots",
).add_to(m)

st_folium(m, width=700, height=500)
