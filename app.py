
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import json
import gzip
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

# ------------------
# CARGAR DATOS Y GEOJSON
# ------------------

with gzip.open("datos_plazas_disponibles_sin_prediccion.csv.gz", 'rt') as f:
    df_modelo = pd.read_csv(f)

with open("geometria_barrios_valido.geojson", "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# ------------------
# ENTRENAR MODELO INTEGRADO
# ------------------
X = df_modelo[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']]
y = df_modelo['plazas_disponibles']

categorical_cols = ['barrio', 'dia_semana', 'tramo_horario']
numeric_cols = ['numero_plazas']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numeric_cols)
])

model = make_pipeline(preprocessor, Ridge(alpha=1.0))
model.fit(X, y)

# ------------------
# INTERFAZ DE USUARIO
# ------------------
st.title("🅿️ Parking Spot Prediction by Neighborhood")

dia = st.selectbox("Day of the week", sorted(df_modelo["dia_semana"].unique()))
hora = st.slider("Hour of the day", int(df_modelo["hora"].min()), int(df_modelo["hora"].max()), 12)
barrios = st.multiselect("Select neighborhood(s)", sorted(df_modelo["barrio"].unique()))

df_input = df_modelo[(df_modelo["dia_semana"] == dia) & (df_modelo["hora"] == hora)]

if barrios:
    df_input = df_input[df_input["barrio"].isin(barrios)]
else:
    st.warning("Select at least one neighborhood to see the results.")
    st.stop()

if df_input.empty:
    st.warning("No data for the selected day, hour and neighborhoods.")
    st.stop()

# ------------------
# HACER PREDICCIÓN
# ------------------
X_pred = df_input[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']]
df_input["predicted_available"] = model.predict(X_pred).round().astype(int)
df_input["predicted_available"] = df_input["predicted_available"].clip(lower=0)

# Agrupar por barrio
resumen = df_input.groupby("barrio").agg(
    total_spots=("numero_plazas", "first"),
    predicted_available=("predicted_available", "mean")
).reset_index()
resumen["predicted_available"] = resumen["predicted_available"].round().astype(int)
resumen["predicted_occupied"] = resumen["total_spots"] - resumen["predicted_available"]

# ------------------
# MOSTRAR TABLA
# ------------------
st.subheader("📋 Predicted Availability by Neighborhood")
st.dataframe(resumen)

# ------------------
# MAPA COROPLETAS
# ------------------
st.subheader("🗺️ Map of Predicted Availability")

m = folium.Map(location=[40.4168, -3.7038], zoom_start=12)

folium.Choropleth(
    geo_data=geojson_data,
    data=resumen,
    columns=["barrio", "predicted_available"],
    key_on="feature.properties.barrio",
    fill_color="YlGnBu",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Available Parking Spots (predicted)"
).add_to(m)

st_folium(m, width=700, height=500)
