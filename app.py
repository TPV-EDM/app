# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import json
import gzip
import numpy as np
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------
# CARGAR DATOS Y GEOJSON
# ------------------
with gzip.open("datos_plazas_disponibles_sin_prediccion.csv.gz", 'rt') as f:
    df_modelo = pd.read_csv(f)

with open("geometria_barrios_valido.geojson", "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# ------------------
# ENTRENAR MODELO
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

# Evaluación del modelo (opcional)
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

# ------------------
# INTERFAZ DE USUARIO
# ------------------
st.set_page_config(layout="wide")
st.title("🅿️ Parking Spot Prediction in Madrid")

# Tabs
tabs = st.tabs(["🔍 Prediction Map", "📈 Model Info", "📊 Data Visuals"])

# ------------------
# PESTAÑA 1: MAPA Y FILTRO
# ------------------
with tabs[0]:
    st.sidebar.header("Filter Options")
    dia = st.sidebar.selectbox("Day of the week", sorted(df_modelo["dia_semana"].unique()))
    hora_inicio, hora_fin = st.sidebar.slider("Hour Range", 0, 23, (12, 14))
    barrios = st.sidebar.multiselect("Select neighborhood(s)", sorted(df_modelo["barrio"].unique()))

    df_input = df_modelo[(df_modelo["dia_semana"] == dia) & (df_modelo["hora"].between(hora_inicio, hora_fin))]

    if barrios:
        df_input = df_input[df_input["barrio"].isin(barrios)]
    else:
        st.warning("Select at least one neighborhood to see the results.")
        st.stop()

    if df_input.empty:
        st.warning("No data for the selected filters.")
        st.stop()

    # Predicción
    X_pred = df_input[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']]
    df_input["predicted_available"] = model.predict(X_pred).round().astype(int)
    df_input["predicted_available"] = df_input["predicted_available"].clip(lower=0)

    df_input["predicted_occupied"] = df_input["numero_plazas"] - df_input["predicted_available"]
    df_input = df_input[df_input["predicted_occupied"] > 0]

    resumen = df_input.groupby("barrio").agg(
        total_spots=("numero_plazas", "first"),
        predicted_available=("predicted_available", "mean")
    ).reset_index()
    resumen["predicted_available"] = resumen["predicted_available"].round().astype(int)
    resumen["predicted_occupied"] = resumen["total_spots"] - resumen["predicted_available"]

    st.subheader("📋 Predicted Occupancy by Neighborhood")
    st.dataframe(resumen)

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

# ------------------
# PESTAÑA 2: INFO MODELO
# ------------------
with tabs[1]:
    st.subheader("ℹ️ Model Info")
    st.markdown("""
    - **Model Type**: Ridge Regression with One-Hot Encoding for categorical features.
    - **Features**: Barrio, día de la semana, tramo horario, número de plazas.
    - **Performance Metrics:**
    """)
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("MAE", f"{mae:.2f}")
    st.metric("R²", f"{r2:.4f}")

# ------------------
# PESTAÑA 3: VISUALIZACIONES
# ------------------
with tabs[2]:
    st.subheader("📊 Data Visualizations")

    fig1 = px.histogram(df_modelo, x="plazas_disponibles", nbins=30, title="Distribution of Available Spots", opacity=0.7)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(df_modelo, x="dia_semana", y="plazas_disponibles", title="Availability by Day of the Week")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(df_modelo, x="tramo_horario", y="plazas_disponibles", title="Availability by Time Slot")
    st.plotly_chart(fig3, use_container_width=True)
