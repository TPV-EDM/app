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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

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

# Métricas del modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ------------------
# INTERFAZ DE USUARIO
# ------------------
st.set_page_config(layout="wide")
tabs = st.tabs(["🔍 Prediction Map", "📊 Model Info", "📈 Data Visuals"])

# ------------------
# TAB 1: PREDICCIÓN MAPA
# ------------------
with tabs[0]:
    st.sidebar.header("Filter Options")
    dia = st.sidebar.selectbox("Day of the week", sorted(df_modelo["dia_semana"].unique()))
    hora = st.sidebar.slider("Hour of the day", int(df_modelo["hora"].min()), int(df_modelo["hora"].max()), 12)
    barrios = st.sidebar.multiselect("Select neighborhood(s)", sorted(df_modelo["barrio"].unique()))

    df_input = df_modelo[(df_modelo["dia_semana"] == dia) & (df_modelo["hora"] == hora)]

    if barrios:
        df_input = df_input[df_input["barrio"].isin(barrios)]
    else:
        st.warning("Select at least one neighborhood to see the results.")
        st.stop()

    if df_input.empty:
        st.warning("No data for the selected day, hour and neighborhoods.")
        st.stop()

    X_pred = df_input[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']]
    df_input["predicted_available"] = model.predict(X_pred).round().astype(int).clip(lower=0)

    resumen = df_input.groupby("barrio").agg(
        total_spots=("numero_plazas", "first"),
        predicted_available=("predicted_available", "mean")
    ).reset_index()
    resumen["predicted_available"] = resumen["predicted_available"].round().astype(int)
    resumen["predicted_occupied"] = resumen["total_spots"] - resumen["predicted_available"]

    st.subheader("📋 Predicted Availability by Neighborhood")
    st.dataframe(resumen)

    st.subheader("🗺️ Map of Predicted Availability")
    m = folium.Map(location=[40.4168, -3.7038], zoom_start=12)

    folium.Choropleth(
        geo_data=geojson_data,
        data=resumen,
        columns=["barrio", "predicted_occupied"],
        key_on="feature.properties.barrio",
        fill_color="OrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Occupied Parking Spots (predicted)"
    ).add_to(m)

    st_folium(m, width=900, height=500)

# ------------------
# TAB 2: INFO MODELO
# ------------------
with tabs[1]:
    st.subheader("📌 Model Metrics")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("MAE", f"{mae:.2f}")
    st.metric("R²", f"{r2:.4f}")

    st.markdown("### 📋 Model Details")
    st.code("""
    Model: Ridge Regression
    Features: barrio, dia_semana, tramo_horario, numero_plazas
    Categorical Encoding: OneHotEncoder
    Pipeline: ColumnTransformer + Ridge
    """)

# ------------------
# TAB 3: VISUALIZACIONES
# ------------------
with tabs[2]:
    st.subheader("📈 Data Visualizations")
    st.markdown("#### Distribution of Available Spots")
    fig, ax = plt.subplots()
    df_modelo['plazas_disponibles'].hist(ax=ax, bins=30, color='skyblue')
    st.pyplot(fig)

    st.markdown("#### Spots by Hour")
    fig2, ax2 = plt.subplots()
    df_modelo.groupby("hora")["plazas_disponibles"].mean().plot(ax=ax2)
    ax2.set_ylabel("Avg. Available Spots")
    st.pyplot(fig2)
