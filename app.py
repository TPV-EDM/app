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

# ------------------
# STREAMLIT UI
# ------------------
st.set_page_config(layout="wide")
st.title("🅿️ Parking Spot Prediction in Madrid")
tabs = st.tabs(["📍 Prediction Map", "📊 Model Info", "📈 Data Visuals"])

# ------------------
# TAB 1: PREDICTION MAP
# ------------------
with tabs[0]:
    st.sidebar.header("Filter Options")
    dia = st.sidebar.selectbox("Day of the week", sorted(df_modelo["dia_semana"].unique()))
    hora_inicio, hora_fin = st.sidebar.slider("Hour Range", 0, 23, (12, 14))
    barrios = st.sidebar.multiselect("Select neighborhood(s)", sorted(df_modelo["barrio"].unique()))

    st.subheader("📍 Prediction Map")
    df_input = df_modelo[
        (df_modelo["dia_semana"] == dia) &
        (df_modelo["hora"] >= hora_inicio) &
        (df_modelo["hora"] <= hora_fin)
    ]

    if barrios:
        df_input = df_input[df_input["barrio"].isin(barrios)]
    else:
        st.warning("Select at least one neighborhood to see the results.")
        st.stop()

    if df_input.empty:
        st.warning("No data for the selected filters.")
        st.stop()

    # Predicción y filtrado
    X_pred = df_input[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']]
    pred = model.predict(X_pred)
    df_input["predicted_available"] = np.round(pred).clip(min=0).astype(int)
    df_input["predicted_occupied"] = df_input["numero_plazas"] - df_input["predicted_available"]
    df_input = df_input[df_input["predicted_occupied"] > 0]

    resumen = df_input.groupby("barrio").agg(
        total_spots=("numero_plazas", "first"),
        predicted_available=("predicted_available", "mean")
    ).reset_index()
    resumen["predicted_available"] = resumen["predicted_available"].round().astype(int)
    resumen["predicted_occupied"] = resumen["total_spots"] - resumen["predicted_available"]

    st.dataframe(resumen)

    m = folium.Map(location=[40.4168, -3.7038], zoom_start=12)
    folium.Choropleth(
        geo_data=geojson_data,
        data=resumen,
        columns=["barrio", "predicted_available"],
        key_on="feature.properties.barrio",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Available Parking Spots (predicted)"
    ).add_to(m)
    st_folium(m, width=700, height=500)

# ------------------
# TAB 2: MODEL INFO
# ------------------
with tabs[1]:
    st.subheader("📊 Model Info")
    y_pred = model.predict(X)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("MAE", f"{mae:.2f}")
    st.metric("R²", f"{r2:.4f}")

# ------------------
# TAB 3: DATA VISUALS
# ------------------
with tabs[2]:
    st.subheader("📈 Data Visualizations")
    sub_tabs = st.tabs(["Individual", "Compare"])

    with sub_tabs[0]:
        barrio_indiv = st.selectbox("Choose a neighborhood", sorted(df_modelo["barrio"].unique()))
        df_barrio = df_modelo[df_modelo["barrio"] == barrio_indiv]
        fig1 = px.histogram(df_barrio, x="plazas_disponibles", nbins=20, opacity=0.7)
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = px.box(df_barrio, x="dia_semana", y="plazas_disponibles")
        st.plotly_chart(fig2, use_container_width=True)

    with sub_tabs[1]:
        barrios_comp = st.multiselect("Compare neighborhoods", sorted(df_modelo["barrio"].unique()), default=barrios[:2])
        df_comp = df_modelo[df_modelo["barrio"].isin(barrios_comp)]
        fig3 = px.histogram(df_comp, x="plazas_disponibles", color="barrio", nbins=20, barmode="overlay", opacity=0.6)
        st.plotly_chart(fig3, use_container_width=True)
        fig4 = px.box(df_comp, x="dia_semana", y="plazas_disponibles", color="barrio")
        st.plotly_chart(fig4, use_container_width=True)
