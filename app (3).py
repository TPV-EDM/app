# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 18:48:48 2025

@author: Usuario
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import json

# ------------------
# CARGAR DATOS Y GEOJSON
# ------------------
df = pd.read_csv("datos_plazas_disponibles_sin_prediccion (2).csv")
with open("geometria_barrios_valido.geojson", "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# ------------------
# INTERFAZ DE USUARIO
# ------------------
st.title("🅿️ Predicción de Plazas de Aparcamiento por Barrio")

dia = st.selectbox("Día de la semana", df["dia_semana"].unique())
hora = st.slider("Hora del día", int(df["hora"].min()), int(df["hora"].max()), 12)
barrio_seleccionado = st.multiselect("Selecciona barrio(s)", sorted(df["barrio"].unique()))

# ------------------
# CÁLCULO DE MÉTRICAS
# ------------------
df_filtrado = df[(df["dia_semana"] == dia) & (df["hora"] == hora)]

if barrio_seleccionado:
    df_filtrado = df_filtrado[df_filtrado["barrio"].isin(barrio_seleccionado)]
else:
    st.warning("Selecciona al menos un barrio para ver resultados.")
    st.stop()

if df_filtrado.empty:
    st.warning("No hay datos para ese día, hora y barrios seleccionados.")
    st.stop()

# Agrupar datos por barrio
agrupado = df_filtrado.groupby("barrio").agg(
    plazas_ocupadas=("num_tiques", "mean"),
    plazas_totales=("numero_plazas", "first")
)
agrupado["plazas_ocupadas"] = agrupado["plazas_ocupadas"].round().astype(int)
agrupado["plazas_totales"] = agrupado["plazas_totales"].round().astype(int)
agrupado["plazas_libres"] = (agrupado["plazas_totales"] - agrupado["plazas_ocupadas"]).clip(lower=0)

# Reset index para mostrar
resultado = agrupado.reset_index()[["barrio", "plazas_libres", "plazas_ocupadas", "plazas_totales"]]

# ------------------
# MOSTRAR TABLA
# ------------------
st.subheader("📋 Predicción por Barrio")
st.dataframe(resultado)

# ------------------
# MAPA COROPLET
# ------------------
st.subheader("🗺️ Mapa de plazas predichas")

m = folium.Map(location=[40.4168, -3.7038], zoom_start=12)

folium.Choropleth(
    geo_data=geojson_data,
    data=resultado,
    columns=["barrio", "plazas_libres"],
    key_on="feature.properties.barrio",
    fill_color="YlGnBu",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Plazas libres promedio",
).add_to(m)

st_folium(m, width=700, height=500)
