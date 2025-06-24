import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import gzip
import json

# ---------- CARGA DE DATOS ----------
@st.cache_data

def load_data():
    with gzip.open("datos_plazas_disponibles_sin_prediccion.csv.gz", 'rt') as f:
        df = pd.read_csv(f)
    return df

@st.cache_data

def load_geojson():
    with open("geometria_barrios.geojson", "r", encoding="utf-8") as f:
        geojson = json.load(f)
        for feature in geojson['features']:
            b = feature['properties']['BARRIO']
            feature['properties']['BARRIO_NORM'] = (
                b.upper()
                 .replace('Á', 'A').replace('É', 'E').replace('Í', 'I')
                 .replace('Ó', 'O').replace('Ú', 'U').replace('Ü', 'U')
            )
    return geojson

def build_model(df):
    X = df[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']]
    y = df['plazas_disponibles']
    categorical_cols = ['barrio', 'dia_semana', 'tramo_horario']
    numeric_cols = ['numero_plazas']
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ])
    model = make_pipeline(preprocessor, Ridge(alpha=1.0))
    model.fit(X, y)
    return model

# ---------- LAYOUT ----------
st.set_page_config(layout="wide", page_title="Parking Prediction")
st.title("\U0001F17F Parking Spot Prediction in Madrid")

# Tabs
tabs = st.tabs(["\U0001F5FA\ufe0f Prediction Map", "\U0001F4CA Model Info", "\U0001F4C8 Data Visuals"])

# Load
geojson = load_geojson()
df = load_data()
model = build_model(df)

with tabs[0]:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Filter Options")
        dia = st.selectbox("Day of the week", df['dia_semana'].unique())
        hora_range = st.slider("Hour Range", 0, 23, (8, 10))
        barrios = st.multiselect("Select neighborhood(s)", df['barrio'].unique(), default=df['barrio'].unique()[0:5])

    with col2:
        df_filtered = df[
            (df['dia_semana'] == dia) &
            (df['barrio'].isin(barrios)) &
            (df['hora'].between(hora_range[0], hora_range[1]))
        ].copy()

        df_filtered['pred'] = model.predict(df_filtered[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']])
        df_filtered['ocupacion_%'] = df_filtered['pred'] / df_filtered['numero_plazas']

        # Normalizar nombre del barrio (coincidir con GeoJSON)
        df_filtered['barrio_norm'] = (
            df_filtered['barrio']
            .str.upper()
            .str.replace('Á', 'A')
            .str.replace('É', 'E')
            .str.replace('Í', 'I')
            .str.replace('Ó', 'O')
            .str.replace('Ú', 'U')
            .str.replace('Ü', 'U')
        )

        # Agrupar para el mapa
        agg = df_filtered.groupby('barrio_norm').agg(total_pred_occupied=('pred', 'sum')).reset_index()

        fig = px.choropleth_mapbox(
            agg,
            geojson=geojson,
            locations='barrio_norm',
            color='total_pred_occupied',
            featureidkey="properties.BARRIO_NORM",
            mapbox_style="carto-positron",
            center={"lat": 40.4168, "lon": -3.7038},
            zoom=10,
            color_continuous_scale="Reds",
            opacity=0.65,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Mostrar tabla
        st.subheader("🧾 Predicted Occupied Spots by Hour")
        df_summary = df_filtered.groupby(['barrio', 'hora']).agg(
            numero_plazas=('numero_plazas', 'sum'),
            pred=('pred', 'sum')
        ).reset_index()
        df_summary['ocupacion_%'] = df_summary['pred'] / df_summary['numero_plazas']
        st.dataframe(df_summary.sort_values(['barrio', 'hora']), use_container_width=True)
# TAB 2
with tabs[1]:
    st.subheader("\U0001F4BB Model Info")
    st.markdown("""
    **Model**: Ridge Regression  
    **Input Features**: `barrio`, `dia_semana`, `tramo_horario`, `numero_plazas`  
    **Target**: `plazas_disponibles`  
    **Alpha**: 1.0  
    **Preprocessing**: OneHotEncoder for categorical vars + passthrough for numeric
    """)

# TAB 3
with tabs[2]:
    subtab = st.radio("Choose a View", ["Individual Barrios", "Compare Barrios"])

    df['pred'] = model.predict(df[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']])

    if subtab == "Individual Barrios":
        barrio_sel = st.selectbox("Choose a Neighborhood", df['barrio'].unique())
        fig = px.histogram(df[df['barrio'] == barrio_sel], x='pred', nbins=30, title=f"Distribution of Predictions in {barrio_sel}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.box(df[df['barrio'].isin(barrios)], x='barrio', y='pred', points="outliers",
                     title="Comparative Prediction by Barrio")
        st.plotly_chart(fig, use_container_width=True)
