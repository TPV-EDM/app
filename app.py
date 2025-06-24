import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from datetime import datetime
import gzip
import json

# ---------- DATA LOADING ----------
@st.cache_data
def load_data():
    with gzip.open("datos_plazas_disponibles_sin_prediccion.csv.gz", 'rt') as f:
        df = pd.read_csv(f)
    return df

@st.cache_data
def load_geojson():
    with open("geometria_barrios.json", "r", encoding="utf-8") as f:
        geojson = json.load(f)
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

# ---------- APP LAYOUT ----------
st.set_page_config(layout="wide", page_title="Parking Prediction")
st.title("\U0001F17F Parking Spot Prediction in Madrid")

# Tabs
tabs = st.tabs(["\U0001F5FA\ufe0f Prediction Map", "\U0001F4CA Model Info", "\U0001F4C8 Data Visuals"])

df = load_data()
geojson = load_geojson()
model = build_model(df)

# --------------- TAB 1: PREDICTIONS ------------------
with tabs[0]:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Filter Options")
        dia = st.selectbox("Day of the week", df['dia_semana'].unique())
        hora_range = st.slider("Hour Range", 0, 23, (8, 10))
        barrios = st.multiselect("Select neighborhood(s)", df['barrio'].unique(), default=df['barrio'].unique()[0:2])

    with col2:
        df_filtered = df[
            (df['dia_semana'] == dia) &
            (df['barrio'].isin(barrios)) &
            (df['hora'].between(hora_range[0], hora_range[1]))
        ]

        df_filtered['pred'] = model.predict(df_filtered[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']])

        # Agregación por barrio
        agg = df_filtered.groupby('barrio').agg(predicted_mean=('pred', 'mean')).reset_index()

        fig = px.choropleth_mapbox(
            agg,
            geojson=geojson,
            locations='barrio',
            featureidkey="properties.BARRIO",  # asegúrate de que esta clave exista
            color='predicted_mean',
            color_continuous_scale="OrRd",
            mapbox_style="carto-positron",
            zoom=10,
            center={"lat": 40.4168, "lon": -3.7038},  # Centro de Madrid
            opacity=0.6,
            labels={'predicted_mean': 'Mean Prediction'}
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

# --------------- TAB 2: MODEL INFO ------------------
with tabs[1]:
    st.subheader("\U0001F4BB Model Info")
    st.markdown("""
    **Model**: Ridge Regression  
    **Input Features**: `barrio`, `dia_semana`, `tramo_horario`, `numero_plazas`  
    **Target**: `plazas_disponibles`  
    **Alpha**: 1.0  
    **Preprocessing**: OneHotEncoder for categorical vars + passthrough for numeric  
    **Note**: Only predictions with `hora` within selected range and selected barrios are shown.
    """)

# --------------- TAB 3: VISUALIZATIONS ------------------
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
