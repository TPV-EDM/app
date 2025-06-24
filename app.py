import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import gzip  # ← IMPORTANTE
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from datetime import datetime

# ---------- DATA LOADING ----------
@st.cache_data
def load_data():
    with gzip.open("datos_plazas_disponibles_sin_prediccion.csv.gz", 'rt') as f:
        df = pd.read_csv(f)
    return df

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

        # Solo mostrar ocupadas (plazas < 0.3 del total)
        df_filtered = df_filtered[df_filtered['pred'] < df_filtered['numero_plazas'] * 0.3]

        agg = df_filtered.groupby(['barrio']).agg(predicted_occupied=('pred', lambda x: (x).sum())).reset_index()
        fig = px.bar(agg, x='barrio', y='predicted_occupied', title="Predicted Occupied Spots by Neighborhood",
                     labels={'predicted_occupied': 'Occupied Spots'}, height=500)
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
