import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import gzip
import geopandas as gpd
import json
import folium
from streamlit_folium import folium_static

# ------------------ CARGA DE DATOS ------------------
@st.cache_data
def load_data():
    with gzip.open("datos_plazas_disponibles_sin_prediccion.csv.gz", 'rt') as f:
        df = pd.read_csv(f)
    return df

@st.cache_data
def load_geojson():
    gdf = gpd.read_file("geometria_barrios.geojson")
    gdf['barrio'] = (
        gdf['barrio']
        .str.upper()
        .str.normalize('NFKD')
        .str.encode('ascii', errors='ignore')
        .str.decode('utf-8')
    )
    return gdf

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

# ------------------ CONFIG STREAMLIT ------------------
st.set_page_config(layout="wide", page_title="Parking Prediction")
st.title("\U0001F17F Parking Spot Prediction in Madrid")

tabs = st.tabs(["\U0001F5FA\ufe0f Prediction Map", "\U0001F4CA Model Info", "\U0001F4C8 Data Visuals"])

# ------------------ LOAD DATA ------------------
gdf = load_geojson()
df = load_data()
model = build_model(df)

# ------------------ TAB 1 ------------------
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

        df_filtered['plazas_libres_pred'] = model.predict(
            df_filtered[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']]
        )

        # Agrupamos por barrio
        agg = df_filtered.groupby('barrio').agg(
            plazas_libres_medianas=('plazas_libres_pred', 'median'),
            plazas_totales_medianas=('numero_plazas', 'median')
        ).reset_index()
        agg['plazas_ocupadas_predichas'] = agg['plazas_totales_medianas'] - agg['plazas_libres_medianas']

        # CENTROIDES PARA PUNTOS
        centroides = gdf[['barrio', 'geometry']].copy()
        centroides['centroide'] = centroides.geometry.centroid
        centroides['lat'] = centroides.centroide.y
        centroides['lon'] = centroides.centroide.x

        agg_coords = agg.merge(centroides, on='barrio', how='left')

        # MAPA
        st.subheader("Predicted Occupied Spots by Neighborhood (circle = size)")
        m = folium.Map(location=[40.4168, -3.7038], zoom_start=12, tiles="cartodbpositron")

        for _, row in agg_coords.iterrows():
            if pd.notnull(row['lat']) and pd.notnull(row['lon']):
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=max(5, row['plazas_ocupadas_predichas'] / 1000),
                    color='red',
                    fill=True,
                    fill_opacity=0.6,
                    popup=f"{row['barrio']}: {int(row['plazas_ocupadas_predichas'])} ocupadas"
                ).add_to(m)

        folium_static(m, width=1000, height=500)

        # TABLA FINAL
        st.subheader("\U0001F697 Predicted Occupied Spots")
        df_summary = agg[['barrio', 'plazas_ocupadas_predichas']].copy()
        df_summary.columns = ['barrio', 'plazas_predichas']
        st.dataframe(df_summary, use_container_width=True)

# ------------------ TAB 2 ------------------
with tabs[1]:
    st.subheader("\U0001F4BB Model Info")
    st.markdown("""
    **Model**: Ridge Regression  
    **Input Features**: `barrio`, `dia_semana`, `tramo_horario`, `numero_plazas`  
    **Target**: `plazas_disponibles` (plazas libres)  
    **Output (calculado)**: Plazas ocupadas = total - predicción  
    """)

# ------------------ TAB 3 ------------------
with tabs[2]:
    subtab = st.radio("Choose a View", ["Individual Barrios", "Compare Barrios"])

    df['plazas_libres_pred'] = model.predict(df[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']])
    df['plazas_ocupadas_pred'] = df['numero_plazas'] - df['plazas_libres_pred']

    if subtab == "Individual Barrios":
        barrio_sel = st.selectbox("Choose a Neighborhood", df['barrio'].unique())
        fig = px.histogram(df[df['barrio'] == barrio_sel], x='plazas_ocupadas_pred', nbins=30,
                           title=f"Predicted Occupied Spots in {barrio_sel}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.box(df[df['barrio'].isin(barrios)], x='barrio', y='plazas_ocupadas_pred', points="outliers",
                     title="Comparative Occupied Spots by Barrio")
        st.plotly_chart(fig, use_container_width=True)
