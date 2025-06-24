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

with tabs[0]:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Filter Options")
        dia = st.selectbox("Day of the week", df['dia_semana'].unique())
        hora_range = st.slider("Hour Range", 0, 23, (8, 10))
        barrios = st.multiselect("Select neighborhood(s)", df['barrio'].unique(), default=df['barrio'].unique()[0:5])

    with col2:
        # Filtramos primero
        df_filtered = df[
            (df['dia_semana'] == dia) &
            (df['barrio'].isin(barrios)) &
            (df['hora'].between(hora_range[0], hora_range[1]))
        ].copy()

        # AGRUPAMOS: una fila por barrio y hora
        df_grouped = df_filtered.groupby(['barrio', 'dia_semana', 'hora']).agg(
            numero_plazas=('numero_plazas', 'median'),
            tramo_horario=('tramo_horario', 'first')  # asumimos que es único por hora
        ).reset_index()

        # Predecimos sobre el df ya agrupado
        df_grouped['plazas_libres_pred'] = model.predict(
            df_grouped[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']]
        )
        df_grouped['plazas_ocupadas_pred'] = df_grouped['numero_plazas'] - df_grouped['plazas_libres_pred']

        # Agrupamos por barrio (para mapa y tabla)
        agg = df_grouped.groupby('barrio').agg(
            plazas_ocupadas_predichas=('plazas_ocupadas_pred', 'sum')
        ).reset_index()

        # CENTROIDES PARA MAPA
        centroides = gdf[['barrio', 'geometry']].copy()
        centroides['centroide'] = centroides.geometry.centroid
        centroides['lat'] = centroides.centroide.y
        centroides['lon'] = centroides.centroide.x

        agg_coords = agg.merge(centroides, on='barrio', how='left')

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

        # Tabla resumen
        st.subheader("\U0001F697 Predicted Occupied Spots")
        df_summary = agg.rename(columns={'plazas_ocupadas_predichas': 'plazas_predichas'})
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
