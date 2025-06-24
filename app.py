import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import folium_static
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import gzip
from geopy.geocoders import Nominatim
import time

# ----------- CACHEO DE DATOS Y GEO ------------
@st.cache_data
def load_data():
    with gzip.open("datos_plazas_disponibles_sin_prediccion.csv.gz", 'rt') as f:
        df = pd.read_csv(f)
    return df

@st.cache_data
def get_coords(barrios):
    geolocator = Nominatim(user_agent="pred-parking")
    coords = []
    for barrio in barrios:
        try:
            loc = geolocator.geocode(f"{barrio}, Madrid, España")
            if loc:
                coords.append({'barrio': barrio, 'lat': loc.latitude, 'lon': loc.longitude})
            time.sleep(1)  # evitar baneos
        except:
            continue
    return pd.DataFrame(coords)

# ----------- MODELO -----------
def build_model(df):
    X = df[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']]
    y = df['plazas_disponibles']
    categorical = ['barrio', 'dia_semana', 'tramo_horario']
    numeric = ['numero_plazas']
    preproc = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', 'passthrough', numeric)
    ])
    model = make_pipeline(preproc, Ridge(alpha=1.0))
    model.fit(X, y)
    return model

# ----------- LAYOUT APP -----------
st.set_page_config(layout="wide", page_title="Parking Prediction")
st.title("🚗 Parking Spot Prediction in Madrid")

tabs = st.tabs(["🗺️ Prediction Map", "💻 Model Info", "📊 Data Visuals"])
df = load_data()
model = build_model(df)

# ----------- TAB 1 -----------
with tabs[0]:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Filter Options")
        dia = st.selectbox("Day of the week", df['dia_semana'].unique())
        hora_min, hora_max = st.slider("Hour Range", 0, 23, (8, 10))
        barrios = st.multiselect("Select neighborhood(s)", df['barrio'].unique(), default=df['barrio'].unique()[:5])

    with col2:
        df_filtered = df[
            (df['dia_semana'] == dia) &
            (df['barrio'].isin(barrios)) &
            (df['hora'].between(hora_min, hora_max))
        ].copy()

        if df_filtered.empty:
            st.warning("No data for selected filters.")
        else:
            # Agrupamos para evitar duplicados por hora y segmento
            df_grouped = df_filtered.groupby(['barrio', 'dia_semana', 'hora']).agg(
                numero_plazas=('numero_plazas', 'median'),
                tramo_horario=('tramo_horario', 'first')
            ).reset_index()

            df_grouped['plazas_libres_pred'] = model.predict(df_grouped[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']])
            df_grouped['plazas_ocupadas_pred'] = df_grouped['numero_plazas'] - df_grouped['plazas_libres_pred']

            # Agregamos para el mapa
            agg = df_grouped.groupby('barrio').agg(
                plazas_ocupadas_predichas=('plazas_ocupadas_pred', 'sum')
            ).reset_index()

            # Conseguir coordenadas reales con geopy
            coords_df = get_coords(agg['barrio'].unique())
            agg_coords = agg.merge(coords_df, on='barrio', how='left')

            # Mapa con Folium
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

            # Tabla simple
            st.subheader("📋 Predicted Occupied Spots")
            tabla = agg.rename(columns={'plazas_ocupadas_predichas': 'plazas_predichas'})
            st.dataframe(tabla, use_container_width=True)

# ----------- TAB 2 -----------
with tabs[1]:
    st.subheader("Model Information")
    st.markdown("""
    **Model**: Ridge Regression  
    **Input features**: `barrio`, `dia_semana`, `tramo_horario`, `numero_plazas`  
    **Target**: `plazas_disponibles` (free spots)  
    **Prediction displayed**: `ocupadas = total - libres_pred`  
    """)

# ----------- TAB 3 -----------
with tabs[2]:
    subtab = st.radio("Choose a View", ["Individual Barrio", "Compare Barrios"])

    df['plazas_libres_pred'] = model.predict(df[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']])
    df['plazas_ocupadas_pred'] = df['numero_plazas'] - df['plazas_libres_pred']

    if subtab == "Individual Barrio":
        barrio_sel = st.selectbox("Choose a Neighborhood", df['barrio'].unique())
        fig = px.histogram(df[df['barrio'] == barrio_sel], x='plazas_ocupadas_pred', nbins=30,
                           title=f"Predicted Occupied Spots in {barrio_sel}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.box(df[df['barrio'].isin(barrios)], x='barrio', y='plazas_ocupadas_pred',
                     title="Comparative Occupied Spots by Barrio")
        st.plotly_chart(fig, use_container_width=True)
