import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import gzip
import plotly.express as px

# ---------- ESTILO ----------
st.set_page_config(layout="wide", page_title="Parking Spot Prediction")
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            color: #333333;
            background-color: #ffffff;
        }
        .stButton>button {
            color: white;
            background-color: #4472C4;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("Parking Spot Prediction in Madrid")

# ---------- DATA ----------
@st.cache_data
def load_data():
    with gzip.open("datos_plazas_disponibles_sin_prediccion.csv.gz", 'rt') as f:
        return pd.read_csv(f)

@st.cache_data
def load_coords():
    df_coords = pd.read_csv("coordenadas_barrios_madrid.csv")
    return df_coords.set_index("barrio")[["lat", "lon"]].to_dict(orient="index")

@st.cache_data
def build_model(df):
    X = df[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']]
    y = df['plazas_disponibles']
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['barrio', 'dia_semana', 'tramo_horario']),
        ('num', 'passthrough', ['numero_plazas'])
    ])
    model = make_pipeline(preprocessor, Ridge(alpha=1.0))
    model.fit(X, y)
    return model

# ---------- LOAD ----------
df = load_data()
coords_dict = load_coords()
model = build_model(df)

# ---------- TABS ----------
tabs = st.tabs(["Prediction Map", "Model Info", "Data Visuals"])

with tabs[0]:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Filter Options")
        dia = st.selectbox("Day of the week", sorted(df['dia_semana'].unique()))
        hora_min, hora_max = st.slider("Hour Range", 0, 23, (8, 10))

        lista_barrios = sorted(df['barrio'].unique())
        opciones_barrios = ["TODOS"] + lista_barrios
        seleccion = st.multiselect("Select neighborhoods", opciones_barrios, default=opciones_barrios[:6])
        barrios = lista_barrios if "TODOS" in seleccion else seleccion

    with col2:
        df_filtered = df[
            (df['dia_semana'] == dia) &
            (df['barrio'].isin(barrios)) &
            (df['hora'].between(hora_min, hora_max))
        ].copy()

        if df_filtered.empty:
            st.warning("No data for the selected filters.")
        else:
            df_grouped = df_filtered.groupby(['barrio', 'dia_semana', 'hora']).agg(
                numero_plazas=('numero_plazas', 'median'),
                tramo_horario=('tramo_horario', 'first')
            ).reset_index()

            df_grouped['plazas_libres_pred'] = model.predict(
                df_grouped[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']]
            )
            df_grouped['plazas_ocupadas_pred'] = df_grouped['numero_plazas'] - df_grouped['plazas_libres_pred']

            agg = df_grouped.groupby('barrio').agg(
                plazas_ocupadas_predichas=('plazas_ocupadas_pred', 'sum')
            ).reset_index()

            agg['lat'] = agg['barrio'].map(lambda b: coords_dict.get(b, {}).get('lat'))
            agg['lon'] = agg['barrio'].map(lambda b: coords_dict.get(b, {}).get('lon'))

            st.subheader("Predicted Occupied Spots by Neighborhood")
            m = folium.Map(location=[40.4168, -3.7038], zoom_start=12, tiles="cartodbpositron")

            max_val = agg['plazas_ocupadas_predichas'].max()
            for _, row in agg.iterrows():
                if pd.notnull(row['lat']) and pd.notnull(row['lon']):
                    radius = 5 + 15 * (row['plazas_ocupadas_predichas'] / max_val)
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=radius,
                        color='darkred',
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"{row['barrio']}: {int(row['plazas_ocupadas_predichas'])} occupied"
                    ).add_to(m)

            folium_static(m, width=1000, height=500)

            st.subheader("Predicted Values Table")
            tabla = agg[['barrio', 'plazas_ocupadas_predichas']].rename(
                columns={'barrio': 'Neighborhood', 'plazas_ocupadas_predichas': 'Occupied Spots'}
            )
            st.dataframe(tabla, use_container_width=True)

with tabs[1]:
    st.subheader("Model Details")
    st.markdown("""
    **Model**: Ridge Regression  
    **Input features**: `barrio`, `dia_semana`, `tramo_horario`, `numero_plazas`  
    **Target**: `plazas_disponibles` (free spots)  
    **Prediction used**: Occupied = Total - Predicted Free  
    """)

with tabs[2]:
    subtab = st.radio("Choose a View", ["Individual Neighborhood", "Compare Neighborhoods"])

    df['plazas_libres_pred'] = model.predict(df[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']])
    df['plazas_ocupadas_pred'] = df['numero_plazas'] - df['plazas_libres_pred']

    if subtab == "Individual Neighborhood":
        barrio_sel = st.selectbox("Select a Neighborhood", df['barrio'].unique())
        fig = px.histogram(df[df['barrio'] == barrio_sel], x='plazas_ocupadas_pred', nbins=30,
                           title=f"Predicted Occupied Spots in {barrio_sel}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.box(df[df['barrio'].isin(barrios)], x='barrio', y='plazas_ocupadas_pred',
                     title="Occupied Spots Comparison")
        st.plotly_chart(fig, use_container_width=True)
