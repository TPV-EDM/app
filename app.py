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

# ---------- PAGE STYLE ----------
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

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    with gzip.open("datos_plazas_disponibles_sin_prediccion.csv.gz", 'rt') as f:
        return pd.read_csv(f)

@st.cache_data
def load_coords():
    return pd.read_csv("coordenadas_barrios.csv")

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

# ---------- INIT ----------
df = load_data()
coords_df = load_coords()
model = build_model(df)

# ---------- TABS ----------
tabs = st.tabs(["Prediction Map", "Model Info", "Data Visuals"])

with tabs[0]:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Filter Options")
        day = st.selectbox("Day of the week", sorted(df['dia_semana'].unique()))
        hour_min, hour_max = st.slider("Hour Range", 0, 23, (8, 10))

        neighborhoods_list = sorted(df['barrio'].unique())
        neighborhood_options = neighborhoods_list
        selected_neighborhoods = st.multiselect("Select neighborhoods", neighborhood_options, default=neighborhood_options[:6])
        show_all = st.checkbox("Show all neighborhoods")

        if show_all:
            neighborhoods = neighborhoods_list
        else:
            neighborhoods = [b for b in selected_neighborhoods if b in neighborhoods_list]

    with col2:
        df_filtered = df[
            (df['dia_semana'] == day) &
            (df['barrio'].isin(neighborhoods)) &
            (df['hora'].between(hour_min, hour_max))
        ].copy()

        if df_filtered.empty:
            st.warning("No data available for the selected filters.")
        else:
            df_filtered['plazas_libres_pred'] = model.predict(
                df_filtered[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']]
            )
            df_filtered['plazas_ocupadas_pred'] = df_filtered['numero_plazas'] - df_filtered['plazas_libres_pred']

            agg = df_filtered.groupby('barrio').agg(
                predicted_occupied_spots=('plazas_ocupadas_pred', 'sum')
            ).reset_index()

            merged = agg.merge(coords_df, on='barrio', how='left')

            st.subheader("Predicted Occupied Spots by Neighborhood")
            m = folium.Map(location=[40.4168, -3.7038], zoom_start=12, tiles="cartodbpositron")

            max_value = merged['predicted_occupied_spots'].max()
            min_value = merged['predicted_occupied_spots'].min()
            range_value = max_value - min_value if max_value != min_value else 1

            for _, row in merged.iterrows():
                if pd.notnull(row['lat']) and pd.notnull(row['lon']):
                    norm_radius = 5 + 15 * (row['predicted_occupied_spots'] - min_value) / range_value
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=norm_radius,
                        color='red',
                        fill=True,
                        fill_opacity=0.6,
                        popup=f"{row['barrio']}: {int(row['predicted_occupied_spots'])} occupied"
                    ).add_to(m)

            folium_static(m, width=1000, height=500)

            st.subheader("Predicted Values Table")
            st.dataframe(merged.rename(columns={"barrio": "neighborhood"}), use_container_width=True)

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
        selected = st.selectbox("Select a Neighborhood", df['barrio'].unique())
        fig = px.histogram(df[df['barrio'] == selected], x='plazas_ocupadas_pred', nbins=30,
                           title=f"Predicted Occupied Spots in {selected}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.box(df[df['barrio'].isin(neighborhoods)], x='barrio', y='plazas_ocupadas_pred',
                     title="Occupied Spots Comparison")
        st.plotly_chart(fig, use_container_width=True)
