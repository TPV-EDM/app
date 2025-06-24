import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import gzip
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(layout="wide", page_title="Parking Spot Prediction in Madrid")
st.title("Parking Spot Prediction in Madrid")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    with gzip.open("datos_plazas_disponibles_sin_prediccion.csv.gz", 'rt') as f:
        return pd.read_csv(f)

@st.cache_data
def load_coords():
    return pd.read_csv("coordenadas_barrios_madrid.csv")

# ---------- BUILD MODEL ----------
def build_model(df):
    X = df[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']]
    y = df['plazas_disponibles']
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['barrio', 'dia_semana', 'tramo_horario']),
        ('num', 'passthrough', ['numero_plazas'])
    ])
    model = make_pipeline(preprocessor, Ridge(alpha=1.0))
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = {
        'MAE': mean_absolute_error(y, y_pred),
        'RMSE': mean_squared_error(y, y_pred, squared=False),
        'R2': r2_score(y, y_pred)
    }
    return model, metrics

# ---------- LOAD ----------
df = load_data()
coords_df = load_coords()
model, metrics = build_model(df)

# ---------- PREDICTIONS ----------
df['plazas_libres_pred'] = model.predict(df[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']])
df['plazas_ocupadas_pred'] = df['numero_plazas'] - df['plazas_libres_pred']

# ---------- TABS ----------
tabs = st.tabs(["Prediction Map", "Model Info", "Data Visuals"])

# ---------- MAP TAB ----------
with tabs[0]:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Filter Options")
        dia = st.selectbox("Day of the week", sorted(df['dia_semana'].unique()))
        hora_min, hora_max = st.slider("Hour Range", 0, 23, (8, 10))

        lista_barrios = sorted(df['barrio'].unique())
        mostrar_todos = st.checkbox("Show all neighborhoods", value=False)
        seleccion = st.multiselect("Select neighborhoods", lista_barrios, default=lista_barrios[:5])
        barrios = lista_barrios if mostrar_todos else seleccion

    with col2:
        df_filtered = df[
            (df['dia_semana'] == dia) &
            (df['barrio'].isin(barrios)) &
            (df['hora'].between(hora_min, hora_max))
        ].copy()

        if df_filtered.empty:
            st.warning("No data for the selected filters.")
        else:
            df_grouped = df_filtered.groupby(['barrio']).agg(
                numero_plazas=('numero_plazas', 'median'),
                plazas_ocupadas_pred=('plazas_ocupadas_pred', 'sum')
            ).reset_index()

            agg_coords = df_grouped.merge(coords_df, on='barrio', how='left')
            st.subheader("Predicted Occupied Spots by Neighborhood")

            m = folium.Map(location=[40.4168, -3.7038], zoom_start=12, tiles="cartodbpositron")

            for _, row in agg_coords.iterrows():
                if pd.notnull(row['lat']) and pd.notnull(row['lon']):
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=np.interp(row['plazas_ocupadas_pred'], [df['plazas_ocupadas_pred'].min(), df['plazas_ocupadas_pred'].max()], [4, 20]),
                        color='red',
                        fill=True,
                        fill_opacity=0.6,
                        popup=f"{row['barrio']}: {int(row['plazas_ocupadas_pred'])} occupied"
                    ).add_to(m)

            folium_static(m, width=1000, height=500)

            st.subheader("Predicted Values Table")
            df_table = agg_coords[['barrio', 'plazas_ocupadas_pred']].rename(columns={'plazas_ocupadas_pred': 'occupied_spots'})
            st.dataframe(df_table, use_container_width=True)

# ---------- MODEL INFO TAB ----------
with tabs[1]:
    st.subheader("Model Details and Metrics")
    st.markdown("""
    **Model**: Ridge Regression  
    **Input features**: `barrio`, `dia_semana`, `tramo_horario`, `numero_plazas`  
    **Target**: `plazas_disponibles` (free spots)  
    **Prediction used**: Occupied = Total - Predicted Free  
    """)

    st.metric("MAE (Mean Absolute Error)", f"{metrics['MAE']:.2f}")
    st.metric("RMSE (Root Mean Squared Error)", f"{metrics['RMSE']:.2f}")
    st.metric("R² Score", f"{metrics['R2']:.3f}")

    st.subheader("Error Distribution")
    errors = df['numero_plazas'] - df['plazas_libres_pred']
    fig = px.histogram(errors, nbins=40, title="Prediction Errors")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction vs Real Values")
    fig2 = px.scatter(df, x='plazas_libres_pred', y='plazas_disponibles',
                      title="Predicted vs Actual Free Spots",
                      labels={"plazas_libres_pred": "Predicted", "plazas_disponibles": "Actual"})
    st.plotly_chart(fig2, use_container_width=True)

# ---------- DATA VISUALS TAB ----------
with tabs[2]:
    st.subheader("Compare Neighborhoods")
    barrio_sel = st.selectbox("Select a Neighborhood", sorted(df['barrio'].unique()))
    df_barrio = df[df['barrio'] == barrio_sel]

    fig = px.line(df_barrio, x='hora', y='plazas_ocupadas_pred', color='dia_semana',
                  title=f"Predicted Occupancy Through the Day - {barrio_sel}")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(df[df['barrio'].isin(barrios)], x='barrio', y='plazas_ocupadas_pred',
                  title="Occupied Spots Distribution by Neighborhood")
    st.plotly_chart(fig2, use_container_width=True)
