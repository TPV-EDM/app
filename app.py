import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_absolute_percentage_error
import gzip
import plotly.express as px

# ---------- CONFIG ----------
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

# ---------- LOAD ----------
@st.cache_data
def load_data():
    with gzip.open("datos_plazas_disponibles_sin_prediccion.csv.gz", 'rt') as f:
        return pd.read_csv(f)

@st.cache_data
def load_coords():
    return pd.read_csv("coordenadas_barrios_madrid.csv")  # Must include: barrio, lat, lon

df = load_data()
coords_df = load_coords()

# ---------- MODEL ----------
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

model = build_model(df)
df['plazas_libres_pred'] = model.predict(df[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']])
df['plazas_ocupadas_pred'] = df['numero_plazas'] - df['plazas_libres_pred']

# ---------- TABS ----------
tabs = st.tabs(["Prediction Map", "Model Info", "Data Visuals"])

with tabs[0]:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Filter Options")
        dia = st.selectbox("Day of the week", sorted(df['dia_semana'].unique()))
        hora_min, hora_max = st.slider("Hour Range", 0, 23, (8, 10))

        lista_barrios = sorted(df['barrio'].unique())
        seleccion = st.multiselect("Select neighborhoods", lista_barrios[:6], options=lista_barrios)
        mostrar_todos = st.checkbox("Show all neighborhoods")

        barrios_filtrados = lista_barrios if mostrar_todos else seleccion

    with col2:
        df_filtered = df[
            (df['dia_semana'] == dia) &
            (df['barrio'].isin(barrios_filtrados)) &
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
                occupied_spots=('plazas_ocupadas_pred', 'sum')
            ).reset_index()

            agg_coords = agg.merge(coords_df, on='barrio', how='left')

            st.subheader("Predicted Occupied Spots by Neighborhood")
            m = folium.Map(location=[40.4168, -3.7038], zoom_start=12, tiles="cartodbpositron")

            max_radius = agg_coords['occupied_spots'].max()
            for _, row in agg_coords.iterrows():
                if pd.notnull(row['lat']) and pd.notnull(row['lon']):
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=5 + 10 * (row['occupied_spots'] / max_radius),
                        color='red',
                        fill=True,
                        fill_opacity=0.6,
                        popup=f"{row['barrio']}: {int(row['occupied_spots'])} occupied"
                    ).add_to(m)

            folium_static(m, width=1000, height=500)

            st.subheader("Predicted Values Table")
            tabla = agg_coords[['barrio', 'occupied_spots']].sort_values('occupied_spots', ascending=False)
            st.dataframe(tabla, use_container_width=True)

with tabs[1]:
    st.subheader("Model Details")

    y_true = df['plazas_disponibles']
    y_pred = df['plazas_libres_pred']

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mean_absolute_error(y_true, y_pred):.2f}")
    col2.metric("Median AE", f"{median_absolute_error(y_true, y_pred):.2f}")
    col3.metric("MAPE", f"{mean_absolute_percentage_error(y_true, y_pred) * 100:.2f}%")

    st.markdown("### Prediction vs. Actual")
    fig = px.scatter(df.sample(2000), x='plazas_libres_pred', y='plazas_disponibles',
                     trendline="ols", opacity=0.4,
                     labels={"plazas_libres_pred": "Predicted", "plazas_disponibles": "Actual"})
    fig.add_shape(type='line', x0=0, y0=0, x1=df['plazas_disponibles'].max(),
                  y1=df['plazas_disponibles'].max(), line=dict(dash='dash'))
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    subtab = st.radio("Choose a View", ["Compare Neighborhoods", "Hourly Evolution"])

    if subtab == "Compare Neighborhoods":
        st.markdown("### Occupied Spots by Neighborhood")
        fig = px.box(df[df['barrio'].isin(barrios_filtrados)], x='barrio', y='plazas_ocupadas_pred',
                     title="Occupied Spots Distribution")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("### Average Occupancy by Hour")
        hourly = df[df['barrio'].isin(barrios_filtrados)].groupby('hora').agg(
            avg_occupied=('plazas_ocupadas_pred', 'mean')).reset_index()
        fig = px.line(hourly, x='hora', y='avg_occupied', title="Hourly Occupancy Trend")
        st.plotly_chart(fig, use_container_width=True)
