import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import plotly.express as px
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_absolute_percentage_error

# ----- CONFIG -----
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

# ----- LOAD DATA -----
@st.cache_data
def load_data():
    return pd.read_csv("datos_plazas_disponibles_sin_prediccion.csv.gz")

@st.cache_data
def load_coords():
    return pd.read_csv("coordenadas_barrios_madrid.csv")

df = load_data()
coords_df = load_coords()

# ----- TRAIN MODEL -----
def train_model(df):
    X = df[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']]
    y = df['plazas_disponibles']
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['barrio', 'dia_semana', 'tramo_horario']),
        ('num', 'passthrough', ['numero_plazas'])
    ])
    model = make_pipeline(preprocessor, Ridge(alpha=1.0))
    model.fit(X, y)
    return model

model = train_model(df)
df['plazas_libres_pred'] = model.predict(df[['barrio', 'dia_semana', 'tramo_horario', 'numero_plazas']])
df['plazas_ocupadas_pred'] = df['numero_plazas'] - df['plazas_libres_pred']

# ----- TABS -----
tabs = st.tabs(["Prediction Map", "Model Info", "Data Visuals"])

# ------------------- TAB 1: MAP -------------------
with tabs[0]:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Filter Options")
        dia = st.selectbox("Day of the week", sorted(df['dia_semana'].unique()))
        hora_min, hora_max = st.slider("Hour Range", 0, 23, (8, 10))
        lista_barrios = sorted(df['barrio'].unique())
        seleccion = st.multiselect("Select neighborhoods", options=lista_barrios, default=lista_barrios[:6])
        mostrar_todos = st.checkbox("Show all neighborhoods")

        barrios_filtrados = lista_barrios if mostrar_todos else seleccion

    with col2:
        df_filtrado = df[
            (df['dia_semana'] == dia) &
            (df['hora'].between(hora_min, hora_max)) &
            (df['barrio'].isin(barrios_filtrados))
        ].copy()

        if df_filtrado.empty:
            st.warning("No data matches the selected filters.")
        else:
            agg = df_filtrado.groupby('barrio').agg(
                plazas_ocupadas_predichas=('plazas_ocupadas_pred', 'sum')
            ).reset_index()

            # Merge coords
            agg = agg.merge(coords_df, on='barrio', how='left')

            st.subheader("Predicted Occupied Spots by Neighborhood")
            m = folium.Map(location=[40.4168, -3.7038], zoom_start=12, tiles="cartodbpositron")

            max_val = agg['plazas_ocupadas_predichas'].max()

            for _, row in agg.iterrows():
                if pd.notnull(row['lat']) and pd.notnull(row['lon']):
                    scaled_radius = np.interp(row['plazas_ocupadas_predichas'], [0, max_val], [4, 20])
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=scaled_radius,
                        color='crimson',
                        fill=True,
                        fill_opacity=0.6,
                        popup=f"{row['barrio']}: {int(round(row['plazas_ocupadas_predichas']))} occupied"
                    ).add_to(m)

            folium_static(m, width=1000, height=500)

            st.subheader("Predicted Values Table")
            tabla = agg[['barrio', 'plazas_ocupadas_predichas']].copy()
            tabla['plazas_ocupadas_predichas'] = tabla['plazas_ocupadas_predichas'].round().astype(int)
            tabla.columns = ['Neighborhood', 'Occupied spots']
            st.dataframe(tabla, use_container_width=True)
# ------------------- TAB 2: MODEL INFO -------------------
with tabs[1]:
    st.subheader("Model Details")
    st.markdown("""
    **Model:** Ridge Regression  
    **Input features:** `barrio`, `dia_semana`, `tramo_horario`, `numero_plazas`  
    **Target:** `plazas_disponibles` (free spots)  
    **Prediction used:** Occupied = Total - Predicted Free  
    """)

    mae = mean_absolute_error(df['plazas_disponibles'], df['plazas_libres_pred'])
    medae = median_absolute_error(df['plazas_disponibles'], df['plazas_libres_pred'])
    mape = mean_absolute_percentage_error(df['plazas_disponibles'], df['plazas_libres_pred'])

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("Median AE", f"{medae:.2f}")
    col3.metric("MAPE", f"{mape*100:.2f}%")

    st.subheader("Prediction vs. Actual")
    fig = px.scatter(
        df.sample(2000),
        x='plazas_libres_pred',
        y='plazas_disponibles',
        opacity=0.4,
        color_discrete_sequence=["#4472C4"],
        labels={"plazas_libres_pred": "Predicted", "plazas_disponibles": "Actual"},
        title="Predicted vs. Actual Values"
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Data Visualizations")

    # Rename time slots to English
    df['time_slot_en'] = df['tramo_horario'].replace({
        'mañana': 'Morning',
        'mediodia': 'Midday',
        'tarde': 'Afternoon',
        'noche': 'Evening'
    })

    sub_tab = st.radio("Choose a visualization", ["Spot Distribution", "Neighborhood Evolution"])

    if sub_tab == "Spot Distribution":
        st.markdown("### Distribution of Free Parking Spots")

        fig = px.histogram(df, x='plazas_disponibles', nbins=50,
                           title="Overall Distribution",
                           labels={"plazas_disponibles": "Free Spots"})
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(df, x='time_slot_en', y='plazas_disponibles',
                      title="Free Spots by Time Slot",
                      labels={"plazas_disponibles": "Free Spots", "time_slot_en": "Time Slot"})
        st.plotly_chart(fig2, use_container_width=True)

    elif sub_tab == "Neighborhood Evolution":
        st.markdown("### Hourly Evolution by Neighborhood")

        barrio_selected = st.selectbox("Select a neighborhood", sorted(df['barrio'].unique()))
        df_barrio = df[df['barrio'] == barrio_selected]
        avg_by_hour = df_barrio.groupby('hora')['plazas_disponibles'].mean().reset_index()

        fig = px.line(avg_by_hour, x='hora', y='plazas_disponibles',
                      markers=True,
                      title=f"Hourly Average Free Spots in {barrio_selected}",
                      labels={"hora": "Hour", "plazas_disponibles": "Free Spots"})
        st.plotly_chart(fig, use_container_width=True)
