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
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif !important;
            background-color: #fffce6 !important;  /* Fondo amarillo claro */
            color: #1a2e52 !important;  /* Texto en azul oscuro */
        }
        .stButton > button {
            color: white !important;
            background-color: #2e6eb5 !important;  /* Azul elegante */
            border-radius: 5px !important;
            border: none !important;
        }
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 16px !important;
            color: #2e6eb5 !important;
            font-weight: bold !important;
        }
        h1, h2, h3 {
            color: #1a2e52 !important;  /* Azul oscuro para títulos */
        }
        .block-container {
            padding-top: 2rem !important;
        }
        .stDataFrame thead tr th {
            background-color: #cbe0ff !important;
            color: #1a2e52 !important;
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

    **Interpretation of Metrics:**
    - **MAE (Mean Absolute Error):** On average, the prediction differs by this many spots. Lower is better.
    - **Median AE:** The middle value of all absolute errors, less sensitive to outliers.
    - **MAPE (Mean Absolute Percentage Error):** Average percentage error; here it's under 1%, which indicates excellent relative performance.
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
    st.subheader("Data Visuals")

    df['dia_hora'] = df['dia_semana'] + " - " + df['hora'].astype(str) + "h"
    df['is_weekend'] = df['dia_semana'].isin(['Saturday', 'Sunday'])

    subtab = st.radio("Explore:", [
        "By Time",
        "Explore a Neighborhood",
        "Busiest Neighborhoods",
        "Hourly Evolution Animation",
        "Neighborhood Daily Animation",
        "Weekend vs Weekdays",
        "Animated Insights"
    ])

    if subtab == "By Time":
        df_time = df.groupby(['dia_semana', 'hora'])['plazas_disponibles'].mean().reset_index()
        fig1 = px.line(df_time, x='hora', y='plazas_disponibles', color='dia_semana',
                       title="Hourly Free Spots by Day",
                       labels={"plazas_disponibles": "Free Spots", "hora": "Hour", "dia_semana": "Day"})
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.box(df, x='tramo_horario', y='plazas_disponibles',
                      title="Free Spots by Time Slot",
                      labels={"plazas_disponibles": "Free Spots", "tramo_horario": "Time Slot"})
        st.plotly_chart(fig2, use_container_width=True)

    elif subtab == "Explore a Neighborhood":
        barrio_elegido = st.selectbox("Choose a Neighborhood", sorted(df['barrio'].unique()))
        df_barrio = df[df['barrio'] == barrio_elegido]
        df_line = df_barrio.groupby(['dia_semana', 'hora'])['plazas_disponibles'].mean().reset_index()

        fig3 = px.line(df_line, x='hora', y='plazas_disponibles', color='dia_semana',
                       title=f"Hourly Evolution of Free Spots in {barrio_elegido}",
                       labels={"plazas_disponibles": "Free Spots", "hora": "Hour", "dia_semana": "Day"})
        st.plotly_chart(fig3, use_container_width=True)

    elif subtab == "Busiest Neighborhoods":
        df_busiest = df.groupby("barrio")['plazas_disponibles'].mean().reset_index()
        fig4 = px.bar(df_busiest.sort_values("plazas_disponibles").head(10),
                      x="plazas_disponibles", y="barrio", orientation='h',
                      labels={"plazas_disibles": "Avg. Free Spots", "barrio": "Neighborhood"},
                      title="Top 10 Busiest Neighborhoods (Fewer Free Spots)")
        st.plotly_chart(fig4, use_container_width=True)

    elif subtab == "Hourly Evolution Animation":
        top_barrios = df['barrio'].value_counts().head(5).index.tolist()
        df_subset = df[df['barrio'].isin(top_barrios)]
        df_anim = df_subset.groupby(['barrio', 'dia_semana', 'hora'])['plazas_disponibles'].mean().reset_index()

        fig5 = px.line(df_anim, x='hora', y='plazas_disponibles', color='barrio',
                       animation_frame='dia_semana', markers=True,
                       title="Animated Hourly Evolution in Top Neighborhoods",
                       labels={"hora": "Hour", "plazas_disponibles": "Free Spots", "barrio": "Neighborhood"})
        st.plotly_chart(fig5, use_container_width=True)

    elif subtab == "Neighborhood Daily Animation":
        barrio_anim = st.selectbox("Choose a Neighborhood", sorted(df['barrio'].unique()), key="ani_barrio")
        df_anim_barrio = df[df['barrio'] == barrio_anim]
        df_anim_day = df_anim_barrio.groupby(['dia_semana', 'hora'])['plazas_disponibles'].mean().reset_index()

        fig6 = px.line(df_anim_day, x='hora', y='plazas_disponibles',
                       animation_frame='dia_semana', markers=True,
                       title=f"Animated Daily Evolution in {barrio_anim}",
                       labels={"hora": "Hour", "plazas_disponibles": "Free Spots"})
        st.plotly_chart(fig6, use_container_width=True)

    elif subtab == "Weekend vs Weekdays":
        df_grouped = df.groupby(['is_weekend', 'hora'])['plazas_disponibles'].mean().reset_index()

        fig7 = px.line(df_grouped, x='hora', y='plazas_disponibles', color='is_weekend',
                       labels={"hora": "Hour", "plazas_disponibles": "Free Spots", "is_weekend": "Weekend?"},
                       title="Free Spots: Weekend vs Weekdays")
        st.plotly_chart(fig7, use_container_width=True)

    elif subtab == "Animated Insights":
        st.markdown("### Top 10 Neighborhoods by Hour (Animated)")
        df_top10 = df.groupby(['hora', 'barrio'])['plazas_disponibles'].mean().reset_index()
        top10_all = df_top10.groupby('barrio')['plazas_disponibles'].mean().nlargest(10).index.tolist()
        df_top10_filtered = df_top10[df_top10['barrio'].isin(top10_all)]

        fig8 = px.bar(df_top10_filtered, x='plazas_disponibles', y='barrio', color='barrio',
                      animation_frame='hora', orientation='h',
                      labels={"plazas_disponibles": "Free Spots", "barrio": "Neighborhood"},
                      title="Animated Hourly Ranking of Top Neighborhoods")
        st.plotly_chart(fig8, use_container_width=True)

        st.markdown("### Daily Evolution of Top 5 Neighborhoods")
        top5_barr = df['barrio'].value_counts().head(5).index.tolist()
        df_day_anim = df[df['barrio'].isin(top5_barr)]
        df_day_anim_grouped = df_day_anim.groupby(['barrio', 'dia_semana'])['plazas_disponibles'].mean().reset_index()

        fig9 = px.bar(df_day_anim_grouped, x='dia_semana', y='plazas_disponibles', color='barrio',
                      barmode='group',
                      labels={"plazas_disponibles": "Free Spots", "dia_semana": "Day"},
                      title="Average Free Spots by Day for Top Neighborhoods")
        st.plotly_chart(fig9, use_container_width=True)
