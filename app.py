import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Dashboard de Viviendas", page_icon="🏠", layout="wide")

# --- ESTILOS PROFESIONALES ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #fff; border-radius: 4px 4px 0px 0px; padding: 10px 20px; box-shadow: 0px 2px 5px rgba(0,0,0,0.05); }
    .stTabs [aria-selected="true"] { background-color: #4F46E5 !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# --- CARGA Y CACHÉ DE DATOS ---
@st.cache_data
def load_data():
    # Intentar cargar dataset limpio si existe, si no, descargar y transformar
    if os.path.exists("cleaned_housing.csv"):
        df = pd.read_csv("cleaned_housing.csv")
    else:
        path = kagglehub.dataset_download("yadavhim/housing-csv")
        df = pd.read_csv(path + "/housing.csv")
        
        # Proceso ETL básico
        df = df.dropna()
        df["price_category"] = pd.cut(
            df["median_house_value"],
            bins=[0, 150000, 300000, 500000, 1000000],
            labels=["Bajo", "Medio", "Alto", "Muy Alto"]
        )
        df.to_csv("cleaned_housing.csv", index=False)
    return df

df = load_data()

# --- MODELO PREDICTIVO ---
@st.cache_resource
def train_model(data):
    # Variables predictoras simples
    features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                'total_bedrooms', 'population', 'households', 'median_income']
    X = data[features]
    y = data['median_house_value']
    
    # Manejo de nulos para el modelo
    X = X.fillna(X.median())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, features

model, model_features = train_model(df)

# --- PANEL DE NAVEGACIÓN LATERAL (FILTROS) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/25/25694.png", width=100)
st.sidebar.title("Filtros Globales")

ocean_filter = st.sidebar.multiselect("Proximidad al Océano", options=df['ocean_proximity'].unique(), default=df['ocean_proximity'].unique())
age_filter = st.sidebar.slider("Edad Media de la Vivienda", int(df['housing_median_age'].min()), int(df['housing_median_age'].max()), (10, 40))

# Aplicar filtros
filtered_df = df[
    (df['ocean_proximity'].isin(ocean_filter)) &
    (df['housing_median_age'] >= age_filter[0]) & 
    (df['housing_median_age'] <= age_filter[1])
]

# --- TÍTULO PRINCIPAL ---
st.title("🏠 Análisis y Predicción: Viviendas en California")
st.markdown("**Desarrollado por:** Sara Cardona")

# --- FICHAS DE PASO (TABS) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Datos y Frecuencias", 
    "📈 Gráficos Descriptivos", 
    "🗺️ Mapa Geoespacial", 
    "🔗 Análisis de Correlación",
    "🤖 Predicción en Tiempo Real"
])

# PASO 1: DATOS Y FRECUENCIAS
with tab1:
    st.header("Exploración de Datos (EDA)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Registros", f"{len(filtered_df):,}")
    col2.metric("Precio Promedio", f"${filtered_df['median_house_value'].mean():,.2f}")
    col3.metric("Edad Promedio", f"{filtered_df['housing_median_age'].mean():.1f} años")
    
    st.subheader("Muestra de Datos (Head/Tail)")
    st.dataframe(filtered_df.head(10), use_container_width=True)
    
    st.subheader("Tabla de Frecuencias: Categoría de Precios")
    freq_table = filtered_df['price_category'].value_counts().reset_index()
    freq_table.columns = ['Categoría de Precio', 'Cantidad']
    st.table(freq_table)

# PASO 2: GRÁFICOS DESCRIPTIVOS
with tab2:
    st.header("Visualizaciones Descriptivas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Proporción por Proximidad al Océano")
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = sns.color_palette('pastel')
        filtered_df['ocean_proximity'].value_counts().plot.pie(autopct='%1.1f%%', colors=colors, ax=ax, startangle=90)
        ax.set_ylabel('')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Distribución del Precio (Boxplot)")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=filtered_df["median_house_value"], ax=ax2, color="skyblue")
        ax2.set_xlabel("Valor Medio de Vivienda")
        st.pyplot(fig2)

    st.subheader("Distribución de Edad de Viviendas")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.histplot(filtered_df['housing_median_age'], bins=30, kde=True, color="indigo", ax=ax3)
    ax3.set_xlabel("Edad Media")
    st.pyplot(fig3)

# PASO 3: MAPA GEOESPACIAL
with tab3:
    st.header("Distribución Geográfica de Precios")
    st.markdown("El mapa muestra la ubicación de las casas. El color y tamaño se basan en la densidad y los precios, reflejando el alto valor en zonas costeras.")
    
    # Streamlit native map requires specific columns 'lat' and 'lon'
    map_data = filtered_df[['latitude', 'longitude', 'median_house_value']].copy()
    map_data.columns = ['lat', 'lon', 'valor']
    
    # Para visualizacion mejorada en map nativo
    st.map(map_data, size=10, color="#4F46E5")

# PASO 4: CORRELACIÓN
with tab4:
    st.header("Matriz de Correlación")
    st.markdown("Analiza la relación entre variables numéricas. Colores cálidos indican correlación positiva.")
    
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    numeric_df = filtered_df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False, ax=ax4)
    st.pyplot(fig4)

# PASO 5: PREDICCIÓN EN TIEMPO REAL
with tab5:
    st.header("Modelo Predictivo: Random Forest")
    st.markdown("Ingresa las características de la zona para estimar el valor medio de la vivienda.")
    
    with st.form("predict_form"):
        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            p_lon = st.number_input("Longitud", value=float(df['longitude'].median()))
            p_lat = st.number_input("Latitud", value=float(df['latitude'].median()))
            p_age = st.number_input("Edad de Vivienda", value=int(df['housing_median_age'].median()))
            p_rooms = st.number_input("Total Habitaciones", value=int(df['total_rooms'].median()))
            
        with p_col2:
            p_bed = st.number_input("Total Dormitorios", value=int(df['total_bedrooms'].median()))
            p_pop = st.number_input("Población", value=int(df['population'].median()))
            p_house = st.number_input("Hogares", value=int(df['households'].median()))
            p_inc = st.number_input("Ingreso Medio (en decenas de miles)", value=float(df['median_income'].median()))
            
        submit_button = st.form_submit_button(label='🔮 Predecir Precio de Vivienda')
        
        if submit_button:
            input_data = pd.DataFrame([[p_lon, p_lat, p_age, p_rooms, p_bed, p_pop, p_house, p_inc]], columns=model_features)
            prediction = model.predict(input_data)[0]
            
            st.success(f"### Precio Estimado: ${prediction:,.2f}")
            st.info("Nota: Este es un modelo de demostración basado en el dataset de California Housing.")
