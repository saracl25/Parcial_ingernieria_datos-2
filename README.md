# 🏠 Análisis Exploratorio y Dashboard Predictivo de Viviendas

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)

Este proyecto consiste en un Análisis Exploratorio de Datos (EDA), un proceso de Extracción, Transformación y Carga (ETL), y un **Dashboard Interactivo** construido en **Streamlit** sobre el conjunto de datos de viviendas de California.

**Creado por:** Sara Cardona  
**Repositorio Original:** [Parcial_ingernieria_datos-2](https://github.com/saracl25/Parcial_ingernieria_datos-2.git)

---

## 🌟 Características del Proyecto

- **Landing Page (`index.html`)**: Una página web moderna que explica paso a paso el contenido y los hallazgos del notebook.
- **Dashboard en Streamlit (`app.py`)**: 
  - 📊 **Datos y Frecuencias:** Visualización de métricas generales y tablas de frecuencia.
  - 📈 **Gráficos Descriptivos:** Gráficos de barras, circulares, cajas (boxplots) e histogramas.
  - 🗺️ **Mapa Geoespacial:** Representación de las viviendas según longitud, latitud y valor en el mapa.
  - 🔗 **Análisis de Correlación:** Matriz de calor con la relación entre variables numéricas.
  - 🤖 **Predicción en Tiempo Real:** Un modelo predictivo Random Forest integrado que estima el precio de una vivienda en base a los parámetros ingresados.
  - 🔍 **Filtros Globales:** Filtra dinámicamente los datos por proximidad al océano y rango de edad de la vivienda.

---

## 🚀 Cómo Ejecutar el Proyecto (Instrucciones de Clonación)

Sigue estos pasos para clonar el repositorio, instalar las dependencias y ejecutar la aplicación de Streamlit de forma local.

### 1. Clonar el repositorio

Abre tu terminal o consola de comandos y ejecuta el siguiente comando:

```bash
git clone https://github.com/saracl25/Parcial_ingernieria_datos-2.git
```

### 2. Acceder al directorio del proyecto

```bash
cd Parcial_ingernieria_datos-2
```
*(Si los archivos están en una carpeta específica como `Cuaderno`, asegúrate de navegar hacia ella: `cd Cuaderno`)*

### 3. Crear un entorno virtual (Recomendado)

Es una buena práctica utilizar un entorno virtual de Python:

```bash
# En Windows:
python -m venv env
env\Scripts\activate

# En macOS/Linux:
python3 -m venv env
source env/bin/activate
```

### 4. Instalar las dependencias requeridas

Instala las librerías indicadas en el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 5. Ejecutar la Aplicación Streamlit

Finalmente, lanza el dashboard con el siguiente comando:

```bash
streamlit run app.py
```

El navegador se abrirá automáticamente (normalmente en `http://localhost:8501`) mostrando el panel interactivo. También puedes abrir el archivo `index.html` en tu navegador favorito haciendo doble clic en él para visualizar la Landing Page.

---

## 🗂️ Estructura de Archivos Principal

- `Tarea_ingeniería.ipynb`: El cuaderno original con el EDA y proceso ETL.
- `index.html`: La landing page con las "fichas de paso".
- `app.py`: La aplicación interactiva principal construida con Streamlit.
- `requirements.txt`: Listado de librerías necesarias.
- `cleaned_housing.csv`: Dataset generado después de aplicar el ETL (se crea automáticamente si no existe).
