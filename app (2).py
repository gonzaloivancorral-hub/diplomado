# ==========================================================================
# PANEL DE RESEÑAS - aplicación completa
# ==========================================================================
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from wordcloud import WordCloud, STOPWORDS

nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

COLORES = {"Positivo": "#17A2A2", "Neutral": "#E8A33D", "Negativo": "#C0392B"}
MAPAS = {"Positivo": "Greens", "Neutral": "Oranges", "Negativo": "Reds"}
ORDEN = ["Positivo", "Neutral", "Negativo"]
IGNORAR = set(STOPWORDS) | {"backpack", "bag", "product", "one", "will", "now"}


@st.cache_data
def cargar_datos():
    sia = SentimentIntensityAnalyzer()
    # Archivo de mochilas configurado correctamente
    datos = pd.read_csv("Backpack_Reviews_with_Ratings (5).csv")
    
    datos["Compound"] = datos["Review"].apply(
        lambda texto: sia.polarity_scores(str(texto))["compound"])
    datos["Sentimiento"] = datos["Compound"].apply(
        lambda c: "Positivo" if c >= 0.05 else ("Negativo" if c <= -0.05 else "Neutral"))
    return datos


@st.cache_data(ttl=1800)
def obtener_clima():
    respuesta = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": -32.93, "longitude": -71.13,
                "daily": "temperature_2m_max,temperature_2m_min",
                "timezone": "auto", "past_days": 7, "forecast_days": 7},
        timeout=10)
    respuesta.raise_for_status()
    datos = respuesta.json()["daily"]
    return pd.DataFrame({"Fecha": pd.to_datetime(datos["time"]),
                         "T_Max": datos["temperature_2m_max"],
                         "T_Min": datos["temperature_2m_min"]})


st.set_page_config(page_title="Mochilas - Reseñas", layout="wide")

# --- Título de la página ---------------------------------------------------
st.title("Mochilas - Panel de Reseñas de Clientes")

# --- Descripción de la página ----------------------------------------------
st.write("Este panel muestra el análisis de sentimiento de las reseñas de clientes sobre las "
         "mochilas, clasificadas con VADER en tres categorías.")

df = cargar_datos()

# --- Visualización 1: reseñas por categoría --------------------------------
st.subheader("Reseñas por categoría de sentimiento")

conteo = df["Sentimiento"].value_counts().reset_index()
conteo.columns = ["Sentimiento", "Cantidad"]

fig = px.bar(conteo, x="Sentimiento", y="Cantidad", color="Sentimiento",
             color_discrete_map=COLORES,
             category_orders={"Sentimiento": ORDEN},
             text="Cantidad")
fig.update_layout(height=420, template="plotly_white", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# --- Visualización 2: una nube de palabras por categoría -------------------
st.subheader("Nubes de palabras por categoría")

columnas = st.columns(3)
for columna, categoria in zip(columnas, ORDEN):
    textos = df[df["Sentimiento"] == categoria]["Review"].dropna()
    texto_unido = " ".join(textos.astype(str))

    columna.markdown(f"**{categoria}** ({len(textos)} reseñas)")

    if texto_unido.strip():
        nube = WordCloud(width=700, height=450, background_color="white",
                         colormap=MAPAS[categoria], stopwords=IGNORAR,
                         max_words=60, collocations=False).generate(texto_unido)
        figura, eje = plt.subplots(figsize=(7, 4.5))
        eje.imshow(nube)
        eje.axis("off")
        columna.pyplot(figura)
    else:
        columna.info(f"Sin reseñas en la categoría {categoria}.")

# --- Datos externos desde una API ------------------------------------------
st.subheader("Temperatura diaria en Villa Alemana (API Open-Meteo)")

try:
    clima = obtener_clima()
    fig_clima = px.line(clima, x="Fecha", y=["T_Max", "T_Min"], markers=True,
                        labels={"value": "°C", "variable": "Serie"})
    fig_clima.update_layout(height=380, template="plotly_white")
    st.plotly_chart(fig_clima, use_container_width=True)
except requests.exceptions.RequestException:
    st.warning("No se pudo consultar la API en este momento.")
