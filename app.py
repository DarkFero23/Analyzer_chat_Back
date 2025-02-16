import os
import io
import psycopg2
import pandas as pd
from flask import Flask, request, jsonify, send_file, session
import json
import matplotlib
matplotlib.use('Agg')
import nltk
from dotenv import load_dotenv
import jwt
from datetime import timedelta
from nltk.sentiment import SentimentIntensityAnalyzer
import secrets
from flask_session import Session
from sqlalchemy import create_engine, text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter , defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from flask_cors import CORS
import numpy as np
import re
from limpiar_datos import DataFrame_Data
import emoji
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import string
from collections import Counter
from textblob import TextBlob
from nrclex import NRCLex
import uuid
from nltk.corpus import stopwords
import zipfile
import matplotlib.font_manager as fm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import requests
import csv
import unicodedata
from limpiar_datos import IsAuthor, Date_Chat, DataPoint
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))  # Lista de palabras irrelevantes en español
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
palabras_neutrales = set([
    "mano", "mas", "ahi", "asi", "vas", "puedo", "aun", "voy", 
    "hago", "ver", "opcion", "gente", "casa", "wtf", "lol", "hahaha", "god", "dream"
]
)
# Verificar y descargar el lexicón si no está disponible
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Inicializar el analizador de sentimientos
sia = SentimentIntensityAnalyzer()
load_dotenv()

# 🔹 Inicializar Flask
app = Flask(__name__)

CORS(app, supports_credentials=True)
archivos_por_usuario = {}  # Diccionario para almacenar archivos temporalmente

#####CONEXION A LA BD########
DATABASE_URL = os.environ.get("DATABASE_URL")
#DATABASE_URL = "dbname=chat_analyzers user=postgres password=SylasMidM7 host=localhost port=5433"
if not DATABASE_URL:
    print("⚠️ No se encontró DATABASE_URL, usando base de datos local.")
else:
    
    print(f"🔹 Usando base de datos en Render: {DATABASE_URL}")
# 🔹 Conectar a PostgreSQL (local o en Render)

def normalize_text(text):
    return unicodedata.normalize("NFC", text)  # NFC mantiene los emojis compuestos

def conectar_bd():
    try:
        sslmode = "require" if "render.com" in DATABASE_URL else None
        conn = psycopg2.connect(DATABASE_URL, sslmode=sslmode)
        print("✅ Conexión exitosa a PostgreSQL")
        return conn
    except Exception as e:
        print(f"❌ Error al conectar con PostgreSQL: {e}")
        return None
####FIN DE CONEXION A LA BD########
def obtener_datos(user_token):
    conn = conectar_bd()  # Conectamos a la BD
    if conn is None:
        return None

    try:
        query = """
        SELECT nombre_archivo, fecha, dia_semana, num_dia, mes, num_mes, anio, hora, autor, mensaje, formato, user_token
        FROM archivos_limpiados WHERE user_token = %s;
        """        
        df = pd.read_sql_query(query, conn, params=(user_token,))  # Cargar datos en DataFrame
        conn.close()
        return df if not df.empty else None  # Retornar DataFrame si hay datos
    
    except Exception as e:
        print(f"❌ Error al obtener datos: {e}")
        return None

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No se ha subido ningún archivo"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    conn = conectar_bd()
    if not conn:
        return jsonify({"error": "No se pudo conectar a la base de datos"}), 500

    try:
        cursor = conn.cursor()

        cursor.execute("SELECT COALESCE(MAX(user_token::INTEGER), 0) + 1 FROM archivos_chat;")
        user_token = cursor.fetchone()[0]

        nombre_archivo = file.filename
        extension = nombre_archivo.split('.')[-1].lower()

        if extension == 'zip':
            with zipfile.ZipFile(io.BytesIO(file.read()), 'r') as zip_ref:
                archivos_txt = [f for f in zip_ref.namelist() if f.endswith('.txt')]

                if not archivos_txt:
                    return jsonify({
                        "error": "El archivo .zip no contiene archivos .txt",
                        "archivos_encontrados": zip_ref.namelist()
                    }), 400

                if len(archivos_txt) > 1:
                    return jsonify({
                        "error": "El archivo .zip debe contener solo un archivo .txt",
                        "archivos_encontrados": archivos_txt
                    }), 400

                archivo_txt = archivos_txt[0]
                print(f"📂 Archivo encontrado en ZIP: {archivo_txt}")

                with zip_ref.open(archivo_txt) as f:
                    contenido = f.read().decode("utf-8", errors="replace")
                    print("📂 Contenido leído del archivo:")
                    print(contenido[:500])  # Muestra los primeros 500 caracteres
                
                    cursor.execute("""
                        INSERT INTO archivos_chat (nombre_archivo, contenido, user_token)
                        VALUES (%s, %s, %s) RETURNING id;
                    """, (archivo_txt, contenido, user_token))
                    archivo_id = cursor.fetchone()[0]
                    conn.commit()

                    df = DataFrame_Data(contenido, archivo_txt, user_token)
                    if df.empty:
                        return jsonify({"error": f"No se pudieron procesar los mensajes de {archivo_txt}"}), 500

                    # 💡 Asegurar que user_token está presente en todas las filas
                    if "user_token" not in df.columns:
                        df["user_token"] = user_token  # Si no existe la columna, la creamos

                    df["user_token"] = df["user_token"].fillna(user_token)  # Rellenar NaN con el user_token actual
                    df["user_token"] = df["user_token"].replace("", user_token)  # Rellenar valores vacíos
                    df["user_token"] = df["user_token"].astype(str).str.strip()  # Asegurar que es string sin espacios raros

                    # Verificar si aún hay filas sin user_token (esto imprimirá si hay problemas)
                    if df["user_token"].isnull().sum() > 0 or (df["user_token"] == "").sum() > 0:
                        print("⚠ Advertencia: Algunas filas aún tienen user_token vacío")

                    csv_buffer = io.StringIO()
                    csv_writer = csv.writer(csv_buffer, delimiter="|", quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    for index, row in enumerate(df.itertuples(index=False, name=None)):
                        sanitized_row = tuple(str(value).replace('|', ' ') for value in row)
                        print(f"📌 Insertando fila {index + 1}: {sanitized_row}")

                        try:
                            csv_writer.writerow(sanitized_row)
                        except Exception as e:
                            print(f"🚨 ERROR en la fila {index + 1}: {sanitized_row}")
                            print(f"⚠️ Detalle del error: {str(e)}")
                        break  # Detener el proceso en la primera fila con error
                    csv_buffer.seek(0)
                    print(df.head())  # 💡 Para verificar que 'user_token' tiene valores antes de insertarlo

                    cursor.copy_from(csv_buffer, 'archivos_limpiados', sep="|", columns=[
                        'nombre_archivo', 'fecha', 'dia_semana', 'num_dia', 'mes', 'num_mes', 'anio',
                        'hora', 'formato', 'autor', 'mensaje', 'user_token'
                    ])
                    conn.commit()
                    print(f"✅ Archivo limpio guardado con user_token {user_token}: {archivo_txt}")

        else:
            return jsonify({"error": "Solo se aceptan archivos .zip"}), 400

        return jsonify({
            "message": f"Archivo '{archivo_txt}' listo para analizar.",
            "user_token": user_token
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error al procesar el archivo: {str(e)}"}), 500

    finally:
        cursor.close()
        conn.close()

@app.route('/get_statistics', methods=['GET'])
def get_statistics():
    user_token = request.args.get("user_token")  # 🔥 Recibe el token dinámicamente

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    # 🔹 Convertir user_token a entero (importante para evitar errores)
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # 🔥 Aquí llamamos a la BD para obtener datos filtrados por el user_token
    df = obtener_datos(user_token)  # Función que trae los datos de la BD

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # 🔥 Convertir valores para evitar errores de serialización
    total_message = int(df.shape[0])
    media_message = int(df[df['mensaje'] == '<Multimedia omitido>'].shape[0])
    del_message = int(df[df['mensaje'] == 'Eliminaste este mensaje.'].shape[0])

    media_percentage = float((media_message / total_message) * 100) if total_message > 0 else 0.0
    del_percentage = float((del_message / total_message) * 100) if total_message > 0 else 0.0
    total_characters = int(df['mensaje'].apply(len).sum())
    avg_characters = float(df['mensaje'].apply(len).mean()) if total_message > 0 else 0.0

    url_pattern = r'https?://\S+|www\.\S+'
    df['URL_count'] = df['mensaje'].apply(lambda x: len(re.findall(url_pattern, x)))
    total_links = int(df['URL_count'].sum())

    stats = {
        "total_mensajes": total_message,
        "mensajes_multimedia": media_message,
        "mensajes_eliminados": del_message,
        "porcentaje_multimedia": f"{media_percentage:.2f}%",
        "porcentaje_eliminados": f"{del_percentage:.2f}%",
        "total_caracteres": total_characters,
        "promedio_caracteres": f"{avg_characters:.2f}",
        "total_links": total_links,
    }

    return jsonify(stats), 200

@app.route('/plot.png', methods=['GET'])
def plot_png():
    
    user_token = request.args.get("user_token")  # 🔥 Recibe el token dinámicamente

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    # 🔹 Convertir user_token a entero (importante para evitar errores)
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # 🔥 Llamar a la función que trae los datos desde la base de datos con el user_token
    df = obtener_datos(user_token)  # Función que trae los datos de la BD

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # Verificar si el DataFrame contiene la columna 'Day'
    if 'dia_semana' not in df.columns:
        return jsonify({"error": "Los datos no contienen la columna 'Day'."}), 400

    # Contar los registros para cada día de la semana
    active_day = df['dia_semana'].value_counts()

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Crear el gráfico de barras
    bars = ax.bar(active_day.index, active_day.values, color='#32CD32')

    # Agregar etiquetas encima de las barras
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2 - 0.1, yval + 0.5, int(yval), color='black', fontsize=10)

    # Configurar etiquetas y título
    ax.set_xticks(range(len(active_day.index)))
    ax.set_xticklabels(active_day.index, rotation=0, fontsize=10)
    ax.set_yticks([])  # No necesitamos los ticks en el eje Y
    ax.set_title('Actividad del chat por día', fontsize=13, fontweight='bold')

    # Guardar el gráfico en un objeto BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Devolver la imagen como respuesta HTTP
    return send_file(img, mimetype='image/png')


# --- Endpoint para generar el gráfico de emojis agrupados por usuario ---
@app.route('/top_emojis', methods=['GET'])
def obtener_top_emojis():
    user_token = request.args.get("user_token")

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    df = obtener_datos(user_token)

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    no_emojis = ['🏻', '🏼', '🪄', '🪛', '🏿']
    top_emojis_por_usuario = {}

    # Iteramos sobre cada mensaje y su respectivo usuario
    for _, row in df.iterrows():
        usuario = row['autor']
        mensaje = str(row['mensaje'])

        for ch in mensaje:
            if emoji.is_emoji(ch) and ch not in no_emojis:
                if usuario not in top_emojis_por_usuario:
                    top_emojis_por_usuario[usuario] = {}
                if ch not in top_emojis_por_usuario[usuario]:
                    top_emojis_por_usuario[usuario][ch] = 0
                top_emojis_por_usuario[usuario][ch] += 1

    # Convertimos el diccionario a una lista de objetos en el formato correcto
    top_emojis = [
        {"user": usuario, "emoji": emoji, "count": count}
        for usuario, emojis in top_emojis_por_usuario.items()
        for emoji, count in sorted(emojis.items(), key=lambda x: x[1], reverse=True)[:10]
    ]

    return jsonify({"top_emojis": top_emojis})

@app.route('/plot_dates.png', methods=['GET'])
def plot_dates_png():
    user_token = request.args.get("user_token")  # Obtener el user_token de los parámetros de la URL

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    # Convertir user_token a entero (importante para evitar errores)
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos con el user_token
    df = obtener_datos(user_token)  # Llamar a la función que trae los datos

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # Contar el número de mensajes por fecha y seleccionar los 10 días con mayor actividad
    TopDate = df['fecha'].value_counts().head(10)
    print("Datos por fecha:", TopDate)

    # Crear la figura y los ejes para el gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))

    # Graficar la serie en un gráfico de barras
    bars = ax.bar(TopDate.index, TopDate.values, color='#32CD32')

    # Agregar etiquetas encima de cada barra
    for idx, value in enumerate(TopDate):
        ax.text(idx - 0.15, value + 2, str(int(value)), color='black', fontsize=10)

    # Configurar las etiquetas del eje x y el título
    ax.set_xticklabels(TopDate.index, rotation=5, fontsize=8)
    ax.set_yticks([])
    ax.set_title('Los 10 Días más activos del chat', fontsize=13, fontweight='bold')

    # Guardar el gráfico en un objeto BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)

    # Devolver la imagen como respuesta HTTP
    return send_file(img, mimetype='image/png')

@app.route('/plot_mensajes_año.png', methods=['GET'])
def plot_mensajes_año():
    user_token = request.args.get("user_token")  # Obtener el user_token de los parámetros de la URL

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    # Convertir user_token a entero (importante para evitar errores)
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos con el user_token
    df = obtener_datos(user_token)  # Llamar a la función que trae los datos

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # Verificar que la columna 'Year' existe en el DataFrame
    if 'anio' not in df.columns:
        return jsonify({"error": "Los datos no contienen la columna 'anio'."}), 400

    # Contar el número de mensajes por año y ordenarlos (opcionalmente en orden ascendente)
    TopYear = df['anio'].value_counts().sort_index()  # sort_index() ordena por año
    print("Datos por año:", TopYear)

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Crear el gráfico de barras
    bars = ax.bar(TopYear.index.astype(str), TopYear.values, color='#32CD32')

    # Agregar etiquetas encima de cada barra
    for idx, value in enumerate(TopYear.values):
        ax.text(idx, value + 15, str(int(value)), ha='center', va='bottom', color='black', fontsize=10)

    # Configurar etiquetas del eje X y el título
    ax.set_xticklabels(TopYear.index.astype(str), rotation=0, fontsize=10)
    ax.set_yticks([])  # Ocultar marcas del eje Y
    ax.set_title('Mensajes por Año', fontsize=13, fontweight='bold')
    fig.tight_layout()

    # Guardar el gráfico en un objeto BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)

    # Devolver la imagen como respuesta HTTP
    return send_file(img, mimetype='image/png')

@app.route('/plot_mensajes_mes.png', methods=['GET'])
def plot_mensajes_mes():
    user_token = request.args.get("user_token")  # Obtener el user_token de los parámetros de la URL

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    # Convertir user_token a entero (importante para evitar errores)
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos con el user_token
    df = obtener_datos(user_token)  # Llamar a la función que trae los datos

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # Verificar que la columna 'Month' existe en el DataFrame
    if 'mes' not in df.columns:
        return jsonify({"error": "Los datos no contienen la columna 'mes'."}), 400

    # Contar el número de mensajes por mes y ordenarlos (opcionalmente en orden ascendente)
    TopMonth = df['mes'].value_counts().sort_index()  # sort_index() ordena por mes
    print("Conteo por mes:", TopMonth)

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(TopMonth.index, TopMonth.values, color='#32CD32')

    # Agregar etiquetas encima de cada barra
    for a, b in enumerate(TopMonth.values):
        ax.text(a - 0.12, b + 15, str(int(b)), ha='center', color='black', fontsize=10)

    # Configurar las etiquetas del eje X y el título
    ax.set_xticklabels(TopMonth.index, rotation=0, fontsize=10)
    ax.set_yticks([])  # Ocultar marcas del eje Y
    ax.set_title('Mensajes por Mes', fontsize=13, fontweight='bold')
    fig.tight_layout()

    # Guardar el gráfico en un objeto BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)

    # Devolver la imagen como respuesta HTTP
    return send_file(img, mimetype='image/png')

@app.route('/plot_horas_completo.png', methods=['GET'])
def plot_horas_completo_png():
    user_token = request.args.get("user_token")  # Obtener el user_token de los parámetros de la URL

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    # Convertir user_token a entero
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos con el user_token
    df = obtener_datos(user_token)  # Llamar a la función que trae los datos

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # Verificar que las columnas necesarias existen
    if 'hora' not in df.columns or 'formato' not in df.columns:
        return jsonify({"error": "Los datos no contienen la información de tiempo y formato."}), 400

    # Extraer solo la hora (sin minutos) y convertirla en número
    df['Hora'] = df['hora'].str.split(':').str[0].astype(int)  # Obtener solo la hora en número

    # Concatenar la hora con AM o PM
    df['Hora_Formato'] = df['Hora'].astype(str) + ' ' + df['formato']

    # Contar mensajes por cada hora con AM/PM
    horas_activas = df['Hora_Formato'].value_counts().sort_index()

    print("Conteo de mensajes por hora (con AM/PM):", horas_activas)

    # Crear la figura y los ejes para el gráfico de barras
    fig, ax = plt.subplots(figsize=(12, 6))
    barras = ax.bar(horas_activas.index, horas_activas.values, color='#32CD32')

    # Agregar etiquetas encima de cada barra
    for idx, valor in enumerate(horas_activas.values):
        ax.text(idx, valor + 3, str(int(valor)), ha='center', color='black', fontsize=9)

    # Configurar etiquetas del eje X y título del gráfico
    ax.set_xticks(range(len(horas_activas.index)))
    ax.set_xticklabels(horas_activas.index, rotation=45, fontsize=10)  # Rotar para que sean legibles
    ax.set_yticks([])
    ax.set_title('Cantidad de mensajes por hora (AM/PM)', fontsize=13, fontweight='bold')
    fig.tight_layout()

    # Guardar el gráfico en un objeto BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)

    # Devolver la imagen como respuesta HTTP
    return send_file(img, mimetype='image/png')

@app.route('/plot_timeline.png', methods=['GET'])
def plot_timeline():
    user_token = request.args.get("user_token")  # Obtener el user_token de los parámetros de la URL

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    # Convertir user_token a entero
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos con el user_token
    df = obtener_datos(user_token)  # Llamar a la función que trae los datos

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404
    # Agrupar por año, número de mes y mes, contando la cantidad de mensajes
    TimeLine = df.groupby(['anio', 'num_mes', 'mes']).count()['mensaje'].reset_index()

    # Crear una nueva columna con la combinación de Mes y Año
    TimeLine['hora'] = TimeLine.apply(lambda row: f"{row['mes']}-{row['anio']}", axis=1)

    # Crear la figura del gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(TimeLine['hora'], TimeLine['mensaje'], marker='o', linestyle='-', color='#32CD32')

    # Mejoras en el gráfico
    plt.xticks(rotation=45, size=10)  # Rotar las etiquetas del eje X para mejor lectura
    plt.yticks(size=10)
    plt.title('Línea Temporal de Mensajes por Mes', fontsize=13, fontweight='bold')
    plt.xlabel('Mes - Año', fontsize=11)
    plt.ylabel('Cantidad de Mensajes', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Guardar el gráfico en un objeto BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)

    # Devolver la imagen como respuesta HTTP
    return send_file(img, mimetype='image/png')


@app.route('/plot_mensajes_por_dia.png', methods=['GET'])
def plot_mensajes_por_dia():
    user_token = request.args.get("user_token")  # Obtener el user_token de los parámetros de la URL

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    # Convertir user_token a entero
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos con el user_token
    df = obtener_datos(user_token)  # Llamar a la función que trae los datos

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # Convertir la columna de fecha a formato datetime
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')

    # Agrupar por fecha y contar los mensajes
    Daily_LineTime = df.groupby('fecha').count()['mensaje'].reset_index()

    # Ordenar los días con más mensajes
    Daily_LineTime_Sort = Daily_LineTime.sort_values(by='mensaje', ascending=False).reset_index(drop=True)

    # Crear la figura
    fig, ax = plt.subplots(figsize=(10, 6))

    # Graficar la línea temporal de mensajes por día
    ax.plot(Daily_LineTime['fecha'], Daily_LineTime['mensaje'], color='#32CD32', marker='o', linestyle='-')

    # Resaltar los 5 días con más mensajes
    colores = ['red', 'green', 'purple', 'orange', 'black']
    for i in range(min(5, len(Daily_LineTime_Sort))):
        ax.scatter(
            Daily_LineTime_Sort.fecha[i], 
            Daily_LineTime_Sort.mensaje[i], 
            color=colores[i], 
            marker='o', 
            label=f"{Daily_LineTime_Sort.fecha[i].strftime('%Y-%m-%d')} ({Daily_LineTime_Sort.mensaje[i]} msg)"
        )

    # Configuración de ejes y etiquetas
    ax.set_xticks(Daily_LineTime['fecha'][::max(1, len(Daily_LineTime) // 10)])  # Espaciar bien las fechas
    ax.set_xticklabels(Daily_LineTime['fecha'][::max(1, len(Daily_LineTime) // 10)].dt.strftime('%Y-%m-%d'), rotation=15, fontsize=9)
    ax.set_yticks([])  # Ocultar líneas del eje Y para un diseño más limpio
    ax.set_title('Línea Temporal de Mensajes por Día', fontsize=13, fontweight='bold')

    # Agregar leyenda
    ax.legend(loc='upper left', fontsize=9, frameon=True)

    # Guardar la imagen en un objeto BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
    img.seek(0)
    plt.close(fig)

    # Devolver la imagen como respuesta HTTP
    return send_file(img, mimetype='image/png')


####FUNCIONES DE LIMPIEZA DE DATOS PARA EL WORDCLOUD
# 🔹 Función para eliminar tildes
def delete_tilde(texto):
    trans = str.maketrans("áéíóúü", "aeiouu")  # Mapeo de caracteres
    return texto.translate(trans).lower()  # Convertir todo a minúsculas

# 🔹 Función para eliminar signos de puntuación
def remove_puntuation(texto):
    return texto.translate(str.maketrans("", "", string.punctuation))

# 🔹 Función para eliminar palabras irrelevantes y patrones específicos
def regex_word(texto):
        word = ['\s\<(M|m)ultimedia\somitido\>', '\somitido\s', '\smultimedia\s','https?\S*',
        '(\<Multimedia\s)', '\w+\.vcf','\(archivo\sadjunto\)',
        'omitido\>', '\s{4}', '\s{3}', '\s{2}', '\s[a-zA-Z]\s',
        '\svcf', '\s(p|P)\s(m|M)\s', '\s(p|P)(m|M)\s', '\sp\s',
        '\sm\s', '\sde\s', '\scon\s', '\sque\s', '\sla\s',
        '\slo\s', '\spara\s', '\ses\s', '\sdel\s', '\spor\s',
        '\sel\s', '\sen\s', '\slos\s', '\stu\s', '\ste\s','\sya\s','\smi\s'
        '[\w\._]{5,30}\+?[\w]{0,10}@[\w\.\-]{3,}\.\w{2,5}',
        '\sun\s', '\sus\s', 'su\s', '\s\u200e', '\u200e' '\s\s',
        '\s\s\s', '\s\u200e3', '\s\u200e2', '\s\.\.\.\s', '/',
        '\s\u200e4', '\s\u200e7', '\s\u200e8', '\suna\s',
        'la\s', '\slas\s', '\sse\s', '\sal\s','\sle\s',
        '\sbuenas\s', '\sbuenos\s', '\sdias\s', '\stardes\s', '\snoches\s',
        '\sesta\s', '\spero\s','\sdia\s', '\sbuenas\s', '\spuede\s', '\spueden\s',
        '\sson\s', '\shay\s', '\seste\s', '\scomo\s', '\salgun\s', '\salguien\s',
        '\stodo\s', '\stodos\s', '\snos\s', '\squien\s', '\seso\s', '\sdesde\s',
        '\sarchivo\sadjunto\s', 'gmailcom', '\sdonde\s', '\shernan\s', '\slavadoras\s',
        'gracias', '\selimino\smensaje\s', '\snnnn\s',
        '\sllll\s', '\slll/\s', 'llll']

        regexes = [re.compile(p) for p in word]

        for regex in regexes:
                patron = re.compile(regex)
                texto = patron.sub(' ', texto)
        return texto

# 🔹 Función para eliminar emojis
def delete_emoji(texto):
    return emoji.replace_emoji(texto, replace='')


######FIN DE FUNCIONES DE LIMPIEZA DE DATOS PARA EL WORDCLOUD
@app.route('/plot_nube_palabras.png', methods=['GET'])
def plot_nube_palabras():
    user_token = request.args.get("user_token")  # Obtener el user_token de los parámetros de la URL

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    # Convertir user_token a entero
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos con el user_token
    df = obtener_datos(user_token)  # Llamar a la función que trae los datos

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # Obtener la fecha desde los parámetros de la URL
    fecha_str = request.args.get('fecha')  # Formato esperado: YYYY-MM-DD

    # Convertir la columna de fecha a formato datetime
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')

    # Validar el rango de fechas disponibles
    fecha_min = df['fecha'].min().strftime('%Y-%m-%d')
    fecha_max = df['fecha'].max().strftime('%Y-%m-%d')

    if fecha_str:
        try:
            fecha_seleccionada = pd.to_datetime(fecha_str, format='%Y-%m-%d')
        except ValueError:
            return jsonify({"error": f"Formato de fecha incorrecto. Usa YYYY-MM-DD. Rango válido: {fecha_min} a {fecha_max}"}), 400

        # Verificar si la fecha está dentro del rango de datos
        if fecha_seleccionada < df['fecha'].min() or fecha_seleccionada > df['fecha'].max():
            return jsonify({"error": f"La fecha seleccionada está fuera del rango disponible ({fecha_min} - {fecha_max})."}), 400
    else:
        # Si no se especifica una fecha, seleccionar el día con más mensajes
        Daily_LineTime = df.groupby('fecha').count()['mensaje'].reset_index()
        fecha_seleccionada = Daily_LineTime.sort_values(by='mensaje', ascending=False).iloc[0]['fecha']

    # Filtrar mensajes de la fecha seleccionada
    df_fecha = df[(df['fecha'] == fecha_seleccionada) & (df['mensaje'] != '<Multimedia omitido>')]

    if df_fecha.empty:
        return jsonify({"error": f"No hay mensajes disponibles para la fecha {fecha_seleccionada.strftime('%Y-%m-%d')}."}), 404

    # Unir los mensajes en un solo texto
    text = ' '.join(df_fecha['mensaje'])

    # Aplicar funciones de limpieza
    text = delete_emoji(text)
    text = delete_tilde(text)
    text = remove_puntuation(text)
    text = regex_word(text)

    # Generar la nube de palabras
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', colormap='viridis').generate(text)

    # Crear la figura para la nube de palabras
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f"Nube de Palabras - {fecha_seleccionada.strftime('%Y-%m-%d')}", fontsize=14, fontweight='bold')

    # Guardar el gráfico en un objeto BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
    img.seek(0)
    plt.close(fig)

    # Devolver la imagen como respuesta HTTP
    return send_file(img, mimetype='image/png')

####Aca me quedo , falta lo del front pero esto ya funci    ona####
@app.route('/analisis_sentimientos', methods=['GET'])
def analisis_sentimientos():
    user_token = request.args.get("user_token")  # Obtener el user_token de la URL

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    # Convertir user_token a entero
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos
    df = obtener_datos(user_token)  # Función que consulta la BD

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # Aplicar análisis de sentimiento
    def obtener_sentimiento(texto):
        if not isinstance(texto, str) or texto == "<Multimedia omitido>":
            return "neutro"  # Ignorar multimedia

        puntaje = sia.polarity_scores(texto)
        if puntaje['compound'] >= 0.05:
            return "positivo"
        elif puntaje['compound'] <= -0.05:
            return "negativo"
        else:
            return "neutro"

    df['Sentimiento'] = df['mensaje'].apply(obtener_sentimiento)

    # Contar los tipos de sentimientos en porcentaje
    conteo_sentimientos = df['Sentimiento'].value_counts(normalize=True) * 100

    # Devolver los porcentajes
    resultado = {
        "positivo": f"{conteo_sentimientos.get('positivo', 0):.2f}%",
        "neutro": f"{conteo_sentimientos.get('neutro', 0):.2f}%",
        "negativo": f"{conteo_sentimientos.get('negativo', 0):.2f}%"
    }

    return jsonify(resultado), 200


@app.route('/mensajes_mayor_emocion', methods=['GET'])
def mensajes_mayor_emocion():
    user_token = request.args.get("user_token")  # Obtener el user_token de la URL

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    # Convertir user_token a entero
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos
    df = obtener_datos(user_token)  # Función que consulta la BD

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # Aplicar análisis de sentimiento
    def obtener_puntaje(texto):
        if not isinstance(texto, str) or texto == "<Multimedia omitido>":
            return 0  # Ignorar multimedia

        return sia.polarity_scores(texto)['compound']

    df['Puntaje_Sentimiento'] = df['mensaje'].apply(obtener_puntaje)

    # Obtener los 5 mensajes con mayor y menor carga emocional
    top_positivos = df.nlargest(5, 'Puntaje_Sentimiento')[['fecha', 'autor', 'mensaje', 'Puntaje_Sentimiento']]
    top_negativos = df.nsmallest(5, 'Puntaje_Sentimiento')[['fecha', 'autor', 'mensaje', 'Puntaje_Sentimiento']]

    # Convertir a JSON
    resultado = {
        "mensajes_mas_positivos": top_positivos.to_dict(orient="records"),
        "mensajes_mas_negativos": top_negativos.to_dict(orient="records")
    }

    return jsonify(resultado), 200


@app.route('/sentimientos_por_dia.png', methods=['GET'])
def sentimientos_por_dia():
    user_token = request.args.get("user_token")  # Obtener el user_token de la URL

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    try:
        user_token = int(user_token)  # Convertir user_token a entero
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos
    df = obtener_datos(user_token)  # Función que consulta la BD

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # Asegurar que la fecha está en formato datetime
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')

    # Aplicar análisis de sentimiento
    def obtener_sentimiento(texto):
        if not isinstance(texto, str) or texto == "<Multimedia omitido>":
            return "neutro"

        puntaje = sia.polarity_scores(texto)
        if puntaje['compound'] >= 0.05:
            return "positivo"
        elif puntaje['compound'] <= -0.05:
            return "negativo"
        else:
            return "neutro"

    df['Sentimiento'] = df['mensaje'].apply(obtener_sentimiento)

    # Contar los sentimientos por día
    tendencia = df.groupby(['fecha', 'Sentimiento']).size().unstack(fill_value=0)

    # Graficar la tendencia de sentimientos
    fig, ax = plt.subplots(figsize=(8, 6))
    tendencia.plot(kind='line', ax=ax, marker='o', color=['green', 'gray', 'red'])

    # Etiquetas y título
    ax.set_title("Evolución de Sentimientos en el Chat", fontsize=13, fontweight='bold')
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Cantidad de Mensajes", fontsize=11)
    ax.legend(["Positivo", "Neutro", "Negativo"])
    plt.xticks(rotation=45, fontsize=9)
    plt.grid()

    # Guardar la imagen en un objeto BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)

    return send_file(img, mimetype='image/png')


@app.route('/sentimiento_promedio_dia.png', methods=['GET'])
def sentimiento_promedio_dia():
    user_token = request.args.get("user_token")  # Obtener user_token de la URL

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    # Convertir user_token a entero
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos
    df = obtener_datos(user_token)  # Función que consulta la BD

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # Convertir la columna de fecha a formato datetime
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')

    # Aplicar análisis de sentimiento
    df['Puntaje_Sentimiento'] = df['mensaje'].apply(
        lambda x: sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0
    )

    # Agrupar por día y calcular el puntaje promedio
    sentimiento_por_dia = df.groupby('fecha')['Puntaje_Sentimiento'].mean()

    # Seleccionar los 15 días con mayor carga emocional (positivo o negativo)
    top_dias = sentimiento_por_dia.abs().nlargest(15).index  # Obtener las fechas con mayor magnitud de sentimiento
    sentimiento_filtrado = sentimiento_por_dia.loc[top_dias].sort_index()

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = sentimiento_filtrado.plot(
        kind='bar',
        color=['green' if v > 0 else 'red' for v in sentimiento_filtrado],
        ax=ax
    )

    # Agregar etiquetas con los puntajes sobre las barras
    for bar in bars.patches:
        ax.annotate(
            f'{bar.get_height():.2f}', 
            (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='black'
        )

    ax.set_title("Top 15 días con mayor carga emocional", fontsize=14, fontweight='bold')
    ax.set_xlabel("Fecha", fontsize=12)
    ax.set_ylabel("Puntaje de Sentimiento", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Guardar imagen en BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)

    return send_file(img, mimetype='image/png')


@app.route('/top_palabras_usuario', methods=['GET'])
def top_palabras_usuario():
    user_token = request.args.get("user_token")  # Obtener user_token de la URL

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    # Convertir user_token a entero
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos
    df = obtener_datos(user_token)  # Función que consulta la BD

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    if 'autor' not in df.columns or 'mensaje' not in df.columns:
        return jsonify({"error": "Faltan columnas requeridas (autor, mensaje)"}), 400

    # **FILTRAR MENSAJES QUE CONTENGAN EXACTAMENTE "multimedia omitido"**
    df = df[~df['mensaje'].str.lower().str.contains(r'\bmultimedia omitido\b', na=False, regex=True)]

    # Diccionario para almacenar el top de palabras por usuario
    top_palabras_por_usuario = {}

    # Agrupar por usuario y calcular el top de palabras
    for usuario, mensajes in df.groupby('autor')['mensaje']:
        texto_total = ' '.join(mensajes.dropna()).lower()  # Concatenar mensajes del usuario

        # Limpiar texto y contar palabras
        texto_total = remove_puntuation(delete_tilde(texto_total))  # Quitar tildes y puntuación
        palabras = texto_total.split()
        top_palabras = Counter(palabras).most_common(10)  # Top 10 palabras más usadas

        # Guardar en el diccionario con el usuario como clave
        top_palabras_por_usuario[usuario] = [
            {"palabra": palabra, "cantidad": cantidad} for palabra, cantidad in top_palabras
        ]  # Convertir lista de tuplas en lista de diccionarios

    return jsonify({"top_palabras_por_usuario": top_palabras_por_usuario}), 200



@app.route('/grafico_emociones.png', methods=['GET'])
def grafico_emociones():
    user_token = request.args.get("user_token")  # Obtener user_token de la URL

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    # Convertir user_token a entero
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos
    df = obtener_datos(user_token)  # Función que consulta la BD

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # Aplicar análisis de emociones
    def detectar_emociones(texto):
        if not isinstance(texto, str):
            return {}

        emociones = NRCLex(texto).raw_emotion_scores
        return emociones

    df['Emociones'] = df['mensaje'].apply(detectar_emociones)

    # Contar las emociones totales
    emociones_totales = Counter()
    for emociones in df['Emociones']:
        emociones_totales.update(emociones)

    traduccion_emociones = {
        'joy': 'feli',
        'sadness': 'tite',
        'anger': 'nojado',
        'fear': 'Miedo'
    }
    # Seleccionar las principales emociones a graficar
    emociones_relevantes = ['joy', 'sadness', 'anger', 'fear']
    datos_emociones = {traduccion_emociones[emocion]: emociones_totales.get(emocion, 0) for emocion in emociones_relevantes}

    # Graficar
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(datos_emociones.keys(), datos_emociones.values(), color=['yellow', 'blue', 'red', 'purple'])

    ax.set_title("Distribución de Emociones en el Chat", fontsize=13, fontweight='bold')
    ax.set_xlabel("Emoción", fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    plt.grid()

    # Guardar imagen en BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)

    return send_file(img, mimetype='image/png')

# Diccionario de palabras tóxicas
palabras_toxicas = {
    "idiota", "estúpido", "imbécil", "tonto", "odiar", "asco", "muere", "cállate",
    "gay", "cabro", "mierda", "puta", "perra", "suicidate", "negro", "tonta",
    "muerete", "calla", "concha", "ctm", "pta", "pto", "homosexual" ,"mrda" ,"estupida" , "Lol", "lol", "valo" "ctmr"  
    
}

@app.route('/conteo_toxicidad', methods=['GET'])
def conteo_toxicidad():
    user_token = request.args.get("user_token")

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    df = obtener_datos(user_token)

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # Diccionario para contar palabras tóxicas por usuario
    conteo_por_usuario = defaultdict(Counter)

    # Evaluar toxicidad por usuario
    for _, row in df.iterrows():
        autor = row['autor']
        mensaje = row['mensaje']

        if isinstance(mensaje, str) and mensaje != "<Multimedia omitido>":
            palabras_mensaje = mensaje.lower().split()  # Convertir en lista de palabras
            palabras_toxicas_encontradas = [p for p in palabras_mensaje if p in palabras_toxicas]

            # Actualizar el conteo del usuario
            conteo_por_usuario[autor].update(palabras_toxicas_encontradas)

    # Convertir el resultado en formato JSON
    resultado = {
        usuario: dict(conteo) for usuario, conteo in conteo_por_usuario.items()
    }

    return jsonify({"conteo_toxicidad": resultado}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)

