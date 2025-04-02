import os
import io
import psycopg2
import pandas as pd
from flask import Flask, request, jsonify, send_file, session
import json
import matplotlib

matplotlib.use("Agg")
import nltk
from dotenv import load_dotenv
from datetime import timedelta
from nltk.sentiment import SentimentIntensityAnalyzer
from flask_session import Session
from sqlalchemy import create_engine, text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter, defaultdict
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

nltk.download("stopwords")
stop_words = set(
    stopwords.words("spanish")
)  # Lista de palabras irrelevantes en español
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
palabras_neutrales = set(
    [
        "mano",
        "mas",
        "ahi",
        "asi",
        "vas",
        "puedo",
        "aun",
        "voy",
        "hago",
        "ver",
        "opcion",
        "gente",
        "casa",
        "wtf",
        "lol",
        "hahaha",
        "god",
        "dream",
    ]
)
# Verificar y descargar el lexicón si no está disponible
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# Inicializar el analizador de sentimientos
sia = SentimentIntensityAnalyzer()
load_dotenv()

# 🔹 Inicializar Flask
app = Flask(__name__)

CORS(app, supports_credentials=True)
archivos_por_usuario = {}  # Diccionario para almacenar archivos temporalmente

#####CONEXION A LA BD########
DATABASE_URL = os.environ.get("DATABASE_URL")
# DATABASE_URL = "dbname=chat_analyzers user=postgres password=SylasMidM7 host=localhost port=5433"
if not DATABASE_URL:
    print("⚠️ No se encontró DATABASE_URL, usando base de datos local.")
else:

    print(f"🔹 Usando base de datos en Render: {DATABASE_URL}")
# 🔹 Conectar a PostgreSQL (local o en Render)

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
        df = pd.read_sql_query(
            query, conn, params=(user_token,)
        )  # Cargar datos en DataFrame

        conn.close()

        return df if not df.empty else None  # Retornar DataFrame si hay datos

    except Exception as e:
        print(f"❌ Error al obtener datos: {e}")
        return None


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No se ha subido ningún archivo"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    conn = conectar_bd()
    if not conn:
        return jsonify({"error": "No se pudo conectar a la base de datos"}), 500

    try:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COALESCE(MAX(user_token::INTEGER), 0) + 1 FROM archivos_chat;"
        )
        user_token = cursor.fetchone()[0]

        nombre_archivo = file.filename
        extension = nombre_archivo.split(".")[-1].lower()

        if extension == "zip":
            with zipfile.ZipFile(io.BytesIO(file.read()), "r") as zip_ref:
                archivos_txt = [f for f in zip_ref.namelist() if f.endswith(".txt")]

                if not archivos_txt:
                    return (
                        jsonify(
                            {
                                "error": "El archivo .zip no contiene archivos .txt",
                                "archivos_encontrados": zip_ref.namelist(),
                            }
                        ),
                        400,
                    )

                if len(archivos_txt) > 1:
                    return (
                        jsonify(
                            {
                                "error": "El archivo .zip debe contener solo un archivo .txt",
                                "archivos_encontrados": archivos_txt,
                            }
                        ),
                        400,
                    )

                archivo_txt = archivos_txt[0]
                print(f"📂 Archivo encontrado en ZIP: {archivo_txt}")

                with zip_ref.open(archivo_txt) as f:
                    contenido = f.read().decode("utf-8", errors="replace")
                    print("📂 Contenido leído del archivo:")

        elif extension == "txt":
            archivo_txt = nombre_archivo
            contenido = file.read().decode("utf-8", errors="replace")

        cursor.execute(
            """
            INSERT INTO archivos_chat (nombre_archivo, contenido, user_token)
            VALUES (%s, %s, %s) RETURNING id;
            """,
            (archivo_txt, contenido, user_token),
        )
        archivo_id = cursor.fetchone()[0]
        conn.commit()

        df = DataFrame_Data(contenido, archivo_txt, user_token)
        if df.empty:
            return (
                jsonify(
                    {"error": f"No se pudieron procesar los mensajes de {archivo_txt}"}
                ),
                500,
            )

        # 💡 Asegurar que user_token está presente en todas las filas
        if "user_token" not in df.columns:
            df["user_token"] = user_token  # Si no existe la columna, la creamos

        df["user_token"] = df["user_token"].fillna(user_token)  # Rellenar NaN
        df["user_token"] = df["user_token"].replace("", user_token)  # Rellenar vacíos
        df["user_token"] = df["user_token"].astype(str).str.strip()  # Limpiar espacios

        # Filtrar filas sin user_token
        df = df.dropna(subset=["user_token"])
        df = df[df["user_token"] != ""]

        csv_buffer = io.StringIO()
        csv_writer = csv.writer(
            csv_buffer,
            delimiter="|",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )

        for index, row in enumerate(df.itertuples(index=False, name=None)):
            sanitized_row = tuple(
                str(value).replace("|", " ").replace("\\", "")
                for value in row  # 🔹 Solución aquí
            )
            print(f"📌 Insertando fila {index + 1}: {sanitized_row}")

            try:
                csv_writer.writerow(sanitized_row)
            except Exception as e:
                print(f"🚨 ERROR en la fila {index + 1}: {sanitized_row}")
                print(f"⚠️ Detalle del error: {str(e)}")

        csv_buffer.seek(0)
        print(df.head(10))  # Ver las primeras 10 filas

        cursor.copy_from(
            csv_buffer,
            "archivos_limpiados",
            sep="|",
            columns=[
                "nombre_archivo",
                "fecha",
                "dia_semana",
                "num_dia",
                "mes",
                "num_mes",
                "anio",
                "hora",
                "formato",
                "autor",
                "mensaje",
                "user_token",
            ],
        )
        conn.commit()
        print(f"✅ Archivo limpio guardado con user_token {user_token}: {archivo_txt}")

        return (
            jsonify(
                {
                    "message": f"Archivo '{archivo_txt}' listo para analizar.",
                    "user_token": user_token,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": f"Error al procesar el archivo: {str(e)}"}), 500

    finally:
        cursor.close()
        conn.close()


##YA ESTA


@app.route("/get_statistics", methods=["GET"])
def get_statistics():
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

    # 🔥 Cálculo de estadísticas generales
    total_message = df.shape[0]
    media_message = df[df["mensaje"] == "<Multimedia omitido>"].shape[0]
    del_message = df[df["mensaje"] == "Eliminaste este mensaje."].shape[0]
    total_characters = df["mensaje"].apply(len).sum()

    media_percentage = (
        (media_message / total_message * 100) if total_message > 0 else 0.0
    )
    del_percentage = (del_message / total_message * 100) if total_message > 0 else 0.0
    avg_characters = (total_characters / total_message) if total_message > 0 else 0.0

    url_pattern = r"https?://\S+|www\.\S+"
    df["URL_count"] = df["mensaje"].apply(lambda x: len(re.findall(url_pattern, x)))
    total_links = df["URL_count"].sum()

    # 🔥 Agrupación por autor
    stats_by_author = (
        df.groupby("autor")
        .agg(
            total_mensajes=("mensaje", "count"),
            mensajes_multimedia=(
                "mensaje",
                lambda x: (x == "<Multimedia omitido>").sum(),
            ),
            mensajes_eliminados=(
                "mensaje",
                lambda x: (x == "Eliminaste este mensaje.").sum(),
            ),
            total_caracteres=("mensaje", lambda x: x.apply(len).sum()),
            total_links=("URL_count", "sum"),
        )
        .reset_index()
    )

    stats_by_author["porcentaje_multimedia"] = (
        (
            stats_by_author["mensajes_multimedia"]
            / stats_by_author["total_mensajes"]
            * 100
        )
        .fillna(0)
        .round(2)
    )

    stats_by_author["porcentaje_eliminados"] = (
        (
            stats_by_author["mensajes_eliminados"]
            / stats_by_author["total_mensajes"]
            * 100
        )
        .fillna(0)
        .round(2)
    )

    stats_by_author["promedio_caracteres"] = (
        (stats_by_author["total_caracteres"] / stats_by_author["total_mensajes"])
        .fillna(0)
        .round(2)
    )

    # Convertir DataFrame a lista de diccionarios
    author_stats_list = stats_by_author.to_dict(orient="records")

    # 🔥 Respuesta final
    response = {
        "resumen_general": {
            "total_mensajes": int(total_message),
            "mensajes_multimedia": int(media_message),
            "mensajes_eliminados": int(del_message),
            "porcentaje_multimedia": f"{media_percentage:.2f}%",
            "porcentaje_eliminados": f"{del_percentage:.2f}%",
            "total_caracteres": int(total_characters),
            "promedio_caracteres": f"{avg_characters:.2f}",
            "total_links": int(total_links),
        },
        "estadisticas_por_autor": author_stats_list,
    }

    return jsonify(response), 200


##YA ESTA


@app.route("/plot.json", methods=["GET"])
def plot_png():
    user_token = request.args.get("user_token")  # 🔥 Recibe el token dinámicamente

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    try:
        user_token = int(user_token)  # 🔹 Convertir user_token a entero
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    df = obtener_datos(user_token)  # 🔥 Traer los datos desde la BD

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    if "dia_semana" not in df.columns:
        return (
            jsonify({"error": "Los datos no contienen la columna 'dia_semana'."}),
            400,
        )

    # Contar los registros por día de la semana
    active_day = df["dia_semana"].value_counts().to_dict()

    # 🔹 Estructurar los datos en JSON
    response_data = {
        "user_token": user_token,
        "activity_per_day": [
            {"day": day, "count": count} for day, count in active_day.items()
        ],
    }

    return jsonify(response_data), 200


##YA ESTA


# --- Endpoint para generar el gráfico de emojis agrupados por usuario ---
@app.route("/top_emojis", methods=["GET"])
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

    no_emojis = ["🏻", "🏼", "🪄", "🪛", "🏿"]
    top_emojis_por_usuario = {}

    # Iteramos sobre cada mensaje y su respectivo usuario
    for _, row in df.iterrows():
        usuario = row["autor"]
        mensaje = str(row["mensaje"])

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
        for emoji, count in sorted(emojis.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]
    ]

    return jsonify({"top_emojis": top_emojis})


##YA ESTA


@app.route("/plot_dates.json", methods=["GET"])
def plot_dates_json():
    user_token = request.args.get("user_token")

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    # Convertir user_token a entero
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos con el user_token
    df = obtener_datos(user_token)

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # Contar el número de mensajes por fecha y seleccionar los 10 días con mayor actividad
    top_dates = df["fecha"].value_counts().head(10)

    # Convertir a JSON
    response_data = [
        {"day": date, "count": int(count)} for date, count in top_dates.items()
    ]

    return jsonify({"top_active_days": response_data})


##YA ESTA


@app.route("/plot_mensajes_año.json", methods=["GET"])
def plot_mensajes_año_json():
    user_token = request.args.get("user_token")

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos con el user_token
    df = obtener_datos(user_token)

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    if "anio" not in df.columns:
        return jsonify({"error": "Los datos no contienen la columna 'anio'."}), 400

    # Contar el número de mensajes por año y ordenarlos
    TopYear = df["anio"].value_counts().sort_index()
    print("Datos por año:", TopYear)

    # Convertir los datos a un formato JSON legible
    data = [{"year": str(year), "count": int(count)} for year, count in TopYear.items()]

    return jsonify({"mensajes_por_año": data})


##YA ESTA


@app.route("/plot_mensajes_mes.json", methods=["GET"])
def plot_mensajes_mes():
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

    if "mes" not in df.columns:
        return jsonify({"error": "Los datos no contienen la columna 'mes'."}), 400

    # Lista de meses ordenados correctamente
    meses_ordenados = [
        "Ene",
        "Feb",
        "Mar",
        "Abr",
        "May",
        "Jun",
        "Jul",
        "Ago",
        "Sep",
        "Oct",
        "Nov",
        "Dic",
    ]

    # Contar mensajes por mes
    mensajes_por_mes = df["mes"].value_counts()

    # Ordenar los datos según la lista de meses
    mensajes_por_mes = mensajes_por_mes.reindex(meses_ordenados, fill_value=0)

    # Convertir a JSON
    data = [
        {"mes": mes, "mensajes": int(count)} for mes, count in mensajes_por_mes.items()
    ]

    return jsonify({"mensajes_por_mes": data})


##YA ESTA


@app.route("/horas_completo.json", methods=["GET"])
def horas_completo_json():
    user_token = request.args.get("user_token")

    if not user_token:
        return jsonify({"error": "Falta el user_token"}), 400

    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos con el user_token
    df = obtener_datos(user_token)

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    # Verificar que las columnas necesarias existen
    if "hora" not in df.columns or "formato" not in df.columns:
        return (
            jsonify(
                {"error": "Los datos no contienen la información de tiempo y formato."}
            ),
            400,
        )

    # Extraer solo la hora (sin minutos) y convertirla en número
    df["Hora"] = df["hora"].str.split(":").str[0].astype(int)

    # Concatenar la hora con AM o PM
    df["Hora_Formato"] = df["Hora"].astype(str) + " " + df["formato"]

    # Contar mensajes por cada hora con AM/PM
    horas_activas = df["Hora_Formato"].value_counts().sort_index()

    # Convertir a JSON
    json_resultado = [
        {"hora": hora, "cantidad": int(cantidad)}
        for hora, cantidad in horas_activas.items()
    ]

    return jsonify({"datos_horas": json_resultado})


##YA ESTA


@app.route("/plot_timeline.json", methods=["GET"])
def plot_timeline_json():
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

    # Agrupar por año, número de mes y mes, contando la cantidad de mensajes
    TimeLine = df.groupby(["anio", "num_mes", "mes"]).count()["mensaje"].reset_index()

    # Crear la lista de datos en formato JSON
    timeline_data = [
        {"hora": f"{row['mes']}-{row['anio']}", "mensajes": int(row["mensaje"])}
        for _, row in TimeLine.iterrows()
    ]

    return jsonify({"timeline": timeline_data})


##YA ESTA


@app.route("/plot_mensajes_por_dia.json", methods=["GET"])
def plot_mensajes_por_dia():
    user_token = request.args.get("user_token")

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
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y")

    # Agrupar por fecha y contar los mensajes
    Daily_LineTime = df.groupby("fecha").count()["mensaje"].reset_index()

    # Ordenar los días con más mensajes
    Daily_LineTime_Sort = Daily_LineTime.sort_values(
        by="mensaje", ascending=False
    ).reset_index(drop=True)

    # Ordenar los días con menos mensajes
    Daily_LineTime_Sort_Min = Daily_LineTime.sort_values(
        by="mensaje", ascending=True
    ).reset_index(drop=True)

    # Convertir todo a tipos estándar
    def convertir_fila(row):
        return {
            "fecha": str(row["fecha"].date()),  # Convertir Timestamp a string
            "mensajes": int(row["mensaje"]),  # Convertir int64 a int
        }

    timeline_data = [convertir_fila(row) for _, row in Daily_LineTime.iterrows()]
    top_days = [
        convertir_fila(row) for _, row in Daily_LineTime_Sort.head(5).iterrows()
    ]
    bottom_days = [
        convertir_fila(row) for _, row in Daily_LineTime_Sort_Min.head(5).iterrows()
    ]

    return jsonify(
        {"timelineDay": timeline_data, "top_days": top_days, "bottom_days": bottom_days}
    )


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
    word = [
        "\s\<(M|m)ultimedia\somitido\>",
        "\somitido\s",
        "\smultimedia\s",
        "https?\S*",
        "(\<Multimedia\s)",
        "\w+\.vcf",
        "\(archivo\sadjunto\)",
        "omitido\>",
        "\s{4}",
        "\s{3}",
        "\s{2}",
        "\s[a-zA-Z]\s",
        "\svcf",
        "\s(p|P)\s(m|M)\s",
        "\s(p|P)(m|M)\s",
        "\sp\s",
        "\sm\s",
        "\sde\s",
        "\scon\s",
        "\sque\s",
        "\sla\s",
        "\slo\s",
        "\spara\s",
        "\ses\s",
        "\sdel\s",
        "\spor\s",
        "\sel\s",
        "\sen\s",
        "\slos\s",
        "\stu\s",
        "\ste\s",
        "\sya\s",
        "\smi\s" "[\w\._]{5,30}\+?[\w]{0,10}@[\w\.\-]{3,}\.\w{2,5}",
        "\sun\s",
        "\sus\s",
        "su\s",
        "\s\u200e",
        "\u200e" "\s\s",
        "\s\s\s",
        "\s\u200e3",
        "\s\u200e2",
        "\s\.\.\.\s",
        "/",
        "\s\u200e4",
        "\s\u200e7",
        "\s\u200e8",
        "\suna\s",
        "la\s",
        "\slas\s",
        "\sse\s",
        "\sal\s",
        "\sle\s",
        "\sbuenas\s",
        "\sbuenos\s",
        "\sdias\s",
        "\stardes\s",
        "\snoches\s",
        "\sesta\s",
        "\spero\s",
        "\sdia\s",
        "\sbuenas\s",
        "\spuede\s",
        "\spueden\s",
        "\sson\s",
        "\shay\s",
        "\seste\s",
        "\scomo\s",
        "\salgun\s",
        "\salguien\s",
        "\stodo\s",
        "\stodos\s",
        "\snos\s",
        "\squien\s",
        "\seso\s",
        "\sdesde\s",
        "\sarchivo\sadjunto\s",
        "gmailcom",
        "\sdonde\s",
        "\shernan\s",
        "\slavadoras\s",
        "gracias",
        "\selimino\smensaje\s",
        "\snnnn\s",
        "\sllll\s",
        "\slll/\s",
        "llll",
    ]

    regexes = [re.compile(p) for p in word]

    for regex in regexes:
        patron = re.compile(regex)
        texto = patron.sub(" ", texto)
    return texto


# 🔹 Función para eliminar emojis
def delete_emoji(texto):
    return emoji.replace_emoji(texto, replace="")


######FIN DE FUNCIONES DE LIMPIEZA DE DATOS PARA EL WORDCLOUD


##FALTA ESTE a muerte
@app.route("/nube_palabras", methods=["GET"])
def nube_palabras():
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

    fecha_str = request.args.get("fecha")

    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y")

    fecha_min = df["fecha"].min().strftime("%Y-%m-%d")
    fecha_max = df["fecha"].max().strftime("%Y-%m-%d")

    if fecha_str:
        try:
            fecha_seleccionada = pd.to_datetime(fecha_str, format="%Y-%m-%d")
        except ValueError:
            return (
                jsonify(
                    {
                        "error": f"Formato de fecha incorrecto. Usa YYYY-MM-DD. Rango válido: {fecha_min} a {fecha_max}"
                    }
                ),
                400,
            )

        if (
            fecha_seleccionada < df["fecha"].min()
            or fecha_seleccionada > df["fecha"].max()
        ):
            return (
                jsonify(
                    {
                        "error": f"La fecha seleccionada está fuera del rango disponible ({fecha_min} - {fecha_max})."
                    }
                ),
                400,
            )
    else:
        Daily_LineTime = df.groupby("fecha").count()["mensaje"].reset_index()
        fecha_seleccionada = Daily_LineTime.sort_values(
            by="mensaje", ascending=False
        ).iloc[0]["fecha"]

    df_fecha = df[
        (df["fecha"] == fecha_seleccionada) & (df["mensaje"] != "<Multimedia omitido>")
    ]

    if df_fecha.empty:
        return (
            jsonify(
                {
                    "error": f"No hay mensajes disponibles para la fecha {fecha_seleccionada.strftime('%Y-%m-%d')}."
                }
            ),
            404,
        )

    text = " ".join(df_fecha["mensaje"])

    # Limpieza de texto
    text = delete_emoji(text)
    text = delete_tilde(text)
    text = remove_puntuation(text)
    text = regex_word(text)

    # Contar palabras más usadas
    palabras = text.split()
    conteo_palabras = Counter(palabras)

    # Devolver solo las palabras más frecuentes (las 100 más repetidas)
    palabras_frecuentes = dict(conteo_palabras.most_common(100))

    return jsonify(
        {
            "fecha": fecha_seleccionada.strftime("%Y-%m-%d"),
            "palabras": palabras_frecuentes,
        }
    )


##YA ESTA
####Aca me quedo , falta lo del front pero esto ya funci    ona####
@app.route("/analisis_sentimientos", methods=["GET"])
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

    # Aplicar análisis de sentimiento en español
    def obtener_sentimiento(texto):
        if not isinstance(texto, str) or texto == "<Multimedia omitido>":
            return "neutro"  # Ignorar multimedia

        blob = TextBlob(texto)
        polaridad = blob.sentiment.polarity  # Sentimiento entre -1 y 1

        if polaridad > 0.1:
            return "positivo"
        elif polaridad < -0.1:
            return "negativo"
        else:
            return "neutro"

    df["Sentimiento"] = df["mensaje"].apply(obtener_sentimiento)

    # Contar los tipos de sentimientos en porcentaje
    conteo_sentimientos = df["Sentimiento"].value_counts(normalize=True) * 100

    # Devolver los porcentajes
    resultado = {
        "positivo": f"{conteo_sentimientos.get('positivo', 0):.2f}%",
        "neutro": f"{conteo_sentimientos.get('neutro', 0):.2f}%",
        "negativo": f"{conteo_sentimientos.get('negativo', 0):.2f}%",
    }

    return jsonify(resultado), 200


# ESTE ESTA MAL NO SE PONE ARREGLAR
@app.route("/mensajes_mayor_emocion", methods=["GET"])
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

        return sia.polarity_scores(texto)["compound"]

    df["Puntaje_Sentimiento"] = df["mensaje"].apply(obtener_puntaje)

    # Obtener los 5 mensajes con mayor y menor carga emocional
    top_positivos = df.nlargest(5, "Puntaje_Sentimiento")[
        ["fecha", "autor", "mensaje", "Puntaje_Sentimiento"]
    ]
    top_negativos = df.nsmallest(5, "Puntaje_Sentimiento")[
        ["fecha", "autor", "mensaje", "Puntaje_Sentimiento"]
    ]

    # Convertir a JSON
    resultado = {
        "mensajes_mas_positivos": top_positivos.to_dict(orient="records"),
        "mensajes_mas_negativos": top_negativos.to_dict(orient="records"),
    }

    return jsonify(resultado), 200


# YA ESTA


@app.route("/sentimientos_por_dia.json", methods=["GET"])
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
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y")

    # Aplicar análisis de sentimiento
    def obtener_sentimiento(texto):
        if not isinstance(texto, str) or texto == "<Multimedia omitido>":
            return "neutro"

        puntaje = sia.polarity_scores(texto)
        if puntaje["compound"] >= 0.05:
            return "positivo"
        elif puntaje["compound"] <= -0.05:
            return "negativo"
        else:
            return "neutro"

    df["sentimiento"] = df["mensaje"].apply(obtener_sentimiento)

    # Contar los sentimientos por día
    tendencia = df.groupby(["fecha", "sentimiento"]).size().unstack(fill_value=0)

    # Convertir los datos en formato JSON, asegurando que los valores sean enteros
    json_resultado = []
    for fecha, row in tendencia.iterrows():
        json_resultado.append(
            {
                "fecha": fecha.strftime("%Y-%m-%d"),
                "positivo": int(row.get("positivo", 0)),  # Convertir a int
                "neutro": int(row.get("neutro", 0)),  # Convertir a int
                "negativo": int(row.get("negativo", 0)),  # Convertir a int
            }
        )

    return jsonify({"datos": json_resultado})


# YA ESTA
@app.route("/sentimiento_promedio_dia", methods=["GET"])
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
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y")

    # Aplicar análisis de sentimiento
    df["Puntaje_Sentimiento"] = df["mensaje"].apply(
        lambda x: sia.polarity_scores(x)["compound"] if isinstance(x, str) else 0
    )

    # Agrupar por día y calcular el puntaje promedio
    sentimiento_por_dia = df.groupby("fecha")["Puntaje_Sentimiento"].mean()

    # Seleccionar los 15 días con mayor carga emocional (positivo o negativo)
    top_dias = (
        sentimiento_por_dia.abs().nlargest(15).index
    )  # Obtener las fechas con mayor magnitud de sentimiento
    sentimiento_filtrado = sentimiento_por_dia.loc[top_dias].sort_index()

    # Preparar los datos para la respuesta JSON
    sentimiento_data = []
    for fecha, puntaje in sentimiento_filtrado.items():
        sentimiento_data.append(
            {
                "fecha": fecha.strftime("%Y-%m-%d"),
                "puntaje_sentimiento": puntaje,
                "sentimiento": "positivo" if puntaje > 0 else "negativo",
            }
        )

    return jsonify({"top_dias": sentimiento_data})




# YA ESTA
@app.route("/top_palabras_usuario", methods=["GET"])
def top_palabras_usuario():
    user_token = request.args.get("user_token")

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

    if "autor" not in df.columns or "mensaje" not in df.columns:
        return jsonify({"error": "Faltan columnas requeridas (autor, mensaje)"}), 400

    # **FILTRAR MENSAJES QUE CONTENGAN EXACTAMENTE "multimedia omitido"**
    df = df[
        ~df["mensaje"]
        .fillna("")
        .str.lower()
        .str.contains(r"\bmultimedia omitido\b", regex=True)
    ]

    # **Lista de palabras que queremos ignorar (stopwords)**
    stopwords_es = {
        "yo",
        "ya",
        "tú",
        "él",
        "es",
        "ella",
        "nosotros",
        "nosotras",
        "vosotros",
        "vosotras",
        "ellos",
        "ellas",
        "usted",
        "ustedes",
        "me",
        "te",
        "se",
        "nos",
        "os",
        "lo",
        "la",
        "los",
        "las",
        "le",
        "les",
        "mi",
        "tu",
        "su",
        "nuestro",
        "nuestra",
        "vuestro",
        "vuestra",
        "mis",
        "tus",
        "sus",
        "nuestros",
        "nuestras",
        "vuestros",
        "vuestras",
        "un",
        "una",
        "unos",
        "unas",
        "el",
        "la",
        "los",
        "las",
        "y",
        "o",
        "pero",
        "porque",
        "como",
        "que",
        "cuando",
        "donde",
        "cual",
        "cuales",
        "quien",
        "quienes",
        "cuanto",
        "cuanta",
        "cuantos",
        "cuantas",
        "a",
        "ante",
        "bajo",
        "cabe",
        "con",
        "contra",
        "de",
        "desde",
        "durante",
        "en",
        "entre",
        "hacia",
        "hasta",
        "mediante",
        "para",
        "por",
        "según",
        "sin",
        "sobre",
        "tras",
        "versus",
        "via",
        "no",
        "si",
        "ni",
        "e",
        "u",
        "más",
        "menos",
        "muy",
        "también",
        "solo",
        "solamente",
        "todavía",
        "aún",
        "entonces",
        "luego",
        "después",
        "antes",
        "ayer",
        "hoy",
        "mañana",
        "siempre",
        "nunca",
        "jamás",
        "tampoco",
        "ahora",
        "mientras",
    }

    # Diccionario para almacenar el top de palabras por usuario
    top_palabras_por_usuario = {}

    # Agrupar por usuario y calcular el top de palabras
    for usuario, mensajes in df.groupby("autor")["mensaje"]:
        texto_total = (
            " ".join(mensajes.dropna()).lower().strip()
        )  # Concatenar mensajes del usuario

        # Limpiar texto y contar palabras
        texto_total = remove_puntuation(
            delete_tilde(texto_total)
        )  # Quitar tildes y puntuación
        palabras = texto_total.split()

        # **Filtrar palabras vacías**
        palabras_filtradas = [p for p in palabras if p not in stopwords_es]

        # Contar solo palabras relevantes
        top_palabras = Counter(palabras_filtradas).most_common(
            10
        )  # Top 10 palabras más usadas

        # Guardar en el diccionario con el usuario como clave
        top_palabras_por_usuario[usuario] = [
            {"palabra": p, "cantidad": c} for p, c in top_palabras
        ]

    return jsonify({"top_palabras_por_usuario": top_palabras_por_usuario}), 200


##DUDOSOS NO SE PONE HASTA MEJORAR"""
@app.route("/grafico_emociones", methods=["GET"])
def grafico_emociones():
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

    df["fecha"] = pd.to_datetime(df["fecha"])

    # Analizar emociones
    def detectar_emociones(texto):
        if not isinstance(texto, str) or texto.strip() == "":
            return {}
        return NRCLex(texto).raw_emotion_scores

    df["emociones"] = df["mensaje"].apply(detectar_emociones)

    # Contadores de emociones
    emociones_totales = Counter()
    emociones_por_autor = {}

    for _, row in df.iterrows():
        autor = row["autor"]
        emociones = row["emociones"]

        emociones_totales.update(emociones)

        if autor not in emociones_por_autor:
            emociones_por_autor[autor] = Counter()
        emociones_por_autor[autor].update(emociones)

    # Traducir emociones
    traduccion_emociones = {
        "joy": "feliz",
        "sadness": "triste",
        "anger": "enojado",
        "fear": "miedo",
    }
    emociones_relevantes = ["joy", "sadness", "anger", "fear"]

    emociones_totales_traducidas = {
        traduccion_emociones[emo]: emociones_totales.get(emo, 0)
        for emo in emociones_relevantes
    }

    emociones_por_autor_traducidas = {
        autor: {
            traduccion_emociones[emo]: datos.get(emo, 0) for emo in emociones_relevantes
        }
        for autor, datos in emociones_por_autor.items()
    }

    # Calcular porcentajes
    total_emociones = sum(emociones_totales_traducidas.values())
    porcentaje_emociones = {
        emo: round((cantidad / total_emociones) * 100, 2) if total_emociones > 0 else 0
        for emo, cantidad in emociones_totales_traducidas.items()
    }

    # Encontrar el día más feliz y más triste
    dia_mas_feliz = df.loc[
        df["emociones"].apply(lambda e: e.get("joy", 0)).idxmax(), "fecha"
    ].strftime("%Y-%m-%d")
    dia_mas_triste = df.loc[
        df["emociones"].apply(lambda e: e.get("sadness", 0)).idxmax(), "fecha"
    ].strftime("%Y-%m-%d")

    return jsonify(
        {
            "emociones_totales": emociones_totales_traducidas,
            "porcentaje_emociones": porcentaje_emociones,
            "dia_mas_feliz": dia_mas_feliz,
            "dia_mas_triste": dia_mas_triste,
            "emociones_por_autor": emociones_por_autor_traducidas,
        }
    )


# Diccionario de palabras tóxicas
palabras_toxicas = {
    "idiota",
    "estúpido",
    "imbécil",
    "tonto",
    "odiar",
    "asco",
    "muere",
    "cállate",
    "gay",
    "cabro",
    "mierda",
    "puta",
    "perra",
    "prra",
    "prro" "perro" "suicidate",
    "negro",
    "tonta",
    "muerete",
    "calla",
    "concha",
    "ctm",
    "pta",
    "pto",
    "homosexual",
    "mrda",
    "estupida",
    "ctmr",
    "huevón",
    "huevonazo",
    "cojudo",
    "cojuda",
    "cojudazo",
    "maricón",
    "maricon",
    "pendejo",
    "pendeja",
    "zorra",
    "chucha",
    "chuchasumare",
    "chucha tu madre",
    "webon",
    "webona",
    "tarado",
    "tarada",
    "pelotudo",
    "pelotuda",
    "imbecil",
    "jodete",
    "vete a la mierda",
    "chibolo de mierda",
    "mamarracho",
    "baboso",
    "babosa",
    "mongol",
    "mongolo",
    "mongólica",
    "cagón",
    "cagona",
    "csm",
    "ctmre",
    "maraco",
    "maraca",
    "pajero",
    "pajera",
    "cabron",
    "carajo",
    "mierdero",
    "putamadre",
    "troll",
    "gilipollas",
    "me llega al pincho",
    "me llega",
    "alucina",
    "conchatumadre",
    "pinche",
    "culero",
    "cagaste",
    "cagada",
    "cojudeces",
    "puto",
    "rctm",
    "hdp",
    "hdpt",
    "hdmr",
    "hp",
    "qtp",
    "qlc",
    "qlo",
    "wn",
    "pndj",
    "ttr",
    "vlc",
    "qvrg",
    "y q",
    "concha de tu madre",
    "malnacido",
    "ojete",
    "forro",
    "me llega altamente",
    "qlq",
    "jodido",
    "chamare",
    "infeliz",
    "anormal",
    "come mierda",
    "huevadas",
    "puta madre",
    "reconchatumadre",
    "sidoso",
    "llorón",
    "bastardo",
    "no sirves",
    "lacra",
    "miserable",
    "muerto de hambre",
    "pobre diablo",
    "desgraciado",
    "maldito",
    "perkin",
    "parásito",
    "mariconazo",
    "lame botas",
    "chupamedias",
    "mocoso",
    "saco de mierda",
    "payaso",
    "animal",
    "bestia",
    "idioteces",
    "rata",
    "basura",
    "escombro",
    "tarúpido",
    "mrd",
    "pndj",
    "hdlgp",
    "hdlgpm",
    "fdp",
    "lpm",
    "lpqtp",
    "mrd",
    "ptm",
    "ptmr",
    "ctmm",
    "ctpt",
    "ctmrp",
    "cmr",
    "vrg",
    "kbrn",
    "mrk",
    "kkta",
    "xdmr",
    "qtpm",
    "fracasado",
    "retardado",
    "analfabestia",
    "desubicado",
    "frustrado",
    "inservible",
    "miseria humana",
    "repugnante",
    "despreciable",
    "asqueroso",
    "apestoso",
    "ladrón",
    "ratero",
    "cochino",
    "malparido",
    "insignificante",
    "basurero",
    "sabandija",
    "gusano",
    "cucaracha",
    "escoria",
    "idiotizado",
    "estafador",
    "corrupto",
    "imbecilidad",
    "mrdc",
    "zopenco",
    "tarugo",
    "baboso mental",
    "insulso",
    "fantoche",
    "desabrido",
    "panudo",
    "idioteces",
    "descerebrado",
    "loco de mrd",
    "cabeza hueca",
    "tarúpido",
    "payaso de circo",
    "bufón",
    "mata ilusiones",
    "cara de mrd",
    "pnpj",
    "vlrg",
    "cncr",
    "hdtm",
    "lacho",
    "chistoso de mrd",
    "lento",
    "mediocre",
    "sin talento",
    "sidoso",
    "gonorrea",
    "paria",
    "esquizofrénico",
    "apestado",
    "depravado",
    "asquerosidad",
    "imbécil crónico",
    "inepto",
    "inútil",
    "payaso de barrio",
    "fracasado de mrd",
    "paria de la sociedad",
    "conchadesumadre",
    "culicagado",
    "malviviente",
    "llorón de mrd",
    "aborto de la naturaleza",
    "error de la vida",
    "nn",
    "don nadie",
    "nunca vas a lograr nada",
    "asco de persona",
    "miserable fracasado",
    "gusano social",
    "ser inferior",
    "piltrafa",
    "desecho humano",
    "excremento",
    "deficiente mental",
    "retrasado",
    "cabeza de mrd",
    "soquete",
    "pelmazo",
    "pelele",
    "pan sin sal",
    "cara de nalga",
    "muerto en vida",
    "amargado de mrd",
    "calientapollas",
    "muerete lento",
    "infeliz de mrd",
    "vividor",
    "sin oficio",
    "botado a la basura",
    "sin futuro",
    "cara de verga",
    "quebrado de mrd",
    "sal de acá",
    "desperdicio de oxígeno",
    "vas a morir solo",
    "ojalá nunca hubieras nacido",
    "olvidado por la vida",
    "desnutrido mental",
    "sin neuronas",
    "residuo de sociedad",
    "plaga humana",
    "hongo mental",
    "abominación",
    "retrasado social",
    "lacayo",
    "peón de la vida",
    "esclavo de tu mediocridad",
    "jodete solo",
    "saco de pellejos",
    "kbro",
    "pobre diablo",
    "kabro",
    "kabron",
    "verga",
    "vrga",
    "autista",
    "autismo",
    "furro",
    "furra",
    "lol",
    "valorant",
    "dota",
    "dota2",
    "valo",
}


@app.route("/conteo_toxicidad", methods=["GET"])
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
        autor = row["autor"]
        mensaje = row["mensaje"]

        if isinstance(mensaje, str) and mensaje != "<Multimedia omitido>":
            palabras_mensaje = mensaje.lower().split()  # Convertir en lista de palabras
            palabras_toxicas_encontradas = [
                p for p in palabras_mensaje if p in palabras_toxicas
            ]

            # Actualizar el conteo del usuario
            conteo_por_usuario[autor].update(palabras_toxicas_encontradas)

    # Convertir el resultado en formato JSON
    resultado = {
        usuario: dict(conteo) for usuario, conteo in conteo_por_usuario.items()
    }

    return jsonify({"conteo_toxicidad": resultado}), 200


@app.route("/buscar_palabra", methods=["GET"])
def buscar_palabra():
    user_token = request.args.get("user_token")
    palabra_buscar = request.args.get("palabra")

    if not user_token or not palabra_buscar:
        return jsonify({"error": "Faltan parámetros (user_token, palabra)"}), 400

    # Convertir user_token a entero
    try:
        user_token = int(user_token)
    except ValueError:
        return jsonify({"error": "El user_token debe ser un número válido"}), 400

    # Obtener los datos desde la base de datos
    df = obtener_datos(user_token)  # Función que consulta la BD

    if df is None or df.empty:
        return jsonify({"error": "No hay datos disponibles para este user_token"}), 404

    if "autor" not in df.columns or "mensaje" not in df.columns:
        return jsonify({"error": "Faltan columnas requeridas (autor, mensaje)"}), 400

    # **Filtrar mensajes con "multimedia omitido"**
    df = df[
        ~df["mensaje"]
        .fillna("")
        .str.lower()
        .str.contains(r"\bmultimedia omitido\b", regex=True)
    ]

    # **Limpiar y convertir todo a minúsculas**
    palabra_buscar = delete_tilde(palabra_buscar.lower().strip())  # Normalizar palabra

    # Diccionario para almacenar conteo por usuario
    conteo_por_usuario = {}

    # Contador total de la palabra en todo el chat
    total_ocurrencias = 0

    # Recorremos cada usuario y sus mensajes
    for usuario, mensajes in df.groupby("autor")["mensaje"]:
        texto_total = (
            " ".join(mensajes.dropna()).lower().strip()
        )  # Concatenar mensajes del usuario

        # Limpiar y quitar tildes
        texto_total = remove_puntuation(delete_tilde(texto_total))

        # Contar cuántas veces aparece la palabra en este usuario
        conteo_usuario = texto_total.split().count(palabra_buscar)

        if conteo_usuario > 0:
            conteo_por_usuario[usuario] = conteo_usuario
            total_ocurrencias += conteo_usuario

    return (
        jsonify(
            {
                "palabra": palabra_buscar,
                "total_ocurrencias": total_ocurrencias,
                "detalle_por_usuario": conteo_por_usuario,
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
