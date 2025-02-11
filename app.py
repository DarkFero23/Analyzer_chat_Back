import os
import io
import psycopg2
import pandas as pd
from flask import Flask, request, jsonify, send_file
import matplotlib
matplotlib.use('Agg')
import nltk
from dotenv import load_dotenv

from nltk.sentiment import SentimentIntensityAnalyzer

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

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
# Verificar y descargar el lexicón si no está disponible
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Inicializar el analizador de sentimientos
sia = SentimentIntensityAnalyzer()

# 🔹 Inicializar Flask
app = Flask(__name__)
CORS(app)

#load_dotenv()

# 🔹 Intentar obtener la URL de la base de datos desde las variables de entorno
#SERVIDOR
DATABASE_URL = os.environ.get("DATABASE_URL")

# 🔹 Si no está definida, usar la configuración local
#LOCAL
#DATABASE_URL = "dbname=chat_analyzers user=postgres password=SylasMidM7 host=localhost port=5433"

if not DATABASE_URL:
    print("⚠️ No se encontró DATABASE_URL, usando base de datos local.")
else:
    
    print(f"🔹 Usando base de datos en Render: {DATABASE_URL}")

# 🔹 Conectar a PostgreSQL (local o en Render)
def conectar_bd():
    try:
        # Si se usa Render, requiere SSL
        sslmode = "require" if "render.com" in DATABASE_URL else None
        conn = psycopg2.connect(DATABASE_URL, sslmode=sslmode)
        print("✅ Conexión exitosa a PostgreSQL")
        return conn
    except Exception as e:
        print(f"❌ Error al conectar con PostgreSQL: {e}")
        return None
    
# 🔹 Endpoint para subir archivos y limpiarlos automáticamente
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No se ha subido ningún archivo"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    nombre_archivo = file.filename
    contenido = file.read().decode("utf-8")

    conn = conectar_bd()
    if not conn:
        return jsonify({"error": "No se pudo conectar a la base de datos"}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("CALL insertar_archivo(%s, %s);", (nombre_archivo, contenido))
        conn.commit()
        print(f"✅ Archivo '{nombre_archivo}' guardado en la base de datos.")

        # 🔹 Usar la función corregida con todas las transformaciones
        df = DataFrame_Data(contenido, nombre_archivo)

        if df.empty:
            return jsonify({"error": "No se pudieron procesar los mensajes"}), 500

        # 🔹 Convertir DataFrame a CSV en memoria para usar COPY
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, header=False, sep="|")
        csv_buffer.seek(0)

        # 🔹 Usar COPY en lugar de múltiples INSERT
        cursor.copy_from(csv_buffer, 'archivos_limpiados', sep="|", columns=[
            'nombre_archivo', 'fecha', 'dia_semana', 'num_dia', 'mes', 'num_mes', 'anio', 'hora', 'formato', 'autor', 'mensaje'
        ])
        conn.commit()
        
        print(f"✅ Mensajes limpios guardados en la base de datos (usando COPY).")
        
        global ultimos_mensajes  
        ultimos_mensajes = df.to_dict(orient="records")  

        return jsonify({"message": f"Archivo '{nombre_archivo}' subido y limpiado con éxito"}), 200

    except Exception as e:
        return jsonify({"error": f"Error al procesar el archivo: {str(e)}"}), 500

    finally:
        cursor.close()
        conn.close()

# 🔹 Endpoint para recuperar los datos limpios más recientes sin consultar la BD
@app.route('/get_last_cleaned', methods=['GET'])
def get_last_cleaned():
    global ultimos_mensajes

    if not ultimos_mensajes:  # Verifica si la lista está vacía
        return jsonify({"error": "No hay datos limpios disponibles. Carga un archivo primero."}), 404

    return jsonify({"mensajes_limpios": ultimos_mensajes}), 200


@app.route('/get_statistics', methods=['GET'])
def get_statistics():
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos limpios disponibles. Carga un archivo primero."}), 404

    # Convertir a DataFrame
    df = pd.DataFrame(ultimos_mensajes)

    # 📊 **Cálculo de estadísticas básicas**
    total_message = df.shape[0]  # Total de mensajes
    media_message = df[df['Message'] == '<Multimedia omitido>'].shape[0]  # Mensajes multimedia
    del_message = df[df['Message'] == 'Eliminaste este mensaje.'].shape[0]  # Mensajes eliminados

    # 📈 **Cálculo de porcentajes**
    media_percentage = (media_message / total_message) * 100 if total_message > 0 else 0
    del_percentage = (del_message / total_message) * 100 if total_message > 0 else 0
    total_characters = df['Message'].apply(len).sum()
    avg_characters = df['Message'].apply(len).mean() if total_message > 0 else 0

    url_pattern = r'https?://\S+|www\.\S+'  # Expresión regular para detectar URLs
    df['URL_count'] = df['Message'].apply(lambda x: len(re.findall(url_pattern, x)))
    
    total_links = df['URL_count'].sum()  # Contar todos los links en los mensajes
    

    # 📊 **Devolver las estadísticas**
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

    stats = {k: int(v) if isinstance(v, (np.int64, np.int32)) else v for k, v in stats.items()}
    return jsonify(stats), 200

@app.route('/plot.png')
def plot_png():
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos disponibles para generar el gráfico. Carga un archivo primero."}), 404

    # Convertir 'ultimos_mensajes' a DataFrame
    df = pd.DataFrame(ultimos_mensajes)

    # Verificar si el DataFrame contiene la columna 'Day'
    if 'Day' not in df.columns:
        return jsonify({"error": "Los datos no contienen la columna 'Day'."}), 400

    # Contar los registros para cada día de la semana
    active_day = df['Day'].value_counts()
    print(active_day)

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
    ax.set_yticks([])
    ax.set_title('Actividad del chat por día', fontsize=13, fontweight='bold')

    # Guardar el gráfico en un objeto BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Devolver la imagen como respuesta HTTP
    return send_file(img, mimetype='image/png')

# --- Endpoint para generar el gráfico de emojis ---
@app.route('/plot_emojis.png', methods=['GET'])
def plot_emojis_png():
    global ultimos_mensajes
    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos limpios disponibles para generar el gráfico de emojis. Carga un archivo primero."}), 404

    # Convertir ultimos_mensajes a DataFrame
    df = pd.DataFrame(ultimos_mensajes)

    # Lista para almacenar los emojis
    emojis = []
    # Lista de emojis a excluir (por ejemplo, tonos de piel u otros que no desees)
    no_emojis = ['🏻', '🏼', '🪄', '🪛', '🏿']

    # Recorrer cada mensaje y extraer los emojis
    for message in df['Message']:
        for ch in str(message):
            if ch in emoji.EMOJI_DATA and ch not in no_emojis:
                emojis.append(ch)

    # Convertir la lista en una Serie de Pandas y obtener los 10 emojis más usados
    emo_series = pd.Series(emojis)
    top_emojis = emo_series.value_counts().head(10)
    if top_emojis.empty:
        return jsonify({"error": "No se encontraron emojis en los mensajes."}), 404

    # Convertir la serie en un DataFrame
    emoji_df = pd.DataFrame(top_emojis).reset_index()
    emoji_df.columns = ['emoji', 'count']
    print(emoji_df)
    # Crear el gráfico de pastel con Matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    wedges, texts, autotexts = ax.pie(
        emoji_df['count'],
        labels=emoji_df['emoji'],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 14},
        labeldistance=1.1,
        pctdistance=0.7
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
    ax.set_title('Gráfico de los 10 emojis utilizados en el chat', fontsize=16, fontweight='bold')
    fig.tight_layout()

    # Guardar el gráfico en un objeto BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)

    return send_file(img, mimetype='image/png')

@app.route('/plot_dates.png', methods=['GET'])
def plot_dates_png():
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos limpios disponibles. Carga un archivo primero."}), 404

    # Convertir ultimos_mensajes a DataFrame
    df = pd.DataFrame(ultimos_mensajes)

    # Contar el número de mensajes por fecha y seleccionar los 10 días con mayor actividad
    TopDate = df['Date'].value_counts().head(10)
    print("Datos por fecha:", TopDate)

    # Crear la figura y los ejes para el gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))

    # Graficar la serie en un gráfico de barras
    bars = ax.bar(TopDate.index, TopDate.values, color='#32CD32')

    # Agregar etiquetas encima de cada barra
    for idx, value in enumerate(TopDate):
        # Se coloca la etiqueta centrada sobre la barra; se convierte el valor a entero para evitar problemas
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
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos limpios disponibles para generar el gráfico. Carga un archivo primero."}), 404

    # Convertir ultimos_mensajes a DataFrame
    df = pd.DataFrame(ultimos_mensajes)

    # Verificar que la columna 'Year' existe en el DataFrame
    if 'Year' not in df.columns:
        return jsonify({"error": "Los datos no contienen la columna 'Year'."}), 400

    # Contar el número de mensajes por año y ordenarlos (opcionalmente en orden ascendente)
    TopYear = df['Year'].value_counts().sort_index()  # sort_index() ordena por año
    print("Datos por año:", TopYear)

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Crear el gráfico de barras: 
    # Convertimos el índice a string para que se muestre correctamente en el eje X
    bars = ax.bar(TopYear.index.astype(str), TopYear.values, color='#32CD32')

    # Agregar etiquetas encima de cada barra
    for idx, value in enumerate(TopYear.values):
        # Colocamos la etiqueta centrada sobre la barra
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
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos limpios disponibles para generar el gráfico. Carga un archivo primero."}), 404

    # Convertir ultimos_mensajes a DataFrame
    df = pd.DataFrame(ultimos_mensajes)

    # Verificar que la columna 'Month' exista
    if 'Month' not in df.columns:
        return jsonify({"error": "Los datos no contienen la columna 'Month'."}), 400

    # Contar los mensajes según el mes y ordenar (si los nombres son 'Ene', 'Feb', etc., se puede ordenar manualmente)
    TopMonth = df['Month'].value_counts().sort_index()
    print("Conteo por mes:", TopMonth)

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(TopMonth.index, TopMonth.values, color='#32CD32')

    # Agregar etiquetas encima de cada barra
    for a, b in enumerate(TopMonth.values):
        ax.text(a - 0.12, b + 15, str(int(b)), ha='center', color='black', fontsize=10)

    # Configurar las etiquetas y el título
    ax.set_xticklabels(TopMonth.index, rotation=0, fontsize=10)
    ax.set_yticks([])
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
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos limpios disponibles para generar el gráfico de horas. Carga un archivo primero."}), 404

    # Convertir los datos limpios a DataFrame
    df = pd.DataFrame(ultimos_mensajes)

    # Verificar si las columnas necesarias existen
    if 'Time' not in df.columns or 'Format' not in df.columns:
        return jsonify({"error": "Los datos no contienen la información de tiempo y formato."}), 400

    # Extraer solo la hora (sin minutos) y convertirla en número
    df['Hora'] = df['Time'].str.split(':').str[0].astype(int)  # Obtener solo la hora en número

    # Concatenar la hora con AM o PM
    df['Hora_Formato'] = df['Hora'].astype(str) + ' ' + df['Format']

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
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos disponibles para generar la línea temporal. Carga un archivo primero."}), 404

    # Convertir los datos limpios a DataFrame
    df = pd.DataFrame(ultimos_mensajes)

    # Agrupar por año, número de mes y mes, contando la cantidad de mensajes
    TimeLine = df.groupby(['Year', 'Num_Month', 'Month']).count()['Message'].reset_index()

    # Crear una nueva columna con la combinación de Mes y Año
    TimeLine['Time'] = TimeLine.apply(lambda row: f"{row['Month']}-{row['Year']}", axis=1)

    # Crear la figura del gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(TimeLine['Time'], TimeLine['Message'], marker='o', linestyle='-', color='#32CD32')

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
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos limpios disponibles para generar el gráfico. Carga un archivo primero."}), 404

    # Convertir los datos limpios a DataFrame
    df = pd.DataFrame(ultimos_mensajes)

    # Convertir la columna de fecha a formato datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Agrupar por fecha y contar los mensajes
    Daily_LineTime = df.groupby('Date').count()['Message'].reset_index()

    # Ordenar los días con más mensajes
    Daily_LineTime_Sort = Daily_LineTime.sort_values(by='Message', ascending=False).reset_index(drop=True)

    # Crear la figura
    fig, ax = plt.subplots(figsize=(12, 6))

    # Graficar la línea temporal de mensajes por día
    ax.plot(Daily_LineTime['Date'], Daily_LineTime['Message'], color='#32CD32', marker='o', linestyle='-')

    # Resaltar los 5 días con más mensajes
    colores = ['red', 'green', 'purple', 'orange', 'black']
    for i in range(min(5, len(Daily_LineTime_Sort))):
        ax.scatter(
            Daily_LineTime_Sort.Date[i], 
            Daily_LineTime_Sort.Message[i], 
            color=colores[i], 
            marker='o', 
            label=f"{Daily_LineTime_Sort.Date[i].strftime('%Y-%m-%d')} ({Daily_LineTime_Sort.Message[i]} msg)"
        )

    # Configuración de ejes y etiquetas
    ax.set_xticks(Daily_LineTime['Date'][::max(1, len(Daily_LineTime) // 10)])  # Espaciar bien las fechas
    ax.set_xticklabels(Daily_LineTime['Date'][::max(1, len(Daily_LineTime) // 10)].dt.strftime('%Y-%m-%d'), rotation=15, fontsize=9)
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
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos limpios disponibles para generar la nube de palabras. Carga un archivo primero."}), 404

    # Obtener la fecha desde los parámetros de la URL
    fecha_str = request.args.get('fecha')  # Formato esperado: YYYY-MM-DD

    # Convertir los datos limpios a DataFrame
    df = pd.DataFrame(ultimos_mensajes)

    # Convertir la columna de fecha a formato datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Validar el rango de fechas disponibles
    fecha_min = df['Date'].min().strftime('%Y-%m-%d')
    fecha_max = df['Date'].max().strftime('%Y-%m-%d')

    if fecha_str:
        try:
            fecha_seleccionada = pd.to_datetime(fecha_str, format='%Y-%m-%d')
        except ValueError:
            return jsonify({"error": f"Formato de fecha incorrecto. Usa YYYY-MM-DD. Rango válido: {fecha_min} a {fecha_max}"}), 400

        # Verificar si la fecha está dentro del rango de datos
        if fecha_seleccionada < df['Date'].min() or fecha_seleccionada > df['Date'].max():
            return jsonify({"error": f"La fecha seleccionada está fuera del rango disponible ({fecha_min} - {fecha_max})."}), 400
    else:
        # Si no se especifica una fecha, seleccionar el día con más mensajes
        Daily_LineTime = df.groupby('Date').count()['Message'].reset_index()
        fecha_seleccionada = Daily_LineTime.sort_values(by='Message', ascending=False).iloc[0]['Date']

    # Filtrar mensajes de la fecha seleccionada
    df_fecha = df[(df['Date'] == fecha_seleccionada) & (df['Message'] != '<Multimedia omitido>')]

    if df_fecha.empty:
        return jsonify({"error": f"No hay mensajes disponibles para la fecha {fecha_seleccionada.strftime('%Y-%m-%d')}."}), 404

    # Unir los mensajes en un solo texto
    text = ' '.join(df_fecha['Message'])

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

@app.route('/analisis_sentimientos', methods=['GET'])
def analisis_sentimientos():
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos limpios disponibles. Carga un archivo primero."}), 404

    # Convertir a DataFrame
    df = pd.DataFrame(ultimos_mensajes)

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

    df['Sentimiento'] = df['Message'].apply(obtener_sentimiento)

    # Contar los tipos de sentimientos
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
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos limpios disponibles. Carga un archivo primero."}), 404

    df = pd.DataFrame(ultimos_mensajes)

    # Aplicar análisis de sentimiento
    def obtener_puntaje(texto):
        if not isinstance(texto, str) or texto == "<Multimedia omitido>":
            return 0  # Ignorar multimedia

        return sia.polarity_scores(texto)['compound']

    df['Puntaje_Sentimiento'] = df['Message'].apply(obtener_puntaje)

    # Obtener los mensajes con mayor carga emocional
    top_positivos = df.nlargest(5, 'Puntaje_Sentimiento')[['Date', 'Author', 'Message', 'Puntaje_Sentimiento']]
    top_negativos = df.nsmallest(5, 'Puntaje_Sentimiento')[['Date', 'Author', 'Message', 'Puntaje_Sentimiento']]

    # Convertir a JSON
    resultado = {
        "mensajes_mas_positivos": top_positivos.to_dict(orient="records"),
        "mensajes_mas_negativos": top_negativos.to_dict(orient="records")
    }

    return jsonify(resultado), 200

@app.route('/sentimientos_por_dia.png', methods=['GET'])
def sentimientos_por_dia():
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos limpios disponibles. Carga un archivo primero."}), 404

    df = pd.DataFrame(ultimos_mensajes)

    # Asegurar que la fecha está en formato datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

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

    df['Sentimiento'] = df['Message'].apply(obtener_sentimiento)

    # Contar los sentimientos por día
    tendencia = df.groupby(['Date', 'Sentimiento']).size().unstack(fill_value=0)

    # Graficar la tendencia de sentimientos
    fig, ax = plt.subplots(figsize=(12, 6))
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
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos limpios disponibles. Carga un archivo primero."}), 404

    df = pd.DataFrame(ultimos_mensajes)

    # Convertir la fecha a formato datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Aplicar análisis de sentimiento
    df['Puntaje_Sentimiento'] = df['Message'].apply(lambda x: sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0)

    # Agrupar por día y calcular el puntaje promedio
    sentimiento_por_dia = df.groupby('Date')['Puntaje_Sentimiento'].mean()

    # Graficar
    fig, ax = plt.subplots(figsize=(12, 6))
    sentimiento_por_dia.plot(kind='bar', color=['green' if v > 0 else 'red' for v in sentimiento_por_dia], ax=ax)

    ax.set_title("Sentimiento Promedio por Día", fontsize=13, fontweight='bold')
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Puntaje de Sentimiento", fontsize=11)
    plt.xticks(rotation=45, fontsize=9)
    plt.grid()

    # Guardar imagen en BytesIO
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)

    return send_file(img, mimetype='image/png')


@app.route('/top_palabras_sentimiento', methods=['GET'])
def top_palabras_sentimiento():
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos limpios disponibles. Carga un archivo primero."}), 404

    df = pd.DataFrame(ultimos_mensajes)

    # Aplicar análisis de sentimiento
    df['Puntaje_Sentimiento'] = df['Message'].apply(lambda x: sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0)

    # Separar mensajes positivos y negativos
    positivos = ' '.join(df[df['Puntaje_Sentimiento'] > 0]['Message']).lower()
    negativos = ' '.join(df[df['Puntaje_Sentimiento'] < 0]['Message']).lower()

    # Limpiar y contar palabras
    def contar_palabras(texto):
        texto = remove_puntuation(delete_tilde(texto))
        palabras = texto.split()
        return Counter(palabras).most_common(10)

    top_positivas = contar_palabras(positivos)
    top_negativas = contar_palabras(negativos)

    return jsonify({"top_palabras_positivas": top_positivas, "top_palabras_negativas": top_negativas}), 200



@app.route('/grafico_emociones.png', methods=['GET'])
def grafico_emociones():
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos limpios disponibles. Carga un archivo primero."}), 404

    df = pd.DataFrame(ultimos_mensajes)

    def detectar_emociones(texto):
        if not isinstance(texto, str):
            return {}

        emociones = NRCLex(texto).raw_emotion_scores
        return emociones

    # Aplicar la función a todos los mensajes
    df['Emociones'] = df['Message'].apply(detectar_emociones)

    # Contar las emociones
    emociones_totales = Counter()
    for emociones in df['Emociones']:
        emociones_totales.update(emociones)

    # Seleccionar las principales emociones a graficar
    emociones_relevantes = ['joy', 'sadness', 'anger', 'fear']
    datos_emociones = {emocion: emociones_totales.get(emocion, 0) for emocion in emociones_relevantes}

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


@app.route('/mensajes_conflictivos', methods=['GET'])
def mensajes_conflictivos():
    global ultimos_mensajes

    if not ultimos_mensajes:
        return jsonify({"error": "No hay datos limpios disponibles. Carga un archivo primero."}), 404

    df = pd.DataFrame(ultimos_mensajes)

    def detectar_toxicidad(texto):
        if not isinstance(texto, str):
            return 0
        return TextBlob(texto).sentiment.polarity

    df['Toxicidad'] = df['Message'].apply(detectar_toxicidad)

    # Obtener los mensajes más tóxicos (negativos extremos)
    top_conflictivos = df.nsmallest(5, 'Toxicidad')[['Date', 'Author', 'Message', 'Toxicidad']]

    return jsonify({"mensajes_conflictivos": top_conflictivos.to_dict(orient="records")}), 200
if __name__ == '__main__':
    app.run(debug=True, port=5000)

