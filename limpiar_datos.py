import re
import pandas as pd
import os
import unicodedata
import regex as re  # 🔥 Necesitas usar `regex` en vez de `re`
#-----Funcion que detecta la fecha y hora de los mensajes
def Date_Chat(l):
    pattern = r'^\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2}\s[ap]\. m\. -'
    return re.match(pattern, l) is not None
#---Funcion que detecta el autor de los mensajes
def IsAuthor(l):
    pattern = r"^([\w\s\p{P}\p{S}~]+):"  # Permite letras, espacios, puntuaciones, símbolos y "~"
    result = re.match(pattern, l)
    return result is not None
#---Funcion que detecta el autor de los mensajes
def extract_format(DT):
    DT = DT.replace('\u202f', ' ').replace(',', '').strip()  # 🔹 Elimina espacios raros y comas
    match = re.search(r'\b[ap]\.\s?m\.\b', DT)  # 🔹 Acepta "p.m." y "p. m."

    if match:
        Format = match.group().lower().replace('.', '').strip()
        return "AM" if "a" in Format else "PM"

    print(f"⚠️ No se pudo extraer el formato correctamente en: {DT}")
    return "Desconocido"
#---Funcion que divide los mensajes en fecha, hora, formato, autor y mensaje
def DataPoint(line):
    SplitLine = re.split(r'\s-\s', line, maxsplit=1)

    if len(SplitLine) < 2:
        print(f"⚠️ No se pudo dividir correctamente: {line}")
        return None, None, None, None, None

    DT = SplitLine[0]  # Fecha + hora
    Message = SplitLine[1]  # Mensaje

    # 🔹 Normalizar texto para eliminar caracteres raros
    DT = unicodedata.normalize("NFKC", DT).replace('\u202f', ' ').replace(',', '')  # Normalizar texto


    DateTime = DT.split(' ')  # Separar por espacios

    if len(DateTime) < 3:
        return None, None, None, None, None

    Date = DateTime[0]
    Time = DateTime[1]
    Format = ' '.join(DateTime[2:]).strip().lower()  # Tomar todo lo que está después del Time como Format


    # 🔍 Validar que Format sea 'a. m.' o 'p. m.'
    if Format not in ["a. m.", "p. m."]:
        Format = "Desconocido"

    # 🔹 Extraer autor y mensaje
    if IsAuthor(Message):
        authormes = Message.split(': ', 1)
        Author = authormes[0]
        Message = authormes[1] if len(authormes) > 1 else "(Mensaje vacío)"
    else:
        Author = None

    return Date, Time, Format, Author, Message



#----Funcion que convierte los datos en un DataFrame y los muestra 
# 🔹 Función para procesar el contenido del chat
def DataFrame_Data(content, nombre_archivo, archivo_chat_id):
    parsedData = []
    messageBuffer = []
    Date, Time, Format, Author = None, None, None, None

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue

        if Date_Chat(line):
            if Date and Author and messageBuffer:
                message_text = ' '.join(messageBuffer) if messageBuffer else "(Mensaje vacío)"
                parsedData.append([nombre_archivo, Date, Time, Format, Author, message_text, archivo_chat_id])

            messageBuffer.clear()

            Date, Time, Format, Author, Message = DataPoint(line)
            messageBuffer.append(Message if Message else "(Mensaje vacío)")

        else:
            messageBuffer.append(line)

    if Date and Author and messageBuffer:
        message_text = ' '.join(messageBuffer) if messageBuffer else "(Mensaje vacío)"
        print(f"Autor detectado: {repr(Author)}")
        parsedData.append([nombre_archivo, Date, Time, Format, Author, message_text])

    df = pd.DataFrame(parsedData, columns=['NombreArchivo', 'Date', 'Time', 'Format', 'Author', 'Message', 'archivo_chat_id'])
    

    if df.empty:
        print("⚠️ El DataFrame está vacío, puede haber un problema con el procesamiento.")
        return df
    if 'Format' not in df.columns:
        print("❌ ERROR: La columna 'Format' no existe en el DataFrame en este punto.")
        return df  # Evitar que el código siga ejecutándose con un DataFrame incompleto


    # 🔹 **Aplicar transformaciones (sin omitir nada)**
    df = df.drop(range(0,1), errors='ignore')  # Eliminar primera fila si es necesario
    df = df.dropna().reset_index(drop=True)  # Eliminar valores nulos
    df = df[df['Message'] != ''].reset_index(drop=True)  # Eliminar mensajes vacíos

    # 🔹 Verificar cuántos autores hay
    num_autores = len(df['Author'].unique())

    # 🔹 Eliminar filas con Author vacío
    df = df.drop(df[df['Author'].isnull()].index)

    # 🔹 Unir Time con Format
   # Asegurar que Format no se pierda
    # 🔍 Verificar si 'Format' está en el DataFrame antes de trabajar con ella
    if 'Format' not in df.columns:
        print("❌ ERROR: La columna 'Format' no existe en el DataFrame. Verifica el procesamiento anterior.")
        return df  # Detener el procesamiento si falta la columna

# 🔹 Mostrar las columnas del DataFrame antes de aplicar transformaciones
    df['Formato'] = df['Format']  # Mantener AM/PM
    df['Formato'] = df['Format'].str.lower().str.strip()  # Convertir a minúsculas y eliminar espacios extra

# Reemplazar posibles valores incorrectos con los correctos
    df['Formato'] = df['Formato'].replace({"p.m.": "p. m.", "pm": "p. m.", "a.m.": "a. m.", "am": "a. m."})
# 🔹 **Asegurar que Time NO contenga "a. m." ni "p. m."**
    df['Time'] = df['Time'].str.replace("a. m.", "", regex=False).str.replace("p. m.", "", regex=False).str.strip()
    
    # 🔹 Diccionarios para nombres de días y meses
    week = {6: 'Domingo', 0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado'}
    month = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dic'}

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Day'] = df['Date'].dt.weekday.map(week)
    df['Num_Day'] = df['Date'].dt.day
    df['Num_Month'] = df['Date'].dt.month
    df['Month'] = df['Date'].dt.month.map(month)
    df['Year'] = df['Date'].dt.year
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    df['Formato'] = df['Format']  # Guardar el AM/PM
    df['archivo_chat_id'] = archivo_chat_id  
    
    df = df[['NombreArchivo', 'Date', 'Day', 'Num_Day', 'Month', 'Num_Month', 'Year', 'Time', 'Format', 'Author', 'Message', 'archivo_chat_id']]
    
    return df