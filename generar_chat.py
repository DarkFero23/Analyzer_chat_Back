from datetime import datetime, timedelta

# Configuración
archivo_salida = "chat_super_largo.txt"
cantidad_mensajes = 50  # ajusta si quieres probar más
caracteres_por_mensaje = 3500

mensaje_base = "Este es un mensaje de prueba muy largo. " * (caracteres_por_mensaje // 40)

fecha = datetime(2023, 1, 1, 8, 0)

with open(archivo_salida, "w", encoding="utf-8") as f:
    for i in range(cantidad_mensajes):
        fecha_str = fecha.strftime("%d/%m/%Y, %I:%M %p").lower().replace("am", "a. m.").replace("pm", "p. m.")
        autor = f"Persona{i%3}"  # rotar entre autores
        linea = f"{fecha_str} - {autor}: {mensaje_base.strip()}\n"
        f.write(linea)
        fecha += timedelta(minutes=5)

print(f"✅ Archivo '{archivo_salida}' generado con {cantidad_mensajes} mensajes de {caracteres_por_mensaje} caracteres.")
