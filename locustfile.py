from locust import HttpUser, task, between
import os

class ChatAnalyzerUser(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def subir_archivo(self):
        # Asegúrate de tener este archivo real y ligero en la misma carpeta
        nombre_archivo = "chat_ejemplo.txt"
        if not os.path.exists(nombre_archivo):
            return

        with open(nombre_archivo, "rb") as f:
            files = {"file": (nombre_archivo, f, "text/plain")}
            self.client.post("/upload", files=files)

    @task(2)
    def get_statistics(self):
        self.client.get("/get_statistics?archivo_chat_id=1")

    @task(2)
    def get_plot(self):
        self.client.get("/plot.json?archivo_chat_id=1")

    @task(1)
    def get_emojis(self):
        self.client.get("/top_emojis?archivo_chat_id=1")
