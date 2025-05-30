import pytest
from app import app
import pandas as pd

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_df():
    # DataFrame de ejemplo para mocking
    data = {
        "autor": ["Ana", "Luis", "Ana"],
        "mensaje": ["hola mundo", "idiota", "<Multimedia omitido>"],
        "fecha": ["01/01/2024", "01/01/2024", "02/01/2024"],
        "hora": ["12:00", "16:00", "20:00"],
        "formato": ["AM", "PM", "PM"],
        "dia_semana": ["Lunes", "Lunes", "Martes"],
        "num_dia": [1, 1, 2],
        "mes": ["Ene", "Ene", "Ene"],
        "num_mes": [1, 1, 1],
        "anio": [2024, 2024, 2024],
    }
    return pd.DataFrame(data)

@pytest.fixture(autouse=True)
def patch_obtener_datos(mocker, mock_df):
    mocker.patch("app.obtener_datos", return_value=mock_df)

def test_conteo_toxicidad(client):
    resp = client.get("/conteo_toxicidad?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "conteo_toxicidad" in resp.json

def test_buscar_palabra(client):
    resp = client.get("/buscar_palabra?archivo_chat_id=1&palabra=hola")
    assert resp.status_code == 200
    assert resp.json["palabra"] == "hola"

def test_contar_palabra(client):
    resp = client.get("/contar_palabra?archivo_chat_id=1&palabra=idiota")
    assert resp.status_code == 200
    assert "conteo_por_autor" in resp.json

def test_autor_que_reanuda_mas(client):
    resp = client.get("/autor_que_reanuda_mas?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "autor_que_mas_reanuda" in resp.json

def test_get_statistics(client):
    resp = client.get("/get_statistics?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "resumen_general" in resp.json

def test_plot_json(client):
    resp = client.get("/plot.json?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "activity_per_day" in resp.json

def test_obtener_top_emojis(client):
    resp = client.get("/top_emojis?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "top_emojis" in resp.json

def test_plot_dates_json(client):
    resp = client.get("/plot_dates.json?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "top_active_days" in resp.json

def test_plot_mensajes_año_json(client):
    resp = client.get("/plot_mensajes_año.json?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "mensajes_por_año" in resp.json

def test_plot_mensajes_mes(client):
    resp = client.get("/plot_mensajes_mes.json?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "mensajes_por_mes" in resp.json

def test_horas_completo_json(client):
    resp = client.get("/horas_completo.json?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "datos_horas" in resp.json

def test_plot_timeline_json(client):
    resp = client.get("/plot_timeline.json?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "timeline" in resp.json

def test_plot_mensajes_por_dia(client):
    resp = client.get("/plot_mensajes_por_dia.json?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "timelineDay" in resp.json

def test_nube_palabras(client):
    resp = client.get("/nube_palabras?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "palabras" in resp.json

def test_analisis_sentimientos(client):
    resp = client.get("/analisis_sentimientos?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "positivo" in resp.json

def test_mensajes_mayor_emocion(client):
    resp = client.get("/mensajes_mayor_emocion?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "mensajes_mas_positivos" in resp.json

def test_sentimientos_por_dia(client):
    resp = client.get("/sentimientos_por_dia.json?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "datos" in resp.json

def test_sentimiento_promedio_dia(client):
    resp = client.get("/sentimiento_promedio_dia?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "top_dias" in resp.json

def test_top_palabras_usuario(client):
    resp = client.get("/top_palabras_usuario?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "top_palabras_por_usuario" in resp.json

def test_grafico_emociones(client):
    resp = client.get("/grafico_emociones?archivo_chat_id=1")
    assert resp.status_code == 200
    assert "emociones_totales" in resp.json