import pytest
from limpiar_datos import detectar_plataforma, Date_Chat, IsAuthor, DataPoint

def test_detectar_plataforma_android():
    lineas = ["12/05/2024, 10:00 a. m. - Juan: Hola", "Esto es otra línea"]
    assert detectar_plataforma(lineas) == "android"

def test_detectar_plataforma_iphone():
    lineas = ["[12/05/2024, 10:00:00 a. m.] Juan: Hola", "Otra línea"]
    assert detectar_plataforma(lineas) == "iphone"

def test_Date_Chat_valido():
    assert Date_Chat("12/05/2024, 10:00 a. m. - Juan: Hola") == True

def test_IsAuthor_valido():
    assert IsAuthor("Juan Pérez: Hola") == True


def test_DataPoint_valido():
    line = "12/05/2024, 10:00 a. m. - Juan: Hola"
    date, time, fmt, author, msg = DataPoint(line)
    assert date == "12/05/2024"
    assert time == "10:00"
    assert fmt == "a. m."
    assert author == "Juan"
    assert msg == "Hola"
