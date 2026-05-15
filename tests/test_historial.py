import pytest
from bioimagenes.core.historial import Historial


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def historial_vacio():
    return Historial()


@pytest.fixture
def historial_con_cambios():
    h = Historial()
    h.modificar_historial("Filtro aplicado: Gaussiano")
    h.modificar_historial("Conversión a BN")
    return h


# ─────────────────────────────────────────────
# Inicialización
# ─────────────────────────────────────────────

class TestHistorialInit:

    def test_inicializa_vacio_por_defecto(self, historial_vacio):
        assert len(historial_vacio) == 0

    def test_inicializa_con_lista(self):
        cambios = [{"operacion": "test", "fecha": "2026-01-01 00:00:00"}]
        h = Historial(lista_cambios=cambios)
        assert len(h) == 1

    def test_error_si_lista_cambios_no_es_lista(self):
        with pytest.raises(TypeError):
            Historial(lista_cambios="no soy una lista")

    def test_lista_cambios_property(self, historial_vacio):
        assert isinstance(historial_vacio.lista_cambios, list)


# ─────────────────────────────────────────────
# modificar_historial
# ─────────────────────────────────────────────

class TestModificarHistorial:

    def test_agrega_cambio(self, historial_vacio):
        historial_vacio.modificar_historial("Operación X")
        assert len(historial_vacio) == 1

    def test_cambio_contiene_operacion_y_fecha(self, historial_vacio):
        historial_vacio.modificar_historial("Operación X")
        cambio = historial_vacio.lista_cambios[0]
        assert "operacion" in cambio
        assert "fecha" in cambio
        assert cambio["operacion"] == "Operación X"

    def test_multiples_cambios(self, historial_vacio):
        for i in range(5):
            historial_vacio.modificar_historial(f"Paso {i}")
        assert len(historial_vacio) == 5

    def test_error_entrada_vacia(self, historial_vacio):
        with pytest.raises(ValueError):
            historial_vacio.modificar_historial("   ")

    def test_error_entrada_no_string(self, historial_vacio):
        with pytest.raises(ValueError):
            historial_vacio.modificar_historial(123)


# ─────────────────────────────────────────────
# ultimo_cambio
# ─────────────────────────────────────────────

class TestUltimoCambio:

    def test_retorna_none_si_vacio(self, historial_vacio):
        assert historial_vacio.ultimo_cambio is None

    def test_retorna_ultimo_cambio(self, historial_con_cambios):
        assert historial_con_cambios.ultimo_cambio["operacion"] == "Conversión a BN"

    def test_ultimo_cambio_se_actualiza(self, historial_vacio):
        historial_vacio.modificar_historial("Primero")
        historial_vacio.modificar_historial("Segundo")
        assert historial_vacio.ultimo_cambio["operacion"] == "Segundo"


# ─────────────────────────────────────────────
# Métodos nativos
# ─────────────────────────────────────────────

class TestHistorialDunder:

    def test_len_vacio(self, historial_vacio):
        assert len(historial_vacio) == 0

    def test_len_con_cambios(self, historial_con_cambios):
        assert len(historial_con_cambios) == 2

    def test_iter(self, historial_con_cambios):
        cambios = list(historial_con_cambios)
        assert len(cambios) == 2
        assert all("operacion" in c for c in cambios)

    def test_str_vacio(self, historial_vacio):
        assert "vacío" in str(historial_vacio).lower()

    def test_str_con_cambios(self, historial_con_cambios):
        texto = str(historial_con_cambios)
        assert "2" in texto
        assert "Conversión a BN" in texto
