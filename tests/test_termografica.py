import pytest
import numpy as np
from bioimagenes.medicas.imagen_termografica import ImagenTermografica


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def data_termica():
    np.random.seed(3)
    # Imagen en escala de grises (0–255), como vienen las térmicas
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8).astype(np.float32)


@pytest.fixture
def termografica(data_termica):
    return ImagenTermografica(data=data_termica, t_min=30.0, t_max=40.0)


@pytest.fixture
def termografica_en_celsius(data_termica):
    t = ImagenTermografica(data=data_termica, t_min=30.0, t_max=40.0)
    t.convertir_a_temperatura()
    return t


# ─────────────────────────────────────────────
# Inicialización
# ─────────────────────────────────────────────

class TestTermograficaInit:

    def test_inicializa_correctamente(self, termografica):
        assert termografica.data.ndim == 2

    def test_t_min_t_max_almacenados(self, termografica):
        assert termografica.t_min == 30.0
        assert termografica.t_max == 40.0

    def test_no_esta_en_temperatura_por_defecto(self, termografica):
        assert termografica.en_temperatura is False

    def test_error_imagen_3d(self):
        with pytest.raises(ValueError):
            ImagenTermografica(data=np.ones((10, 10, 3)))

    def test_error_t_min_mayor_que_t_max(self, data_termica):
        with pytest.raises(ValueError):
            ImagenTermografica(data=data_termica, t_min=40.0, t_max=30.0)

    def test_error_t_min_igual_t_max(self, data_termica):
        with pytest.raises(ValueError):
            ImagenTermografica(data=data_termica, t_min=35.0, t_max=35.0)

    def test_info_contiene_tipo(self, termografica):
        assert termografica.info["tipo_imagen"] == "Termografía"

    def test_info_contiene_rango(self, termografica):
        assert termografica.info["t_min_°C"] == 30.0
        assert termografica.info["t_max_°C"] == 40.0


# ─────────────────────────────────────────────
# convertir_a_temperatura
# ─────────────────────────────────────────────

class TestConvertirATemperatura:

    def test_rango_de_temperatura_correcto(self, termografica):
        termografica.convertir_a_temperatura()
        assert termografica.data.min() >= 30.0 - 0.01
        assert termografica.data.max() <= 40.0 + 0.01

    def test_marca_en_temperatura(self, termografica):
        termografica.convertir_a_temperatura()
        assert termografica.en_temperatura is True

    def test_error_doble_conversion(self, termografica):
        termografica.convertir_a_temperatura()
        with pytest.raises(RuntimeError):
            termografica.convertir_a_temperatura()

    def test_convierte_con_rango_personalizado(self, data_termica):
        t = ImagenTermografica(data=data_termica)
        t.convertir_a_temperatura(t_min=20.0, t_max=45.0)
        assert t.data.min() >= 20.0 - 0.01
        assert t.data.max() <= 45.0 + 0.01

    def test_pixel_0_equivale_t_min(self):
        data = np.zeros((10, 10), dtype=np.float32)
        t = ImagenTermografica(data=data, t_min=30.0, t_max=40.0)
        t.convertir_a_temperatura()
        assert np.allclose(t.data, 30.0)

    def test_pixel_255_equivale_t_max(self):
        data = np.full((10, 10), 255, dtype=np.float32)
        t = ImagenTermografica(data=data, t_min=30.0, t_max=40.0)
        t.convertir_a_temperatura()
        assert np.allclose(t.data, 40.0)

    def test_registra_en_historial(self, termografica):
        termografica.convertir_a_temperatura()
        assert len(termografica.info.historial) == 1


# ─────────────────────────────────────────────
# detectar_puntos_calientes
# ─────────────────────────────────────────────

class TestDetectarPuntosCalientes:

    def test_retorna_mascara_booleana(self, termografica_en_celsius):
        mascara = termografica_en_celsius.detectar_puntos_calientes(umbral=35.0)
        assert mascara.dtype == bool

    def test_shape_igual_a_imagen(self, termografica_en_celsius):
        mascara = termografica_en_celsius.detectar_puntos_calientes(umbral=35.0)
        assert mascara.shape == termografica_en_celsius.data.shape

    def test_umbral_muy_alto_mascara_vacia(self, termografica_en_celsius):
        mascara = termografica_en_celsius.detectar_puntos_calientes(umbral=9999.0)
        assert not mascara.any()

    def test_umbral_muy_bajo_mascara_llena(self, termografica_en_celsius):
        mascara = termografica_en_celsius.detectar_puntos_calientes(umbral=-9999.0)
        assert mascara.all()

    def test_error_umbral_no_numerico(self, termografica_en_celsius):
        with pytest.raises(TypeError):
            termografica_en_celsius.detectar_puntos_calientes(umbral="caliente")

    def test_registra_en_historial(self, termografica_en_celsius):
        termografica_en_celsius.detectar_puntos_calientes(umbral=35.0)
        assert len(termografica_en_celsius.info.historial) >= 1


# ─────────────────────────────────────────────
# segmentar_por_umbral
# ─────────────────────────────────────────────

class TestSegmentarPorUmbral:

    def test_retorna_mascara_booleana(self, termografica_en_celsius):
        mascara = termografica_en_celsius.segmentar_por_umbral(33.0, 37.0)
        assert mascara.dtype == bool

    def test_shape_correcto(self, termografica_en_celsius):
        mascara = termografica_en_celsius.segmentar_por_umbral(33.0, 37.0)
        assert mascara.shape == termografica_en_celsius.data.shape

    def test_pixeles_dentro_de_rango(self, termografica_en_celsius):
        mascara = termografica_en_celsius.segmentar_por_umbral(33.0, 37.0)
        valores = termografica_en_celsius.data[mascara]
        assert (valores >= 33.0).all()
        assert (valores <= 37.0).all()

    def test_error_umbral_invertido(self, termografica_en_celsius):
        with pytest.raises(ValueError):
            termografica_en_celsius.segmentar_por_umbral(37.0, 33.0)

    def test_error_umbrales_iguales(self, termografica_en_celsius):
        with pytest.raises(ValueError):
            termografica_en_celsius.segmentar_por_umbral(35.0, 35.0)


# ─────────────────────────────────────────────
# normalizar
# ─────────────────────────────────────────────

class TestNormalizarTermografica:

    def test_normaliza_a_0_1(self, termografica):
        termografica.normalizar()
        assert termografica.data.min() >= 0.0
        assert termografica.data.max() <= 1.0

    def test_error_imagen_constante(self):
        data = np.full((50, 50), 128.0, dtype=np.float32)
        t = ImagenTermografica(data=data)
        with pytest.raises(ValueError):
            t.normalizar()


# ─────────────────────────────────────────────
# Dunder
# ─────────────────────────────────────────────

class TestTermograficaDunder:

    def test_str_contiene_info(self, termografica):
        texto = str(termografica)
        assert "Termografica" in texto or "termografica" in texto.lower()
        assert "30" in texto
        assert "40" in texto

    def test_str_indica_celsius_tras_conversion(self, termografica_en_celsius):
        assert "°C" in str(termografica_en_celsius)

    def test_repr(self, termografica):
        assert "ImagenTermografica" in repr(termografica)