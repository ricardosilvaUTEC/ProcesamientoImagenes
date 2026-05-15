import pytest
import numpy as np
from bioimagenes.medicas.imagen_radiografia import ImagenRadiografia


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def data_rx():
    np.random.seed(2)
    return np.random.randint(0, 256, (128, 128), dtype=np.uint8).astype(np.float32)


@pytest.fixture
def rx(data_rx):
    return ImagenRadiografia(data=data_rx, modalidad="RX-tórax")


# ─────────────────────────────────────────────
# Inicialización
# ─────────────────────────────────────────────

class TestRadiografiaInit:

    def test_inicializa_correctamente(self, rx):
        assert rx.data.ndim == 2

    def test_modalidad_almacenada(self, rx):
        assert rx.modalidad == "RX-tórax"

    def test_modalidad_por_defecto(self, data_rx):
        rx = ImagenRadiografia(data=data_rx)
        assert rx.modalidad == "RX"

    def test_error_imagen_3d(self):
        with pytest.raises(ValueError):
            ImagenRadiografia(data=np.ones((10, 10, 3)))

    def test_info_contiene_tipo(self, rx):
        assert rx.info["tipo_imagen"] == "Radiografía"

    def test_info_contiene_modalidad(self, rx):
        assert rx.info["modalidad"] == "RX-tórax"


# ─────────────────────────────────────────────
# mejorar_contraste (CLAHE)
# ─────────────────────────────────────────────

class TestMejorarContraste:

    def test_mantiene_shape(self, rx):
        shape = rx.data.shape
        rx.mejorar_contraste()
        assert rx.data.shape == shape

    def test_registra_historial(self, rx):
        rx.mejorar_contraste()
        assert len(rx.info.historial) >= 1

    def test_actualiza_brillo(self, rx):
        rx.mejorar_contraste()
        assert isinstance(rx.info["brillo"], float)


# ─────────────────────────────────────────────
# ecualizar_histograma
# ─────────────────────────────────────────────

class TestEcualizarHistograma:

    def test_mantiene_shape(self, rx):
        shape = rx.data.shape
        rx.ecualizar_histograma()
        assert rx.data.shape == shape

    def test_registra_historial(self, rx):
        rx.ecualizar_histograma()
        assert len(rx.info.historial) >= 1


# ─────────────────────────────────────────────
# invertir
# ─────────────────────────────────────────────

class TestInvertir:

    def test_invierte_escala(self, data_rx):
        rx = ImagenRadiografia(data=data_rx.copy())
        data_original = rx.data.copy()
        rx.invertir()
        esperado = 255.0 - data_original
        np.testing.assert_array_almost_equal(rx.data, esperado)

    def test_doble_inversion_restaura_imagen(self, data_rx):
        rx = ImagenRadiografia(data=data_rx.copy())
        original = rx.data.copy()
        rx.invertir()
        rx.invertir()
        np.testing.assert_array_almost_equal(rx.data, original, decimal=3)

    def test_registra_historial(self, rx):
        rx.invertir()
        assert len(rx.info.historial) >= 1


# ─────────────────────────────────────────────
# detectar_bordes
# ─────────────────────────────────────────────

class TestDetectarBordes:

    @pytest.mark.parametrize("metodo", ["sobel", "canny", "laplacian"])
    def test_retorna_ndarray_2d(self, rx, metodo):
        resultado = rx.detectar_bordes(metodo=metodo)
        assert isinstance(resultado, np.ndarray)
        assert resultado.ndim == 2

    def test_no_modifica_data_original(self, rx):
        data_antes = rx.data.copy()
        rx.detectar_bordes(metodo="sobel")
        np.testing.assert_array_equal(rx.data, data_antes)

    def test_error_metodo_invalido(self, rx):
        with pytest.raises(ValueError):
            rx.detectar_bordes(metodo="wavelet")

    def test_registra_historial(self, rx):
        rx.detectar_bordes()
        assert len(rx.info.historial) >= 1


# ─────────────────────────────────────────────
# seleccionar_roi
# ─────────────────────────────────────────────

class TestSeleccionarROI:

    def test_retorna_nueva_instancia(self, rx):
        roi = rx.seleccionar_roi(10, 50, 10, 50)
        assert isinstance(roi, ImagenRadiografia)

    def test_shape_correcto(self, rx):
        roi = rx.seleccionar_roi(10, 50, 20, 80)
        assert roi.data.shape == (40, 60)

    def test_roi_marcada_como_cortada(self, rx):
        roi = rx.seleccionar_roi(0, 30, 0, 30)
        assert roi.info.cortada is True

    def test_error_coordenadas_invertidas(self, rx):
        with pytest.raises(ValueError):
            rx.seleccionar_roi(50, 10, 0, 30)

    def test_error_fuera_de_limite(self, rx):
        with pytest.raises(ValueError):
            rx.seleccionar_roi(0, 200, 0, 200)  # imagen es 128×128

    def test_error_tipo_coordenada(self, rx):
        with pytest.raises(TypeError):
            rx.seleccionar_roi(0.5, 30, 0, 30)


# ─────────────────────────────────────────────
# normalizar
# ─────────────────────────────────────────────

class TestNormalizarRadiografia:

    def test_normaliza_a_0_1(self, rx):
        rx.normalizar()
        assert rx.data.min() >= 0.0
        assert rx.data.max() <= 1.0

    def test_error_imagen_constante(self):
        rx = ImagenRadiografia(data=np.ones((50, 50), dtype=np.float32) * 128)
        with pytest.raises(ValueError):
            rx.normalizar()


# ─────────────────────────────────────────────
# visualizar_cluster (stub)
# ─────────────────────────────────────────────

class TestVisualizarCluster:

    def test_stub_lanza_not_implemented(self, rx):
        with pytest.raises(NotImplementedError):
            rx.visualizar_cluster()


# ─────────────────────────────────────────────
# Dunder
# ─────────────────────────────────────────────

class TestRadiografiaDunder:

    def test_str_contiene_modalidad(self, rx):
        assert "RX-tórax" in str(rx)

    def test_repr(self, rx):
        assert "ImagenRadiografia" in repr(rx)

import pytest
import numpy as np
from bioimagenes.medicas.imagen_radiografia import ImagenRadiografia


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def data_rx():
    np.random.seed(2)
    return np.random.randint(0, 256, (128, 128), dtype=np.uint8).astype(np.float32)


@pytest.fixture
def rx(data_rx):
    return ImagenRadiografia(data=data_rx, modalidad="RX-tórax")


# ─────────────────────────────────────────────
# Inicialización
# ─────────────────────────────────────────────

class TestRadiografiaInit:

    def test_inicializa_correctamente(self, rx):
        assert rx.data.ndim == 2

    def test_modalidad_almacenada(self, rx):
        assert rx.modalidad == "RX-tórax"

    def test_modalidad_por_defecto(self, data_rx):
        rx = ImagenRadiografia(data=data_rx)
        assert rx.modalidad == "RX"

    def test_error_imagen_3d(self):
        with pytest.raises(ValueError):
            ImagenRadiografia(data=np.ones((10, 10, 3)))

    def test_info_contiene_tipo(self, rx):
        assert rx.info["tipo_imagen"] == "Radiografía"

    def test_info_contiene_modalidad(self, rx):
        assert rx.info["modalidad"] == "RX-tórax"


# ─────────────────────────────────────────────
# mejorar_contraste (CLAHE)
# ─────────────────────────────────────────────

class TestMejorarContraste:

    def test_mantiene_shape(self, rx):
        shape = rx.data.shape
        rx.mejorar_contraste()
        assert rx.data.shape == shape

    def test_registra_historial(self, rx):
        rx.mejorar_contraste()
        assert len(rx.info.historial) >= 1

    def test_actualiza_brillo(self, rx):
        rx.mejorar_contraste()
        assert isinstance(rx.info["brillo"], float)


# ─────────────────────────────────────────────
# ecualizar_histograma
# ─────────────────────────────────────────────

class TestEcualizarHistograma:

    def test_mantiene_shape(self, rx):
        shape = rx.data.shape
        rx.ecualizar_histograma()
        assert rx.data.shape == shape

    def test_registra_historial(self, rx):
        rx.ecualizar_histograma()
        assert len(rx.info.historial) >= 1


# ─────────────────────────────────────────────
# invertir
# ─────────────────────────────────────────────

class TestInvertir:

    def test_invierte_escala(self, data_rx):
        rx = ImagenRadiografia(data=data_rx.copy())
        data_original = rx.data.copy()
        rx.invertir()
        esperado = 255.0 - data_original
        np.testing.assert_array_almost_equal(rx.data, esperado)

    def test_doble_inversion_restaura_imagen(self, data_rx):
        rx = ImagenRadiografia(data=data_rx.copy())
        original = rx.data.copy()
        rx.invertir()
        rx.invertir()
        np.testing.assert_array_almost_equal(rx.data, original, decimal=3)

    def test_registra_historial(self, rx):
        rx.invertir()
        assert len(rx.info.historial) >= 1


# ─────────────────────────────────────────────
# detectar_bordes
# ─────────────────────────────────────────────

class TestDetectarBordes:

    @pytest.mark.parametrize("metodo", ["sobel", "canny", "laplacian"])
    def test_retorna_ndarray_2d(self, rx, metodo):
        resultado = rx.detectar_bordes(metodo=metodo)
        assert isinstance(resultado, np.ndarray)
        assert resultado.ndim == 2

    def test_no_modifica_data_original(self, rx):
        data_antes = rx.data.copy()
        rx.detectar_bordes(metodo="sobel")
        np.testing.assert_array_equal(rx.data, data_antes)

    def test_error_metodo_invalido(self, rx):
        with pytest.raises(ValueError):
            rx.detectar_bordes(metodo="wavelet")

    def test_registra_historial(self, rx):
        rx.detectar_bordes()
        assert len(rx.info.historial) >= 1


# ─────────────────────────────────────────────
# seleccionar_roi
# ─────────────────────────────────────────────

class TestSeleccionarROI:

    def test_retorna_nueva_instancia(self, rx):
        roi = rx.seleccionar_roi(10, 50, 10, 50)
        assert isinstance(roi, ImagenRadiografia)

    def test_shape_correcto(self, rx):
        roi = rx.seleccionar_roi(10, 50, 20, 80)
        assert roi.data.shape == (40, 60)

    def test_roi_marcada_como_cortada(self, rx):
        roi = rx.seleccionar_roi(0, 30, 0, 30)
        assert roi.info.cortada is True

    def test_error_coordenadas_invertidas(self, rx):
        with pytest.raises(ValueError):
            rx.seleccionar_roi(50, 10, 0, 30)

    def test_error_fuera_de_limite(self, rx):
        with pytest.raises(ValueError):
            rx.seleccionar_roi(0, 200, 0, 200)  # imagen es 128×128

    def test_error_tipo_coordenada(self, rx):
        with pytest.raises(TypeError):
            rx.seleccionar_roi(0.5, 30, 0, 30)


# ─────────────────────────────────────────────
# normalizar
# ─────────────────────────────────────────────

class TestNormalizarRadiografia:

    def test_normaliza_a_0_1(self, rx):
        rx.normalizar()
        assert rx.data.min() >= 0.0
        assert rx.data.max() <= 1.0

    def test_error_imagen_constante(self):
        rx = ImagenRadiografia(data=np.ones((50, 50), dtype=np.float32) * 128)
        with pytest.raises(ValueError):
            rx.normalizar()


# ─────────────────────────────────────────────
# visualizar_cluster (stub)
# ─────────────────────────────────────────────

class TestVisualizarCluster:

    def test_stub_lanza_not_implemented(self, rx):
        with pytest.raises(NotImplementedError):
            rx.visualizar_cluster()


# ─────────────────────────────────────────────
# Dunder
# ─────────────────────────────────────────────

class TestRadiografiaDunder:

    def test_str_contiene_modalidad(self, rx):
        assert "RX-tórax" in str(rx)

    def test_repr(self, rx):
        assert "ImagenRadiografia" in repr(rx)