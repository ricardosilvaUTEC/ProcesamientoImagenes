import pytest
import numpy as np
from bioimagenes.filtros.filtro import Filtro


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def kernel_promedio():
    return np.ones((3, 3), dtype=np.float32) / 9.0


@pytest.fixture
def filtro_promedio(kernel_promedio):
    return Filtro(tipo="Promedio", kernel=kernel_promedio)


@pytest.fixture
def imagen_gris():
    np.random.seed(42)
    return np.random.randint(0, 256, (50, 50), dtype=np.uint8).astype(np.float32)


@pytest.fixture
def imagen_rgb():
    np.random.seed(42)
    return np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8).astype(np.float32)


# ─────────────────────────────────────────────
# Inicialización
# ─────────────────────────────────────────────

class TestFiltroInit:

    def test_inicializa_correctamente(self, filtro_promedio):
        assert filtro_promedio.tipo == "Promedio"
        assert filtro_promedio.kernel.shape == (3, 3)

    def test_kernel_convertido_a_float32(self, kernel_promedio):
        f = Filtro(tipo="Test", kernel=kernel_promedio.astype(np.float64))
        assert f.kernel.dtype == np.float32

    def test_error_tipo_no_string(self, kernel_promedio):
        with pytest.raises(TypeError):
            Filtro(tipo=123, kernel=kernel_promedio)

    def test_error_tipo_vacio(self, kernel_promedio):
        with pytest.raises(TypeError):
            Filtro(tipo="  ", kernel=kernel_promedio)

    def test_error_kernel_no_ndarray(self):
        with pytest.raises(TypeError):
            Filtro(tipo="Test", kernel=[[1, 0], [0, 1]])

    def test_error_kernel_no_2d(self):
        with pytest.raises(ValueError):
            Filtro(tipo="Test", kernel=np.ones((3, 3, 3)))

    def test_error_kernel_vacio(self):
        with pytest.raises(ValueError):
            Filtro(tipo="Test", kernel=np.array([]).reshape(0, 0))


# ─────────────────────────────────────────────
# Properties
# ─────────────────────────────────────────────

class TestFiltroProperties:

    def test_tamanio(self, filtro_promedio):
        assert filtro_promedio.tamaño == (3, 3)

    def test_tipo(self, filtro_promedio):
        assert filtro_promedio.tipo == "Promedio"

    def test_kernel_es_ndarray(self, filtro_promedio):
        assert isinstance(filtro_promedio.kernel, np.ndarray)


# ─────────────────────────────────────────────
# convolucion / aplicar
# ─────────────────────────────────────────────

class TestFiltroConvolucion:

    def test_convolucion_imagen_gris_mantiene_shape(self, filtro_promedio, imagen_gris):
        resultado = filtro_promedio.convolucion(imagen_gris)
        assert resultado.shape == imagen_gris.shape

    def test_convolucion_imagen_rgb_mantiene_shape(self, filtro_promedio, imagen_rgb):
        resultado = filtro_promedio.convolucion(imagen_rgb)
        assert resultado.shape == imagen_rgb.shape

    def test_aplicar_equivale_a_convolucion(self, filtro_promedio, imagen_gris):
        r1 = filtro_promedio.convolucion(imagen_gris)
        r2 = filtro_promedio.aplicar(imagen_gris)
        np.testing.assert_array_almost_equal(r1, r2)

    def test_suavizado_reduce_varianza(self, filtro_promedio, imagen_gris):
        resultado = filtro_promedio.aplicar(imagen_gris)
        assert resultado.std() <= imagen_gris.std()

    def test_error_convolucion_no_ndarray(self, filtro_promedio):
        with pytest.raises(TypeError):
            filtro_promedio.convolucion([[1, 2], [3, 4]])

    def test_error_convolucion_imagen_1d(self, filtro_promedio):
        with pytest.raises(ValueError):
            filtro_promedio.convolucion(np.ones(100))

    def test_filtro_sobel_detecta_bordes(self, imagen_gris):
        kernel_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        f = Filtro(tipo="Sobel", kernel=kernel_sobel)
        resultado = f.aplicar(imagen_gris)
        # Una imagen uniforme tiene bordes ≈ 0
        uniforme = np.ones((50, 50), dtype=np.float32) * 128
        borde_uniforme = f.aplicar(uniforme)
        assert borde_uniforme.std() < resultado.std()


# ─────────────────────────────────────────────
# Dunder
# ─────────────────────────────────────────────

class TestFiltroDunder:

    def test_str_contiene_tipo_y_tamanio(self, filtro_promedio):
        texto = str(filtro_promedio)
        assert "Promedio" in texto
        assert "3" in texto

    def test_repr(self, filtro_promedio):
        assert "Filtro" in repr(filtro_promedio)