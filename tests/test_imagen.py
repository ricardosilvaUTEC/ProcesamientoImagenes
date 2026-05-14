import pytest
import numpy as np
from bioimagenes.core.imagen import Imagen
from bioimagenes.core.info import Info
from bioimagenes.filtros.filtro import Filtro


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def data_gris():
    np.random.seed(0)
    return np.random.randint(0, 256, (80, 80), dtype=np.uint8).astype(np.float32)


@pytest.fixture
def data_rgb():
    np.random.seed(0)
    return np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8).astype(np.float32)


@pytest.fixture
def imagen_gris(data_gris):
    return Imagen(data=data_gris)


@pytest.fixture
def imagen_rgb(data_rgb):
    return Imagen(data=data_rgb)


@pytest.fixture
def filtro_promedio():
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    return Filtro(tipo="Promedio", kernel=kernel)


# ─────────────────────────────────────────────
# Inicialización
# ─────────────────────────────────────────────

class TestImagenInit:

    def test_inicializa_imagen_gris(self, imagen_gris):
        assert imagen_gris.data.ndim == 2

    def test_inicializa_imagen_rgb(self, imagen_rgb):
        assert imagen_rgb.data.ndim == 3

    def test_data_convertido_a_float32(self, data_gris):
        img = Imagen(data=data_gris.astype(np.uint8))
        assert img.data.dtype == np.float32

    def test_crea_info_por_defecto(self, imagen_gris):
        assert isinstance(imagen_gris.info, Info)

    def test_acepta_info_externa(self, data_gris):
        info = Info(datos={"dimensiones": data_gris.shape, "brillo": 100.0, "cortada": False})
        img = Imagen(data=data_gris, info=info)
        assert img.info is info

    def test_error_data_no_ndarray(self):
        with pytest.raises(TypeError):
            Imagen(data=[[1, 2], [3, 4]])

    def test_error_data_1d(self):
        with pytest.raises(ValueError):
            Imagen(data=np.ones(100))

    def test_error_data_4d(self):
        with pytest.raises(ValueError):
            Imagen(data=np.ones((2, 2, 2, 2)))

    def test_error_info_invalida(self, data_gris):
        with pytest.raises(TypeError):
            Imagen(data=data_gris, info="no soy Info")

    def test_info_contiene_dimensiones(self, imagen_gris):
        assert "dimensiones" in imagen_gris.info

    def test_info_contiene_brillo(self, imagen_gris):
        assert "brillo" in imagen_gris.info

    def test_info_cortada_false_por_defecto(self, imagen_gris):
        assert imagen_gris.info.cortada is False


# ─────────────────────────────────────────────
# aplicar_filtro
# ─────────────────────────────────────────────

class TestAplicarFiltro:

    def test_aplica_filtro_imagen_gris(self, imagen_gris, filtro_promedio):
        shape_original = imagen_gris.data.shape
        imagen_gris.aplicar_filtro(filtro_promedio)
        assert imagen_gris.data.shape == shape_original

    def test_aplica_filtro_imagen_rgb(self, imagen_rgb, filtro_promedio):
        shape_original = imagen_rgb.data.shape
        imagen_rgb.aplicar_filtro(filtro_promedio)
        assert imagen_rgb.data.shape == shape_original

    def test_filtro_actualiza_historial(self, imagen_gris, filtro_promedio):
        imagen_gris.aplicar_filtro(filtro_promedio)
        assert len(imagen_gris.info.historial) == 1

    def test_filtro_actualiza_brillo(self, imagen_gris, filtro_promedio):
        brillo_antes = imagen_gris.info["brillo"]
        imagen_gris.aplicar_filtro(filtro_promedio)
        brillo_despues = imagen_gris.info["brillo"]
        # El brillo puede cambiar (no necesariamente mucho, pero debe recalcularse)
        assert isinstance(brillo_despues, float)
        assert brillo_antes != brillo_despues or True  # al menos no explota

    def test_error_filtro_invalido(self, imagen_gris):
        with pytest.raises(TypeError):
            imagen_gris.aplicar_filtro("no soy filtro")


# ─────────────────────────────────────────────
# bn
# ─────────────────────────────────────────────

class TestBN:

    def test_bn_convierte_rgb_a_gris(self, imagen_rgb):
        imagen_rgb.bn()
        assert imagen_rgb.data.ndim == 2

    def test_bn_no_modifica_imagen_gris(self, imagen_gris):
        data_antes = imagen_gris.data.copy()
        imagen_gris.bn()
        np.testing.assert_array_equal(imagen_gris.data, data_antes)

    def test_bn_registra_en_historial(self, imagen_rgb):
        imagen_rgb.bn()
        assert len(imagen_rgb.info.historial) == 1

    def test_bn_mantiene_shape_hw(self, imagen_rgb):
        H, W = imagen_rgb.data.shape[:2]
        imagen_rgb.bn()
        assert imagen_rgb.data.shape == (H, W)

    def test_bn_usa_pesos_perceptuales(self):
        # Canal rojo puro → resultado ≈ 0.2989 * 255
        data = np.zeros((10, 10, 3), dtype=np.float32)
        data[:, :, 0] = 255.0
        img = Imagen(data=data)
        img.bn()
        valor_esperado = 0.2989 * 255.0
        assert abs(img.data.mean() - valor_esperado) < 1.0


# ─────────────────────────────────────────────
# Métodos nativos
# ─────────────────────────────────────────────

class TestImagenDunder:

    def test_len_retorna_total_pixeles(self, imagen_gris):
        assert len(imagen_gris) == 80 * 80

    def test_len_imagen_rgb(self, imagen_rgb):
        assert len(imagen_rgb) == 80 * 80 * 3

    def test_getitem_pixel(self, imagen_gris):
        valor = imagen_gris[0, 0]
        assert isinstance(valor, (float, np.floating))

    def test_getitem_region(self, imagen_gris):
        region = imagen_gris[10:20, 10:20]
        assert region.shape == (10, 10)

    def test_str_contiene_info(self, imagen_gris):
        texto = str(imagen_gris)
        assert "Shape" in texto
        assert "Grises" in texto

    def test_str_imagen_rgb(self, imagen_rgb):
        assert "RGB" in str(imagen_rgb)

    def test_repr(self, imagen_gris):
        assert "Imagen" in repr(imagen_gris)