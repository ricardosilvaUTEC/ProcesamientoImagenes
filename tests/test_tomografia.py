import pytest
import numpy as np
from bioimagenes.medicas.imagen_tomografia import ImagenTomografia


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def volumen_3d():
    np.random.seed(1)
    # Simulamos 10 cortes de 64×64 con valores tipo CT (-1000 a 3000 HU)
    return np.random.uniform(-1000, 3000, (10, 64, 64)).astype(np.float32)


@pytest.fixture
def corte_2d():
    np.random.seed(1)
    return np.random.uniform(0, 255, (64, 64)).astype(np.float32)


@pytest.fixture
def ct_3d(volumen_3d):
    return ImagenTomografia(data=volumen_3d)


@pytest.fixture
def ct_2d(corte_2d):
    return ImagenTomografia(data=corte_2d)


# ─────────────────────────────────────────────
# Inicialización
# ─────────────────────────────────────────────

class TestTomografiaInit:

    def test_inicializa_3d(self, ct_3d):
        assert ct_3d.data.ndim == 3

    def test_inicializa_2d(self, ct_2d):
        assert ct_2d.data.ndim == 2

    def test_n_slices_3d(self, ct_3d):
        assert ct_3d.n_slices == 10

    def test_n_slices_2d(self, ct_2d):
        assert ct_2d.n_slices == 1

    def test_espaciado_por_defecto(self, ct_3d):
        assert ct_3d.espaciado == (1.0, 1.0, 1.0)

    def test_espaciado_personalizado(self, volumen_3d):
        ct = ImagenTomografia(data=volumen_3d, espaciado=(2.5, 0.7, 0.7))
        assert ct.espaciado == (2.5, 0.7, 0.7)

    def test_error_data_4d(self):
        with pytest.raises(ValueError):
            ImagenTomografia(data=np.ones((5, 5, 5, 5)))

    def test_error_espaciado_negativo(self, volumen_3d):
        with pytest.raises(ValueError):
            ImagenTomografia(data=volumen_3d, espaciado=(-1.0, 1.0, 1.0))

    def test_error_espaciado_no_tupla(self, volumen_3d):
        with pytest.raises(TypeError):
            ImagenTomografia(data=volumen_3d, espaciado=[1.0, 1.0, 1.0])

    def test_info_contiene_tipo(self, ct_3d):
        assert ct_3d.info["tipo_imagen"] == "Tomografía"

    def test_info_contiene_n_slices(self, ct_3d):
        assert ct_3d.info["n_slices"] == 10


# ─────────────────────────────────────────────
# convertir_a_hu
# ─────────────────────────────────────────────

class TestConvertirAHU:

    def test_convierte_correctamente(self):
        data = np.ones((5, 64, 64), dtype=np.float32) * 100.0
        ct = ImagenTomografia(data=data)
        ct.convertir_a_hu(slope=1.0, intercept=-1024.0)
        assert np.allclose(ct.data, 100.0 - 1024.0)

    def test_marca_en_hu(self, volumen_3d):
        ct = ImagenTomografia(data=volumen_3d)
        assert ct.en_hu is False
        ct.convertir_a_hu()
        assert ct.en_hu is True

    def test_error_doble_conversion(self, ct_3d):
        ct_3d.convertir_a_hu()
        with pytest.raises(RuntimeError):
            ct_3d.convertir_a_hu()

    def test_registra_en_historial(self, ct_3d):
        ct_3d.convertir_a_hu()
        assert len(ct_3d.info.historial) == 1


# ─────────────────────────────────────────────
# get_slice
# ─────────────────────────────────────────────

class TestGetSlice:

    def test_retorna_corte_correcto(self, ct_3d, volumen_3d):
        corte = ct_3d.get_slice(0)
        np.testing.assert_array_almost_equal(corte, volumen_3d[0])

    def test_corte_es_2d(self, ct_3d):
        assert ct_3d.get_slice(5).ndim == 2

    def test_error_indice_fuera_de_rango(self, ct_3d):
        with pytest.raises(IndexError):
            ct_3d.get_slice(100)

    def test_error_indice_negativo_invalido(self, ct_3d):
        with pytest.raises(IndexError):
            ct_3d.get_slice(-1)

    def test_error_indice_no_entero(self, ct_3d):
        with pytest.raises(TypeError):
            ct_3d.get_slice(1.5)

    def test_get_slice_en_2d_retorna_data(self, ct_2d):
        corte = ct_2d.get_slice(0)
        assert corte.ndim == 2


# ─────────────────────────────────────────────
# ajustar_ventana
# ─────────────────────────────────────────────

class TestAjustarVentana:

    def test_retorna_ndarray(self, ct_3d):
        resultado = ct_3d.ajustar_ventana(center=40, width=400)
        assert isinstance(resultado, np.ndarray)

    def test_valores_en_rango_0_255(self, ct_3d):
        resultado = ct_3d.ajustar_ventana(center=40, width=400)
        assert resultado.min() >= 0.0
        assert resultado.max() <= 255.0

    def test_error_width_negativo(self, ct_3d):
        with pytest.raises(ValueError):
            ct_3d.ajustar_ventana(center=40, width=-100)

    def test_registra_en_historial(self, ct_3d):
        ct_3d.ajustar_ventana(center=40, width=400)
        assert len(ct_3d.info.historial) >= 1


# ─────────────────────────────────────────────
# aplicar_ventana_predefinida
# ─────────────────────────────────────────────

class TestVentanaPredefinida:

    @pytest.mark.parametrize("nombre", ["pulmón", "hueso", "tejido", "cerebro", "abdomen"])
    def test_ventanas_validas(self, ct_3d, nombre):
        resultado = ct_3d.aplicar_ventana_predefinida(nombre)
        assert resultado is not None

    def test_error_ventana_inexistente(self, ct_3d):
        with pytest.raises(ValueError):
            ct_3d.aplicar_ventana_predefinida("inexistente")


# ─────────────────────────────────────────────
# normalizar
# ─────────────────────────────────────────────

class TestNormalizarTomografia:

    def test_normaliza_a_0_1(self, ct_3d):
        ct_3d.normalizar()
        assert ct_3d.data.min() >= 0.0
        assert ct_3d.data.max() <= 1.0

    def test_error_rango_igual(self):
        data = np.ones((5, 10, 10), dtype=np.float32)
        ct = ImagenTomografia(data=data)
        with pytest.raises(ValueError):
            ct.normalizar()


# ─────────────────────────────────────────────
# cargar_volumen
# ─────────────────────────────────────────────

class TestCargarVolumen:

    def test_reemplaza_volumen(self, ct_3d):
        nuevo = np.ones((20, 64, 64), dtype=np.float32)
        ct_3d.cargar_volumen(nuevo)
        assert ct_3d.n_slices == 20

    def test_error_no_3d(self, ct_3d):
        with pytest.raises(ValueError):
            ct_3d.cargar_volumen(np.ones((64, 64)))


# ─────────────────────────────────────────────
# Dunder
# ─────────────────────────────────────────────

class TestTomografiaDunder:

    def test_str_contiene_info(self, ct_3d):
        texto = str(ct_3d)
        assert "Tomografia" in texto or "tomografia" in texto.lower()
        assert "Slices" in texto

    def test_repr(self, ct_3d):
        assert "ImagenTomografia" in repr(ct_3d)
