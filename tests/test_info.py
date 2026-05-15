import pytest
from bioimagenes.core.historial import Historial
from bioimagenes.core.info import Info


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def info_basica():
    return Info(datos={"dimensiones": (100, 100), "brillo": 128.0, "cortada": False})


@pytest.fixture
def info_vacia():
    return Info()


# ─────────────────────────────────────────────
# Inicialización
# ─────────────────────────────────────────────

class TestInfoInit:

    def test_inicializa_sin_argumentos(self, info_vacia):
        assert isinstance(info_vacia, Info)

    def test_inicializa_con_datos(self, info_basica):
        assert info_basica["dimensiones"] == (100, 100)

    def test_crea_historial_por_defecto(self, info_vacia):
        assert isinstance(info_vacia.historial, Historial)

    def test_acepta_historial_externo(self):
        h = Historial()
        info = Info(historial=h)
        assert info.historial is h

    def test_error_datos_no_dict(self):
        with pytest.raises(TypeError):
            Info(datos="no soy un dict")

    def test_error_historial_invalido(self):
        with pytest.raises(TypeError):
            Info(historial="no soy un historial")


# ─────────────────────────────────────────────
# Properties nombradas
# ─────────────────────────────────────────────

class TestInfoProperties:

    def test_dimensiones_getter(self, info_basica):
        assert info_basica.dimensiones == (100, 100)

    def test_dimensiones_setter(self, info_basica):
        info_basica.dimensiones = (200, 200)
        assert info_basica.dimensiones == (200, 200)

    def test_dimensiones_setter_error(self, info_basica):
        with pytest.raises(TypeError):
            info_basica.dimensiones = [200, 200]  # lista, no tupla

    def test_brillo_getter(self, info_basica):
        assert info_basica.brillo == 128.0

    def test_brillo_setter(self, info_basica):
        info_basica.brillo = 200.0
        assert info_basica.brillo == 200.0

    def test_brillo_setter_error(self, info_basica):
        with pytest.raises(TypeError):
            info_basica.brillo = "mucho"

    def test_cortada_getter(self, info_basica):
        assert info_basica.cortada is False

    def test_cortada_setter(self, info_basica):
        info_basica.cortada = True
        assert info_basica.cortada is True

    def test_cortada_setter_error(self, info_basica):
        with pytest.raises(TypeError):
            info_basica.cortada = 1  # int, no bool

    def test_dimensiones_none_si_no_existe(self, info_vacia):
        assert info_vacia.dimensiones is None


# ─────────────────────────────────────────────
# Métodos nativos
# ─────────────────────────────────────────────

class TestInfoDunder:

    def test_contains_clave_existente(self, info_basica):
        assert "dimensiones" in info_basica

    def test_contains_clave_inexistente(self, info_basica):
        assert "inexistente" not in info_basica

    def test_getitem_clave_existente(self, info_basica):
        assert info_basica["brillo"] == 128.0

    def test_getitem_clave_inexistente(self, info_basica):
        with pytest.raises(KeyError):
            _ = info_basica["no_existe"]

    def test_setitem(self, info_vacia):
        info_vacia["nueva_clave"] = 42
        assert info_vacia["nueva_clave"] == 42

    def test_str_contiene_claves(self, info_basica):
        texto = str(info_basica)
        assert "dimensiones" in texto
        assert "brillo" in texto
