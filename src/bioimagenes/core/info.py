from bioimagenes.core.historial import Historial


class Info:
    """
    Clase para almacenar y gestionar los metadatos asociados a una imagen.

    Funciona de manera similar a un diccionario, permitiendo acceder,
    modificar y consultar información relevante como dimensiones,
    brillo y estado de la imagen.

    Se integra con la clase Historial para mantener coherencia
    entre los datos y las transformaciones aplicadas.
    """

    def __init__(self, datos: dict = None, historial: Historial = None):
        """
        Inicializa una instancia de la clase Info.

        Parámetros
        ----------
        datos : dict, opcional
            Diccionario con los metadatos iniciales.
            Ejemplo: {"dimensiones": (100, 100), "brillo": 120, "cortada": False}
            Si no se proporciona, se inicializa con valores por defecto.
        historial : Historial, opcional
            Instancia de la clase Historial asociada a la imagen.
            Si no se proporciona, se crea una nueva.
        """
        # Validamos el tipo de datos
        if datos is not None and not isinstance(datos, dict):
            raise TypeError("datos debe ser un diccionario.")

        # Validamos el historial
        if historial is not None and not isinstance(historial, Historial):
            raise TypeError("historial debe ser una instancia de Historial.")

        # Guardamos los metadatos (o inicializamos vacío)
        self._datos: dict = datos if datos is not None else {}

        # Historial asociado
        self._historial: Historial = historial if historial is not None else Historial()

    # ------------------------------------------------------------------
    # Propiedades que mapean las claves del dict a atributos nombrados
    # ------------------------------------------------------------------

    @property
    def dimensiones(self) -> tuple | None:
        """Resolución espacial y número de canales de la imagen."""
        return self._datos.get("dimensiones")

    @dimensiones.setter
    def dimensiones(self, valor: tuple) -> None:
        if not isinstance(valor, tuple):
            raise TypeError("dimensiones debe ser una tupla.")
        self._datos["dimensiones"] = valor

    @property
    def brillo(self) -> float | None:
        """Valor promedio de brillo de la imagen."""
        return self._datos.get("brillo")

    @brillo.setter
    def brillo(self, valor: float) -> None:
        if not isinstance(valor, (int, float)):
            raise TypeError("brillo debe ser un número.")
        self._datos["brillo"] = float(valor)

    @property
    def cortada(self) -> bool:
        """Indica si la imagen proviene de un recorte."""
        return self._datos.get("cortada", False)

    @cortada.setter
    def cortada(self, valor: bool) -> None:
        if not isinstance(valor, bool):
            raise TypeError("cortada debe ser un booleano.")
        self._datos["cortada"] = valor

    @property
    def historial(self) -> Historial:
        """Instancia de Historial asociada."""
        return self._historial

    @historial.setter
    def historial(self, valor: Historial) -> None:
        if not isinstance(valor, Historial):
            raise TypeError("historial debe ser una instancia de Historial.")
        self._historial = valor

    # Acceso directo al dict interno (lectura)
    @property
    def datos(self) -> dict:
        """Diccionario completo de metadatos."""
        return self._datos

    # ------------------------------------------------------------------
    # Métodos nativos / dunder
    # ------------------------------------------------------------------

    def __contains__(self, clave: str) -> bool:
        """
        Permite verificar si una clave existe en los metadatos.

        Ejemplo: "brillo" in info  →  True / False
        """
        return clave in self._datos

    def __getitem__(self, clave: str):
        """
        Permite acceder a valores como info["brillo"].

        Lanza KeyError si la clave no existe.
        """
        if clave not in self._datos:
            raise KeyError(f"La clave '{clave}' no existe en los metadatos.")
        return self._datos[clave]

    def __setitem__(self, clave: str, valor) -> None:
        """Permite asignar valores como info["brillo"] = 120."""
        self._datos[clave] = valor

    def __str__(self) -> str:
        lineas = ["── Metadatos de la imagen ──"]
        for clave, valor in self._datos.items():
            lineas.append(f"  {clave}: {valor}")
        lineas.append(f"  historial: {len(self._historial)} cambio(s)")
        return "\n".join(lineas)

    def __repr__(self) -> str:
        return f"Info(claves={list(self._datos.keys())})"
