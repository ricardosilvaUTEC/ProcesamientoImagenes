from datetime import datetime


class Historial:
    """
    Clase para registrar y gestionar la trazabilidad de las operaciones
    realizadas sobre una imagen.

    Permite almacenar de forma cronológica los cambios aplicados,
    incluyendo información relevante como tipo de operación y el momento
    en que fue realizada.

    Es especialmente útil en contextos donde se requiere reproducibilidad
    y seguimiento detallado del procesamiento.
    """

    def __init__(self, lista_cambios: list = None):
        """
        Inicializa una instancia de la clase Historial.

        Parámetros
        ----------
        lista_cambios : list, opcional
            Lista inicial de cambios. Cada cambio es un dict con
            las claves 'operacion' y 'fecha'.
            Si no se proporciona, se inicializa vacía.
        """
        if lista_cambios is not None:
            if not isinstance(lista_cambios, list):
                raise TypeError("lista_cambios debe ser una lista.")
            self._lista_cambios = lista_cambios
        else:
            self._lista_cambios = []

    # ------------------------------------------------------------------
    # Propiedad pública
    # ------------------------------------------------------------------

    @property
    def lista_cambios(self) -> list:
        """Retorna la lista de cambios registrados."""
        return self._lista_cambios

    @property
    def ultimo_cambio(self) -> dict | None:
        """Retorna el último cambio registrado, o None si no hay ninguno."""
        if self._lista_cambios:
            return self._lista_cambios[-1]
        return None

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def modificar_historial(self, entrada: str) -> None:
        """
        Añade una nueva entrada al historial con la operación y la fecha actual.

        Parámetros
        ----------
        entrada : str
            Descripción de la operación realizada.

        Retorna
        -------
        None
        """
        if not isinstance(entrada, str) or not entrada.strip():
            raise ValueError("La entrada debe ser una cadena de texto no vacía.")

        cambio = {
            "operacion": entrada,
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._lista_cambios.append(cambio)

    # ------------------------------------------------------------------
    # Métodos nativos / dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Retorna la cantidad total de cambios registrados."""
        return len(self._lista_cambios)

    def __iter__(self):
        """Permite iterar sobre los cambios del historial."""
        return iter(self._lista_cambios)

    def __str__(self) -> str:
        """Muestra un resumen del historial y el último cambio realizado."""
        total = len(self._lista_cambios)
        if total == 0:
            return "Historial vacío. No se han registrado cambios."
        ultimo = self.ultimo_cambio
        return (
            f"Historial: {total} cambio(s) registrado(s).\n"
            f"Último cambio: [{ultimo['fecha']}] {ultimo['operacion']}"
        )

    def __repr__(self) -> str:
        return f"Historial(cambios={len(self._lista_cambios)})"
