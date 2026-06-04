import numpy as np
import matplotlib.pyplot as plt

from bioimagenes.core.info import Info
from bioimagenes.core.historial import Historial
from bioimagenes.filtros.filtro import Filtro
from bioimagenes.visualizacion.visualizar import mostrar_comparacion
from bioimagenes.visualizacion.histogramas import histograma, histograma_rgb

class Imagen:
    """
    Clase base para el manejo y procesamiento de imágenes digitales.

    Representa una imagen como una matriz de datos y proporciona
    herramientas para su manipulación, visualización y análisis.
    Permite aplicar operaciones como filtrado, conversión a escala de
    grises y visualización.

    Integra metadatos mediante la clase Info y mantiene un registro de
    cambios a través de la clase Historial.
    """

    def __init__(self, data: np.ndarray, info: Info = None):
        """
        Inicializa una instancia de la clase Imagen.

        Parámetros
        ----------
        data : np.ndarray
            Matriz que contiene los valores de los píxeles.
            Puede ser 2D (escala de grises) o 3D (RGB).
        info : Info, opcional
            Objeto con los metadatos de la imagen.
            Si no se proporciona, se genera uno por defecto.

        Raises
        ------
        TypeError
            Si data no es un np.ndarray o info no es una instancia de Info.
        ValueError
            Si data no tiene 2 o 3 dimensiones.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data debe ser un np.ndarray.")

        if data.ndim not in (2, 3):
            raise ValueError("La imagen debe ser 2D (grises) o 3D (RGB).")

        if info is not None and not isinstance(info, Info):
            raise TypeError("info debe ser una instancia de Info.")

        # Convertimos a float32 para facilitar cálculos
        self._data: np.ndarray = data.astype(np.float32)

        if info is None:
            historial = Historial()
            datos = {
                "dimensiones": data.shape,
                "brillo": float(np.mean(data)),
                "cortada": False,
            }
            self._info = Info(datos=datos, historial=historial)
        else:
            self._info = info

    # ------------------------------------------------------------------
    # Propiedades públicas
    # ------------------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """Matriz de datos de la imagen."""
        return self._data

    @data.setter
    def data(self, valor: np.ndarray) -> None:
        if not isinstance(valor, np.ndarray):
            raise TypeError("data debe ser un np.ndarray.")
        if valor.ndim not in (2, 3):
            raise ValueError("La imagen debe ser 2D o 3D.")
        self._data = valor.astype(np.float32)

    @property
    def info(self) -> Info:
        """Metadatos de la imagen."""
        return self._info

    # ------------------------------------------------------------------
    # Métodos principales
    # ------------------------------------------------------------------

    def aplicar_filtro(self, filtro: Filtro) -> None:
        """
        Aplica un objeto Filtro sobre la imagen mediante convolución.

        Parámetros
        ----------
        filtro : Filtro
            Objeto Filtro a aplicar.

        Raises
        ------
        TypeError
            Si filtro no es una instancia de Filtro.
        """
        if not isinstance(filtro, Filtro):
            raise TypeError("Debe pasar un objeto de tipo Filtro.")

        self._data = filtro.aplicar(self._data)

        # Actualizamos metadatos
        self._info["brillo"] = float(np.mean(self._data))
        self._info.historial.modificar_historial(f"Filtro aplicado: {filtro.tipo}")

    def bn(self) -> None:
        """
        Convierte la imagen de RGB a escala de grises (blanco y negro).

        Si la imagen ya es 2D (grises), no realiza ninguna operación.
        La conversión usa los pesos estándar: 0.2989·R + 0.5870·G + 0.1140·B,
        que reflejan la sensibilidad del ojo humano a cada canal.
        """
        if self._data.ndim == 2:
            return  # Ya está en grises

        # Pesos perceptuales estándar (ITU-R BT.601)
        self._data = (
            0.2989 * self._data[:, :, 0]
            + 0.5870 * self._data[:, :, 1]
            + 0.1140 * self._data[:, :, 2]
        ).astype(np.float32)

        self._info["dimensiones"] = self._data.shape
        self._info["brillo"] = float(np.mean(self._data))
        self._info.historial.modificar_historial("Conversión a blanco y negro (BN)")

    def visualizar(self, titulo: str = "Imagen") -> None:
        """
        Muestra la imagen utilizando matplotlib.

        Parámetros
        ----------
        titulo : str, opcional
            Título de la figura. Por defecto "Imagen".
        """
        plt.figure(figsize=(6, 6))

        if self._data.ndim == 2:
            plt.imshow(self._data, cmap="gray")
        else:
            # Normalizamos al rango [0, 1] para imshow si los valores superan 1
            img_vis = self._data
            if img_vis.max() > 1.0:
                img_vis = img_vis / 255.0
            plt.imshow(np.clip(img_vis, 0, 1))

        plt.title(titulo)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    def comparar(self, otra: "Imagen", titulo_self: str = "Original",
                 titulo_otra: str = "Procesada") -> None:
        """
        Muestra esta imagen y otra lado a lado para comparación.

        Parámetros
        ----------
        otra : Imagen
            Segunda imagen a comparar.
        titulo_self : str
            Título de esta imagen (izquierda).
        titulo_otra : str
            Título de la otra imagen (derecha).
        """
        if not isinstance(otra, Imagen):
            raise TypeError("otra debe ser una instancia de Imagen.")
        mostrar_comparacion(self._data, otra._data, titulo_self, titulo_otra)

    def histograma(self, bins: int = 256) -> None:
        """
        Muestra el histograma de intensidades de la imagen.
        Si es RGB muestra los tres canales por separado.

        Parámetros
        ----------
        bins : int
            Cantidad de intervalos del histograma.
        """
        if self._data.ndim == 2:
            histograma(self._data, titulo="Histograma", bins=bins)
        else:
            histograma_rgb(self._data, titulo="Histograma RGB", bins=bins)

    # ------------------------------------------------------------------
    # Métodos nativos / dunder
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """
        Devuelve un resumen técnico de la imagen.

        Ejemplo
        -------
        Imagen | Shape: (100, 100) | Tipo: Grises | Píxeles: 10000 | Brillo: 127.5
        """
        tipo = "RGB" if self._data.ndim == 3 else "Grises"
        return (
            f"Imagen | Shape: {self._data.shape} | Tipo: {tipo} | "
            f"Píxeles: {self._data.size} | Brillo: {np.mean(self._data):.2f}"
        )

    def __repr__(self) -> str:
        return f"Imagen(shape={self._data.shape}, dtype={self._data.dtype})"

    def __len__(self) -> int:
        """Retorna el número total de píxeles de la imagen."""
        return self._data.size

    def __getitem__(self, indice):
        """
        Permite acceder a píxeles mediante índices estilo numpy.

        Ejemplos
        --------
        img[50, 50]       → valor del píxel en fila 50, columna 50
        img[10:20, 10:20] → región de la imagen
        """
        return self._data[indice]
