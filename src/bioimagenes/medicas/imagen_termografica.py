import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from bioimagenes.core.imagen import Imagen
from bioimagenes.core.info import Info


class ImagenTermografica(Imagen):
    """
    Clase para el manejo de imágenes termográficas.

    Cada valor de la matriz representa una magnitud térmica. Permite
    interpretar y analizar distribuciones de temperatura, detectar
    zonas calientes o anómalas, generar mapas de calor y segmentar
    regiones por umbral térmico.

    Las imágenes se cargan típicamente en escala de grises (0–255) y
    se convierten a temperatura (°C) mediante un rango definido por el
    usuario, ya que los archivos termográficos no incluyen metadatos
    de temperatura.
    """

    def __init__(
        self,
        data: np.ndarray,
        info: Info = None,
        t_min: float = 30.0,
        t_max: float = 40.0,
    ):
        """
        Inicializa una instancia de ImagenTermografica.

        Parámetros
        ----------
        data : np.ndarray
            Imagen 2D en escala de grises (0–255) o directamente en °C.
        info : Info, opcional
            Metadatos. Se genera uno por defecto si no se proporciona.
        t_min : float
            Temperatura mínima del rango de conversión (°C). Por defecto 30.0.
        t_max : float
            Temperatura máxima del rango de conversión (°C). Por defecto 40.0.

        Raises
        ------
        ValueError
            Si data no es 2D o si t_min >= t_max.
        """
        if data.ndim != 2:
            raise ValueError(
                "ImagenTermografica requiere una imagen 2D."
            )

        self._validar_rango_temperatura(t_min, t_max)

        self._t_min: float = float(t_min)
        self._t_max: float = float(t_max)
        self._en_temperatura: bool = False  # True después de convertir_a_temperatura()

        super().__init__(data=data, info=info)

        self._info["tipo_imagen"] = "Termografía"
        self._info["t_min_°C"] = self._t_min
        self._info["t_max_°C"] = self._t_max

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def t_min(self) -> float:
        """Temperatura mínima del rango de conversión."""
        return self._t_min

    @property
    def t_max(self) -> float:
        """Temperatura máxima del rango de conversión."""
        return self._t_max

    @property
    def en_temperatura(self) -> bool:
        """Indica si los datos ya están en °C."""
        return self._en_temperatura

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def convertir_a_temperatura(self, t_min: float = None, t_max: float = None) -> None:
        """
        Convierte los valores internos (0–255) a grados Celsius.

        temp = t_min + (pixel / 255) × (t_max - t_min)

        Parámetros
        ----------
        t_min : float, opcional
            Temperatura mínima. Si se omite, usa el valor almacenado.
        t_max : float, opcional
            Temperatura máxima. Si se omite, usa el valor almacenado.

        Raises
        ------
        RuntimeError
            Si la imagen ya fue convertida a temperatura.
        """
        if self._en_temperatura:
            raise RuntimeError("Los datos ya están en grados Celsius.")

        if t_min is not None or t_max is not None:
            t_min = float(t_min) if t_min is not None else self._t_min
            t_max = float(t_max) if t_max is not None else self._t_max
            self._validar_rango_temperatura(t_min, t_max)
            self._t_min = t_min
            self._t_max = t_max
            self._info["t_min_°C"] = self._t_min
            self._info["t_max_°C"] = self._t_max

        # Normalizamos a [0, 1] y luego escalamos al rango de temperatura
        img_norm = self._data / 255.0
        self._data = (self._t_min + img_norm * (self._t_max - self._t_min)).astype(np.float32)
        self._en_temperatura = True
        self._info["brillo"] = float(np.mean(self._data))
        self._info.historial.modificar_historial(
            f"Conversión a temperatura: [{self._t_min}°C, {self._t_max}°C]"
        )

    def normalizar(self) -> None:
        """
        Normaliza los valores al rango [0, 1] para facilitar el análisis.
        """
        vmin = self._data.min()
        vmax = self._data.max()
        if vmax == vmin:
            raise ValueError("Imagen con temperatura constante, imposible normalizar.")
        self._data = ((self._data - vmin) / (vmax - vmin)).astype(np.float32)
        self._info["brillo"] = float(np.mean(self._data))
        self._info.historial.modificar_historial("Normalización térmica [0, 1] aplicada")

    def mapa_calor(self, colormap: str = "inferno") -> None:
        """
        Visualiza la imagen como un mapa de calor.

        Parámetros
        ----------
        colormap : str
            Colormap de matplotlib a usar. Por defecto 'inferno'.
            Otros útiles: 'hot', 'plasma', 'RdYlBu_r'.
        """
        unidad = "°C" if self._en_temperatura else "intensidad"
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(self._data, cmap=colormap)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"Temperatura ({unidad})", fontsize=10)
        ax.set_title(f"Mapa de calor — {colormap}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        self._info.historial.modificar_historial(f"Mapa de calor visualizado (cmap={colormap})")

    def detectar_puntos_calientes(self, umbral: float) -> np.ndarray:
        """
        Retorna una máscara booleana de los píxeles que superan el umbral.

        Parámetros
        ----------
        umbral : float
            Valor de temperatura (en la escala actual) por encima del cual
            un píxel se considera "caliente".

        Retorna
        -------
        np.ndarray
            Array 2D booleano: True donde la temperatura > umbral.

        Raises
        ------
        TypeError
            Si umbral no es numérico.
        """
        if not isinstance(umbral, (int, float)):
            raise TypeError("umbral debe ser un valor numérico.")

        mascara = self._data > float(umbral)
        n_calientes = int(np.sum(mascara))
        self._info.historial.modificar_historial(
            f"Detección de puntos calientes: umbral={umbral}, encontrados={n_calientes}"
        )
        return mascara

    def segmentar_por_umbral(self, umbral_inf: float, umbral_sup: float) -> np.ndarray:
        """
        Segmenta la imagen seleccionando los píxeles dentro de un rango térmico.

        Parámetros
        ----------
        umbral_inf : float
            Límite inferior del rango (inclusive).
        umbral_sup : float
            Límite superior del rango (inclusive).

        Retorna
        -------
        np.ndarray
            Array 2D booleano: True donde umbral_inf ≤ valor ≤ umbral_sup.

        Raises
        ------
        ValueError
            Si umbral_inf >= umbral_sup.
        """
        if umbral_inf >= umbral_sup:
            raise ValueError("umbral_inf debe ser menor que umbral_sup.")

        mascara = (self._data >= umbral_inf) & (self._data <= umbral_sup)
        self._info.historial.modificar_historial(
            f"Segmentación térmica: [{umbral_inf}, {umbral_sup}]"
        )
        return mascara

    def visualizar_segmentacion(self, umbral: float, colormap: str = "inferno") -> None:
        """
        Muestra la imagen original y la región segmentada (puntos calientes)
        superpuesta en rojo.

        Parámetros
        ----------
        umbral : float
            Umbral de temperatura para detectar zonas calientes.
        colormap : str
            Colormap para la imagen de fondo.
        """
        mascara = self.detectar_puntos_calientes(umbral)
        unidad = "°C" if self._en_temperatura else "u"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Imagen original como mapa de calor
        im = axes[0].imshow(self._data, cmap=colormap)
        fig.colorbar(im, ax=axes[0], label=f"Temperatura ({unidad})")
        axes[0].set_title("Imagen térmica original")
        axes[0].axis("off")

        # Imagen con zonas calientes superpuestas
        axes[1].imshow(self._data, cmap="gray")
        overlay = np.zeros((*self._data.shape, 4), dtype=np.float32)  # RGBA
        overlay[mascara] = [1.0, 0.0, 0.0, 0.6]  # rojo semitransparente
        axes[1].imshow(overlay)

        parche = mpatches.Patch(color="red", alpha=0.6, label=f"> {umbral} {unidad}")
        axes[1].legend(handles=[parche], loc="lower right", fontsize=9)
        axes[1].set_title(f"Zonas calientes (umbral = {umbral} {unidad})")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
        self._info.historial.modificar_historial(
            f"Segmentación visualizada con umbral={umbral}"
        )

    def visualizar(self, titulo: str = "Termografía") -> None:
        """
        Visualiza la imagen térmica como mapa de calor (inferno).

        Parámetros
        ----------
        titulo : str, opcional
            Título de la figura.
        """
        unidad = "°C" if self._en_temperatura else "intensidad"
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(self._data, cmap="inferno")
        fig.colorbar(im, ax=ax, label=f"Temperatura ({unidad})")
        ax.set_title(titulo)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    @staticmethod
    def _validar_rango_temperatura(t_min: float, t_max: float) -> None:
        if not isinstance(t_min, (int, float)) or not isinstance(t_max, (int, float)):
            raise TypeError("t_min y t_max deben ser numéricos.")
        if t_min >= t_max:
            raise ValueError("t_min debe ser estrictamente menor que t_max.")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        en_temp = "°C" if self._en_temperatura else "escala gris"
        return (
            f"ImagenTermografica | Shape: {self._data.shape} | "
            f"Escala: {en_temp} | Rango: [{self._t_min}, {self._t_max}]°C | "
            f"Media: {np.mean(self._data):.2f}"
        )

    def __repr__(self) -> str:
        return (
            f"ImagenTermografica(shape={self._data.shape}, "
            f"t_min={self._t_min}, t_max={self._t_max}, "
            f"en_temperatura={self._en_temperatura})"
        )