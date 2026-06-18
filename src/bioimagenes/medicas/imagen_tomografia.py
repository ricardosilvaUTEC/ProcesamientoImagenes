import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from bioimagenes.core.imagen import Imagen
from bioimagenes.core.info import Info


class ImagenTomografia(Imagen):
    """
    Clase para el manejo de imágenes tomográficas (CT).

    Extiende la clase Imagen para trabajar con volúmenes 3D compuestos
    por cortes (slices) apilados. Incorpora soporte para unidades
    Hounsfield (HU), ventanas de visualización médica, segmentación
    de tejidos por color y reconstrucción de vistas multiplanares.

    El atributo `data` puede ser:
      - 3D (n_slices, filas, columnas): volumen CT completo.
      - 2D (filas, columnas): un único corte.
    """

    # ------------------------------------------------------------------
    # Ventanas de visualización médica estándar (center, width) en HU
    # ------------------------------------------------------------------
    VENTANAS = {
        "pulmón":    (-600, 1500),
        "hueso":     (400,  1800),
        "tejido":    (40,   400),
        "cerebro":   (40,   80),
        "abdomen":   (60,   400),
    }

    # ------------------------------------------------------------------
    # Rangos HU aproximados por tejido → color RGB para visualizar_corte
    # ------------------------------------------------------------------
    TEJIDOS = {
        "aire":         ((-1100, -950), (0,   0,   0)),    # negro
        "pulmón":       ((-950,  -200), (100, 149, 237)),  # azul aciano
        "grasa":        ((-200,  -10),  (255, 215,  0)),   # amarillo
        "tejido_bland": ((-10,    80),  (220, 120,  80)),  # salmón
        "sangre":       ((80,    200),  (220,  20,  60)),  # rojo
        "hueso":        ((200,  3000),  (255, 255, 255)),  # blanco
    }

    def __init__(
        self,
        data: np.ndarray,
        info: Info = None,
        espaciado: tuple = (1.0, 1.0, 1.0),
    ):
        """
        Inicializa una instancia de ImagenTomografia.

        Parámetros
        ----------
        data : np.ndarray
            Volumen 3D (n_slices, H, W) o corte 2D (H, W).
            Los valores deben estar en unidades Hounsfield (HU) o sin
            escalar (se pueden convertir con convertir_a_hu()).
        info : Info, opcional
            Metadatos de la imagen. Se genera uno por defecto si no se
            proporciona.
        espaciado : tuple, opcional
            Tamaño físico de cada voxel en mm: (dz, dy, dx).
            Por defecto (1.0, 1.0, 1.0).

        Raises
        ------
        ValueError
            Si data no es 2D ni 3D.
        TypeError
            Si espaciado no es una tupla de 3 números positivos.
        """
        if data.ndim not in (2, 3):
            raise ValueError("data debe ser un array 2D o 3D.")

        self._espaciado = self._validar_espaciado(espaciado)
        self._en_hu: bool = False  # indica si ya se aplicó conversión HU

        # Llamamos al constructor padre (que valida y guarda data e info)
        super().__init__(data=data, info=info)

        # Actualizamos metadatos específicos de tomografía
        self._info["tipo_imagen"] = "Tomografía"
        self._info["espaciado_mm"] = self._espaciado
        self._info["n_slices"] = self._data.shape[0] if self._data.ndim == 3 else 1

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def espaciado(self) -> tuple:
        """Tamaño físico del voxel en mm (dz, dy, dx)."""
        return self._espaciado

    @property
    def n_slices(self) -> int:
        """Número de cortes del volumen."""
        return self._data.shape[0] if self._data.ndim == 3 else 1

    @property
    def en_hu(self) -> bool:
        """Indica si los datos están en unidades Hounsfield."""
        return self._en_hu
        
    def bn(self) -> None:

        """
        Operación no disponible para imágenes tomográficas.

        En la clase base, bn() convierte una imagen RGB a escala de grises.
        En una tomografía, el array 3D representa un volumen (slices, H, W)
        y no canales de color, por lo que esta operación no tiene sentido.

        Raises
        ------
        NotImplementedError
        Siempre. Usar normalizar() o ajustar_ventana() en su lugar.
        """
        raise NotImplementedError(
            "bn() no está disponible en ImagenTomografia. "
            "El array 3D representa un volumen, no canales RGB. "
            "Usá normalizar() o ajustar_ventana() para manipular intensidades."
        )

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def convertir_a_hu(self, slope: float = 1.0, intercept: float = -1024.0) -> None:
        """
        Convierte los valores del volumen a unidades Hounsfield (HU).

        HU = pixel_value × slope + intercept

        Parámetros
        ----------
        slope : float
            Factor de escala (RescaleSlope en DICOM). Por defecto 1.0.
        intercept : float
            Desplazamiento (RescaleIntercept en DICOM). Por defecto -1024.
        """
        if self._en_hu:
            raise RuntimeError("Los datos ya están en unidades Hounsfield.")

        if not isinstance(slope, (int, float)) or not isinstance(intercept, (int, float)):
            raise TypeError("slope e intercept deben ser numéricos.")

        self._data = self._data * float(slope) + float(intercept)
        self._en_hu = True
        self._info.historial.modificar_historial(
            f"Conversión a HU: slope={slope}, intercept={intercept}"
        )

    def normalizar(self, rango_min: float = None, rango_max: float = None) -> None:
        """
        Normaliza las intensidades al rango [0, 1].

        Parámetros
        ----------
        rango_min : float, opcional
            Valor mínimo para la normalización. Por defecto usa el mínimo del volumen.
        rango_max : float, opcional
            Valor máximo para la normalización. Por defecto usa el máximo del volumen.
        """
        vmin = rango_min if rango_min is not None else float(self._data.min())
        vmax = rango_max if rango_max is not None else float(self._data.max())

        if vmax == vmin:
            raise ValueError("rango_min y rango_max no pueden ser iguales.")

        self._data = np.clip((self._data - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)
        self._info.historial.modificar_historial(
            f"Normalización: [{vmin:.1f}, {vmax:.1f}] → [0, 1]"
        )

    def cargar_volumen(self, slices: np.ndarray) -> None:
        """
        Carga o reemplaza el volumen con un nuevo array de cortes.

        Parámetros
        ----------
        slices : np.ndarray
            Array 3D (n_slices, H, W).
        """
        if not isinstance(slices, np.ndarray) or slices.ndim != 3:
            raise ValueError("slices debe ser un np.ndarray 3D.")

        self._data = slices.astype(np.float32)
        self._info["dimensiones"] = self._data.shape
        self._info["n_slices"] = self._data.shape[0]
        self._info.historial.modificar_historial(
            f"Volumen cargado: {self._data.shape}"
        )

    def get_slice(self, indice: int) -> np.ndarray:
        """
        Retorna un corte axial del volumen.

        Parámetros
        ----------
        indice : int
            Índice del corte (0 … n_slices-1).

        Retorna
        -------
        np.ndarray
            Array 2D del corte.
        """
        if self._data.ndim == 2:
            return self._data

        self._validar_indice(indice)
        return self._data[indice]

    def visualizar_corte(self, indice: int, mostrar_referencia: bool = True) -> None:
        """
        Muestra un corte del volumen coloreado por tipo de tejido.

        Genera una imagen RGB del mismo tamaño que el corte, asignando
        un color a cada rango de Hounsfield definido en TEJIDOS.

        Parámetros
        ----------
        indice : int
            Índice del corte axial a visualizar.
        mostrar_referencia : bool
            Si True, muestra una leyenda con los colores de tejido.
        """
        corte = self.get_slice(indice)
        rgb = self._segmentar_tejidos(corte)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Corte original en grises
        axes[0].imshow(corte, cmap="gray")
        axes[0].set_title(f"Corte axial #{indice} — original")
        axes[0].axis("off")

        # Corte segmentado por tejidos
        axes[1].imshow(rgb)
        axes[1].set_title(f"Corte axial #{indice} — tejidos")
        axes[1].axis("off")

        if mostrar_referencia:
            parches = [
                mpatches.Patch(
                    color=np.array(color) / 255.0,
                    label=tejido.replace("_", " ").capitalize(),
                )
                for tejido, (_, color) in self.TEJIDOS.items()
            ]
            axes[1].legend(
                handles=parches,
                loc="lower right",
                fontsize=8,
                framealpha=0.8,
            )

        plt.tight_layout()
        plt.show()

        self._info.historial.modificar_historial(
            f"Visualización de corte #{indice} con segmentación de tejidos"
        )

    def ajustar_ventana(self, center: float, width: float) -> np.ndarray:
        """
        Aplica una ventana de visualización médica al volumen.

        Recorta los valores HU al rango [center - width/2, center + width/2]
        y normaliza a [0, 255].

        Parámetros
        ----------
        center : float
            Centro de la ventana en HU.
        width : float
            Ancho de la ventana en HU.

        Retorna
        -------
        np.ndarray
            Volumen/corte con ventana aplicada (float32, rango [0, 255]).
        """
        if width <= 0:
            raise ValueError("width debe ser positivo.")

        vmin = center - width / 2
        vmax = center + width / 2
        ventaneado = np.clip(self._data, vmin, vmax)
        resultado = ((ventaneado - vmin) / (vmax - vmin) * 255.0).astype(np.float32)

        self._info.historial.modificar_historial(
            f"Ventana aplicada: center={center}, width={width}"
        )
        return resultado

    def aplicar_ventana_predefinida(self, nombre: str) -> np.ndarray:
        """
        Aplica una ventana de visualización predefinida por nombre.

        Ventanas disponibles: 'pulmón', 'hueso', 'tejido', 'cerebro', 'abdomen'.

        Parámetros
        ----------
        nombre : str
            Nombre de la ventana predefinida.

        Retorna
        -------
        np.ndarray
            Volumen ventaneado.
        """
        nombre = nombre.lower()
        if nombre not in self.VENTANAS:
            disponibles = ", ".join(self.VENTANAS.keys())
            raise ValueError(
                f"Ventana '{nombre}' no reconocida. Disponibles: {disponibles}"
            )

        center, width = self.VENTANAS[nombre]
        return self.ajustar_ventana(center, width)

    def visualizar(self, titulo: str = "Tomografía") -> None:
        """
        Visualiza el corte central del volumen (o la imagen si es 2D).

        Parámetros
        ----------
        titulo : str, opcional
            Título de la figura.
        """
        if self._data.ndim == 3:
            corte_central = self._data.shape[0] // 2
            datos_vis = self._data[corte_central]
            titulo = f"{titulo} — corte {corte_central}/{self.n_slices - 1}"
        else:
            datos_vis = self._data

        plt.figure(figsize=(6, 6))
        plt.imshow(datos_vis, cmap="gray")
        plt.title(titulo)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _segmentar_tejidos(self, corte: np.ndarray) -> np.ndarray:
        """
        Genera una imagen RGB coloreando cada píxel según su rango HU.

        Parámetros
        ----------
        corte : np.ndarray
            Array 2D con valores en HU.

        Retorna
        -------
        np.ndarray
            Array (H, W, 3) uint8 con colores por tejido.
        """
        rgb = np.zeros((*corte.shape, 3), dtype=np.uint8)

        for _, (rango, color) in self.TEJIDOS.items():
            mascara = (corte >= rango[0]) & (corte < rango[1])
            rgb[mascara] = color

        return rgb

    @staticmethod
    def _validar_espaciado(espaciado: tuple) -> tuple:
        if not isinstance(espaciado, tuple) or len(espaciado) != 3:
            raise TypeError("espaciado debe ser una tupla de 3 elementos.")
        if not all(isinstance(v, (int, float)) and v > 0 for v in espaciado):
            raise ValueError("Todos los valores de espaciado deben ser números positivos.")
        return espaciado

    def _validar_indice(self, indice: int) -> None:
        if not isinstance(indice, int):
            raise TypeError("El índice debe ser un entero.")
        n = self._data.shape[0]
        if not (0 <= indice < n):
            raise IndexError(f"Índice {indice} fuera de rango (0 … {n - 1}).")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        tipo = "3D" if self._data.ndim == 3 else "2D"
        return (
            f"ImagenTomografia | Shape: {self._data.shape} | {tipo} | "
            f"Slices: {self.n_slices} | HU: {self._en_hu} | "
            f"Espaciado: {self._espaciado} mm"
        )

    def __repr__(self) -> str:
        return f"ImagenTomografia(shape={self._data.shape}, en_hu={self._en_hu})"