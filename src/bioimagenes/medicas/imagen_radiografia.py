import numpy as np
import matplotlib.pyplot as plt

from bioimagenes.core.imagen import Imagen
from bioimagenes.core.info import Info

class ImagenRadiografia(Imagen):
    """
    Clase para el manejo de imágenes radiográficas (RX).

    Extiende la clase Imagen para trabajar con imágenes bidimensionales
    en escala de grises. Incorpora técnicas específicas del dominio
    radiológico: mejora de contraste, ecualización de histograma,
    inversión de escala, detección de bordes y selección de ROI.

    También provee el método visualizar_cluster() para agrupar un
    conjunto de radiografías en clusters y mostrarlos interactivamente
    (stub — se completa en la siguiente etapa).
    """

    def __init__(self, data: np.ndarray, info: Info = None, modalidad: str = "RX"):
        """
        Inicializa una instancia de ImagenRadiografia.

        Parámetros
        ----------
        data : np.ndarray
            Imagen 2D en escala de grises.
        info : Info, opcional
            Metadatos. Se genera uno por defecto si no se proporciona.
        modalidad : str, opcional
            Descripción del tipo de estudio (ej: "RX", "RX-tórax").

        Raises
        ------
        ValueError
            Si data no es 2D.
        """
        if data.ndim != 2:
            raise ValueError(
                "ImagenRadiografia requiere una imagen 2D (escala de grises)."
            )

        super().__init__(data=data, info=info)

        self._modalidad: str = str(modalidad)
        self._info["tipo_imagen"] = "Radiografía"
        self._info["modalidad"] = self._modalidad

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def modalidad(self) -> str:
        """Tipo de estudio radiográfico."""
        return self._modalidad

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def mejorar_contraste(self, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> None:
        """
        Mejora el contraste de la imagen usando CLAHE (Contrast Limited
        Adaptive Histogram Equalization) mediante OpenCV.

        Parámetros
        ----------
        clip_limit : float
            Límite de recorte del contraste. Por defecto 2.0.
        tile_grid : tuple
            Tamaño de la cuadrícula de tiles (filas, cols). Por defecto (8, 8).
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python es necesario para mejorar_contraste().")

        img_u8 = self._normalizar_u8()
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        resultado = clahe.apply(img_u8)
        self._data = resultado.astype(np.float32)
        self._info["brillo"] = float(np.mean(self._data))
        self._info.historial.modificar_historial(
            f"CLAHE aplicado: clip_limit={clip_limit}, tile={tile_grid}"
        )

    def ecualizar_histograma(self) -> None:
        """
        Ecualiza el histograma de la imagen para mejorar la distribución
        de intensidades globalmente (ecualización estándar).
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python es necesario para ecualizar_histograma().")

        img_u8 = self._normalizar_u8()
        self._data = cv2.equalizeHist(img_u8).astype(np.float32)
        self._info["brillo"] = float(np.mean(self._data))
        self._info.historial.modificar_historial("Ecualización de histograma aplicada")

    def invertir(self) -> None:
        """
        Invierte la escala de intensidades de la imagen.

        Para imágenes en [0, 255]: nuevo_valor = 255 - valor.
        Para imágenes en [0, 1]:   nuevo_valor = 1 - valor.
        """
        if self._data.max() <= 1.0:
            self._data = 1.0 - self._data
        else:
            self._data = 255.0 - self._data

        self._data = self._data.astype(np.float32)
        self._info["brillo"] = float(np.mean(self._data))
        self._info.historial.modificar_historial("Inversión de escala de intensidades")

    def detectar_bordes(self, metodo: str = "sobel") -> np.ndarray:
        """
        Detecta bordes en la imagen usando distintos métodos.

        Parámetros
        ----------
        metodo : str
            Método de detección: 'sobel', 'canny', 'laplacian'.
            Por defecto 'sobel'.

        Retorna
        -------
        np.ndarray
            Imagen con los bordes detectados (no modifica self._data).

        Raises
        ------
        ValueError
            Si el método no está soportado.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python es necesario para detectar_bordes().")

        img_u8 = self._normalizar_u8()
        metodo = metodo.lower()

        if metodo == "sobel":
            sx = cv2.Sobel(img_u8, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(img_u8, cv2.CV_64F, 0, 1, ksize=3)
            resultado = np.sqrt(sx**2 + sy**2)
            resultado = np.clip(resultado / resultado.max() * 255, 0, 255).astype(np.uint8)

        elif metodo == "canny":
            resultado = cv2.Canny(img_u8, threshold1=50, threshold2=150)

        elif metodo == "laplacian":
            lap = cv2.Laplacian(img_u8, cv2.CV_64F)
            resultado = np.clip(np.abs(lap) / np.abs(lap).max() * 255, 0, 255).astype(np.uint8)

        else:
            raise ValueError(f"Método '{metodo}' no soportado. Opciones: sobel, canny, laplacian")

        self._info.historial.modificar_historial(f"Detección de bordes: {metodo}")
        return resultado

    def seleccionar_roi(self, fila_ini: int, fila_fin: int, col_ini: int, col_fin: int) -> "ImagenRadiografia":
        """
        Recorta una región de interés (ROI) de la imagen.

        Parámetros
        ----------
        fila_ini, fila_fin : int
            Rango de filas [fila_ini, fila_fin).
        col_ini, col_fin : int
            Rango de columnas [col_ini, col_fin).

        Retorna
        -------
        ImagenRadiografia
            Nueva instancia con la región recortada.

        Raises
        ------
        ValueError
            Si las coordenadas son inválidas.
        """
        H, W = self._data.shape
        self._validar_roi(fila_ini, fila_fin, col_ini, col_fin, H, W)

        recorte = self._data[fila_ini:fila_fin, col_ini:col_fin].copy()
        nueva = ImagenRadiografia(data=recorte.astype(np.uint8), modalidad=self._modalidad)
        nueva.info["cortada"] = True

        self._info.historial.modificar_historial(
            f"ROI seleccionada: filas [{fila_ini}:{fila_fin}], cols [{col_ini}:{col_fin}]"
        )
        return nueva

    def normalizar(self) -> None:
        """
        Normaliza los valores de la imagen al rango [0, 1].
        """
        vmin = self._data.min()
        vmax = self._data.max()
        if vmax == vmin:
            raise ValueError("La imagen tiene intensidad constante, no se puede normalizar.")
        self._data = ((self._data - vmin) / (vmax - vmin)).astype(np.float32)
        self._info["brillo"] = float(np.mean(self._data))
        self._info.historial.modificar_historial("Normalización [0, 1] aplicada")

    def visualizar(self, titulo: str = "Radiografía") -> None:
        """
        Muestra la radiografía en escala de grises con matplotlib.

        Parámetros
        ----------
        titulo : str, opcional
            Título de la figura.
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(self._data, cmap="gray")
        plt.title(f"{titulo} — {self._modalidad}")
        plt.axis("off")
        plt.colorbar(label="Intensidad")
        plt.tight_layout()
        plt.show()

    def visualizar_cluster(self, imagenes: list = None, n_clusters: int = 3) -> None:
        """
        Agrupa un conjunto de radiografías en clusters y los visualiza
        en un scatter 2D interactivo.

        Al pasar el cursor sobre un punto, se muestra una miniatura de
        la radiografía correspondiente en una ventana emergente.

        Parámetros
        ----------
        imagenes : list de np.ndarray
            Lista de imágenes (arrays 2D) a agrupar.
        n_clusters : int
            Número de clusters a generar.

        Nota
        ----
        STUB — implementación pendiente para la Etapa 3.
        Requiere: scikit-learn, scikit-image.
        """
        raise NotImplementedError(
            "visualizar_cluster() será implementado en la Etapa 3.\n"
            "Requerirá: sklearn.cluster.KMeans, sklearn.decomposition.PCA "
            "y matplotlib con eventos de hover."
        )

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _normalizar_u8(self) -> np.ndarray:
        """Convierte self._data a uint8 en [0, 255] para operaciones cv2."""
        datos = self._data.copy()
        vmin, vmax = datos.min(), datos.max()
        if vmax == vmin:
            return np.zeros_like(datos, dtype=np.uint8)
        normalizado = (datos - vmin) / (vmax - vmin) * 255.0
        return normalizado.astype(np.uint8)

    @staticmethod
    def _validar_roi(fi: int, ff: int, ci: int, cf: int, H: int, W: int) -> None:
        for v in (fi, ff, ci, cf):
            if not isinstance(v, int):
                raise TypeError("Las coordenadas de ROI deben ser enteros.")
        if not (0 <= fi < ff <= H):
            raise ValueError(f"Filas inválidas: [{fi}:{ff}] para altura {H}.")
        if not (0 <= ci < cf <= W):
            raise ValueError(f"Columnas inválidas: [{ci}:{cf}] para ancho {W}.")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"ImagenRadiografia | Shape: {self._data.shape} | "
            f"Modalidad: {self._modalidad} | Brillo: {np.mean(self._data):.2f}"
        )

    def __repr__(self) -> str:
        return f"ImagenRadiografia(shape={self._data.shape}, modalidad='{self._modalidad}')"
