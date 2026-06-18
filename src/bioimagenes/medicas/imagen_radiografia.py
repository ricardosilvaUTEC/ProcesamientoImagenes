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
    conjunto de radiografías en clusters y mostrarlos interactivamente.
    """
 
    def __init__(self, data: np.ndarray, info: Info = None, modalidad: str = "RX"):
        if data.ndim != 2:
            raise ValueError(
                "ImagenRadiografia requiere una imagen 2D (escala de grises)."
            )
        super().__init__(data=data, info=info)
        self._modalidad: str = str(modalidad)
        self._info["tipo_imagen"] = "Radiografía"
        self._info["modalidad"] = self._modalidad
 
    @property
    def modalidad(self) -> str:
        return self._modalidad
 
    def mejorar_contraste(self, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> None:
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
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python es necesario para ecualizar_histograma().")
        img_u8 = self._normalizar_u8()
        self._data = cv2.equalizeHist(img_u8).astype(np.float32)
        self._info["brillo"] = float(np.mean(self._data))
        self._info.historial.modificar_historial("Ecualización de histograma aplicada")
 
    def invertir(self) -> None:
        if self._data.max() <= 1.0:
            self._data = 1.0 - self._data
        else:
            self._data = 255.0 - self._data
        self._data = self._data.astype(np.float32)
        self._info["brillo"] = float(np.mean(self._data))
        self._info.historial.modificar_historial("Inversión de escala de intensidades")
 
    def detectar_bordes(self, metodo: str = "sobel") -> np.ndarray:
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
        H, W = self._data.shape
        self._validar_roi(fila_ini, fila_fin, col_ini, col_fin, H, W)
        recorte = self._data[fila_ini:fila_fin, col_ini:col_fin].copy()
        nueva = ImagenRadiografia(data=recorte, modalidad=self._modalidad)
        nueva.info["cortada"] = True
        self._info.historial.modificar_historial(
            f"ROI seleccionada: filas [{fila_ini}:{fila_fin}], cols [{col_ini}:{col_fin}]"
        )
        return nueva
 
    def normalizar(self) -> None:
        vmin = self._data.min()
        vmax = self._data.max()
        if vmax == vmin:
            raise ValueError("La imagen tiene intensidad constante, no se puede normalizar.")
        self._data = ((self._data - vmin) / (vmax - vmin)).astype(np.float32)
        self._info["brillo"] = float(np.mean(self._data))
        self._info.historial.modificar_historial("Normalización [0, 1] aplicada")
 
    def visualizar(self, titulo: str = "Radiografía") -> None:
        plt.figure(figsize=(6, 6))
        plt.imshow(self._data, cmap="gray")
        plt.title(f"{titulo} — {self._modalidad}")
        plt.axis("off")
        plt.colorbar(label="Intensidad")
        plt.tight_layout()
        plt.show()
 
    def visualizar_cluster(self, imagenes: list, n_clusters: int = 3) -> None:
        """
        Agrupa un conjunto de radiografías en clusters y los visualiza
        en un scatter 2D interactivo.
 
        Al pasar el cursor sobre un punto se muestra una miniatura de
        la radiografía correspondiente en una ventana emergente.
 
        Parámetros
        ----------
        imagenes : list de np.ndarray o list de str
            Lista de imágenes (arrays 2D) o rutas a archivos de imagen.
        n_clusters : int
            Número de clusters a generar. Por defecto 3.
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError(
                "scikit-learn es necesario para visualizar_cluster(). "
                "Instalalo con: pip install scikit-learn"
            )
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python es necesario para visualizar_cluster(). "
                "Instalalo con: pip install opencv-python"
            )
 
        if not imagenes:
            raise ValueError("La lista de imágenes no puede estar vacía.")
        if not isinstance(n_clusters, int) or n_clusters < 2:
            raise ValueError("n_clusters debe ser un entero mayor o igual a 2.")
        if n_clusters > len(imagenes):
            raise ValueError(
                f"n_clusters ({n_clusters}) no puede ser mayor que "
                f"el número de imágenes ({len(imagenes)})."
            )
 
        # 1. Carga y preprocesamiento
        TAM = (64, 64)
        arrays, features = [], []
 
        for item in imagenes:
            if isinstance(item, str):
                img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"  [WARN] No se pudo cargar: {item}")
                    continue
            elif isinstance(item, np.ndarray):
                img = item.astype(np.uint8) if item.max() > 1 else (item * 255).astype(np.uint8)
            else:
                continue
            arrays.append(img)
            features.append(cv2.resize(img, TAM).flatten().astype(np.float32))
 
        if len(arrays) < n_clusters:
            raise ValueError(
                f"Solo se pudieron cargar {len(arrays)} imágenes válidas, "
                f"pero n_clusters={n_clusters}."
            )
 
        X = np.array(features)
 
        # 2. Normalización y PCA → 2D
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
        varianza = pca.explained_variance_ratio_ * 100
 
        # 3. K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        etiquetas = kmeans.fit_predict(X_scaled)
 
        # 4. Visualización interactiva
        colores = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter_pts = []
 
        for cluster_id in range(n_clusters):
            mask = etiquetas == cluster_id
            pts = ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[colores[cluster_id]],
                label=f"Cluster {cluster_id + 1}",
                s=80, alpha=0.8, edgecolors="white", linewidths=0.5,
                marker="s",
            )
            scatter_pts.append((pts, np.where(mask)[0]))
 
        ax.set_xlabel(f"Componente 1 ({varianza[0]:.1f}% varianza)", fontsize=11)
        ax.set_ylabel(f"Componente 2 ({varianza[1]:.1f}% varianza)", fontsize=11)
        ax.set_title(
            f"Image Clusters — {n_clusters} grupos, {len(arrays)} radiografías",
            fontsize=13
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
 
        # Miniatura al hacer hover
        ax_info = fig.add_axes([0.02, 0.02, 0.22, 0.30])
        ax_info.axis("off")
        ax_info.set_visible(False)
        estado = {"visible": False}
 
        def on_hover(event):
            if event.inaxes != ax:
                if estado["visible"]:
                    ax_info.set_visible(False)
                    ax_info.cla()
                    ax_info.axis("off")
                    estado["visible"] = False
                    fig.canvas.draw_idle()
                return
            encontrado = False
            for pts, indices in scatter_pts:
                cont, _ = pts.contains(event)
                if cont:
                    distancias = np.sqrt(
                        (X_2d[indices, 0] - event.xdata) ** 2 +
                        (X_2d[indices, 1] - event.ydata) ** 2
                    )
                    idx_local = np.argmin(distancias)
                    idx_global = indices[idx_local]
                    ax_info.cla()
                    ax_info.imshow(arrays[idx_global], cmap="gray", aspect="auto")
                    ax_info.set_title(
                        f"X: {X_2d[idx_global, 0]:.2f}\nY: {X_2d[idx_global, 1]:.2f}",
                        fontsize=8
                    )
                    ax_info.axis("off")
                    ax_info.set_visible(True)
                    estado["visible"] = True
                    encontrado = True
                    fig.canvas.draw_idle()
                    break
            if not encontrado and estado["visible"]:
                ax_info.set_visible(False)
                ax_info.cla()
                ax_info.axis("off")
                estado["visible"] = False
                fig.canvas.draw_idle()
 
        fig.canvas.mpl_connect("motion_notify_event", on_hover)
        plt.tight_layout()
        plt.show()
 
        self._info.historial.modificar_historial(
            f"visualizar_cluster(): {len(arrays)} imágenes, {n_clusters} clusters"
        )
 
    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------
 
    def _normalizar_u8(self) -> np.ndarray:
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