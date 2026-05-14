import numpy as np
from scipy.ndimage import convolve


class Filtro:
    """
    Clase para definir y aplicar operadores de filtrado sobre imágenes.

    Representa filtros espaciales mediante kernels que permiten modificar
    la imagen, como suavizado, detección de bordes o realce de detalles.
    Generalmente se aplican mediante operaciones de convolución.

    Está diseñada para ser reutilizable y extensible a distintos tipos
    de filtros utilizados en procesamiento de imágenes.
    """

    def __init__(self, tipo: str, kernel: np.ndarray):
        """
        Inicializa una instancia de la clase Filtro.

        Parámetros
        ----------
        tipo : str
            Nombre o identificador del filtro (ej: "Gaussiano", "Sobel").
        kernel : np.ndarray
            Matriz 2D que define los pesos del filtro para la operación
            de convolución.

        Raises
        ------
        TypeError
            Si tipo no es str o kernel no es np.ndarray.
        ValueError
            Si el kernel no es 2D o está vacío.
        """
        if not isinstance(tipo, str) or not tipo.strip():
            raise TypeError("tipo debe ser una cadena de texto no vacía.")

        if not isinstance(kernel, np.ndarray):
            raise TypeError("kernel debe ser un np.ndarray.")

        if kernel.ndim != 2:
            raise ValueError("El kernel debe ser una matriz 2D.")

        if kernel.size == 0:
            raise ValueError("El kernel no puede estar vacío.")

        self._tipo: str = tipo
        self._kernel: np.ndarray = kernel.astype(np.float32)

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def tipo(self) -> str:
        """Identificador del filtro."""
        return self._tipo

    @property
    def kernel(self) -> np.ndarray:
        """Matriz de pesos del filtro."""
        return self._kernel

    @property
    def tamaño(self) -> tuple:
        """Dimensiones del kernel (filas, columnas)."""
        return self._kernel.shape

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def convolucion(self, imagen: np.ndarray) -> np.ndarray:
        """
        Ejecuta la operación de convolución del kernel sobre una imagen.

        Soporta imágenes 2D (grises) y 3D (RGB, aplica canal por canal).

        Parámetros
        ----------
        imagen : np.ndarray
            Array 2D o 3D de la imagen sobre la que aplicar la convolución.

        Retorna
        -------
        np.ndarray
            Imagen convolucionada con el mismo shape que la entrada.

        Raises
        ------
        TypeError
            Si imagen no es np.ndarray.
        ValueError
            Si imagen no es 2D ni 3D.
        """
        if not isinstance(imagen, np.ndarray):
            raise TypeError("imagen debe ser un np.ndarray.")

        if imagen.ndim not in (2, 3):
            raise ValueError("La imagen debe ser 2D o 3D.")

        img_f = imagen.astype(np.float32)

        if img_f.ndim == 2:
            resultado = convolve(img_f, self._kernel)
        else:
            # Aplicamos canal por canal (R, G, B)
            canales = [convolve(img_f[:, :, c], self._kernel) for c in range(img_f.shape[2])]
            resultado = np.stack(canales, axis=2)

        return resultado

    def aplicar(self, imagen: np.ndarray) -> np.ndarray:
        """
        Aplica el filtro a una imagen (alias semántico de convolucion).

        Parámetros
        ----------
        imagen : np.ndarray
            Imagen sobre la que aplicar el filtro.

        Retorna
        -------
        np.ndarray
            Imagen filtrada.
        """
        return self.convolucion(imagen)

    # ------------------------------------------------------------------
    # Métodos nativos / dunder
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Imprime el nombre del filtro y el tamaño del kernel."""
        return f"Filtro '{self._tipo}' — kernel: {self.tamaño[0]}×{self.tamaño[1]}"

    def __repr__(self) -> str:
        return f"Filtro(tipo='{self._tipo}', tamaño={self.tamaño})"
