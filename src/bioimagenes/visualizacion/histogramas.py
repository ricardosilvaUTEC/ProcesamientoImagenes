from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
 
 
def histograma(data: np.ndarray, titulo: str = "Histograma", bins: int = 256) -> None:
    """
    Muestra el histograma de intensidades de una imagen en escala de grises.
 
    Parámetros
    ----------
    data : np.ndarray
        Imagen 2D en escala de grises.
    titulo : str
        Título del gráfico.
    bins : int
        Cantidad de intervalos del histograma.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data debe ser un np.ndarray")
    if data.ndim != 2:
        raise ValueError("histograma() espera una imagen 2D (escala de grises)")
 
    plt.figure(figsize=(7, 4))
    plt.hist(data.ravel(), bins=bins, color="steelblue", edgecolor="none")
    plt.title(titulo)
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()
 
 
def histograma_rgb(
    data: np.ndarray,
    titulo: str = "Histograma RGB",
    bins: int = 256,
) -> None:
    """
    Muestra los histogramas de los tres canales (R, G, B) de una imagen a color.
 
    Parámetros
    ----------
    data : np.ndarray
        Imagen 3D RGB.
    titulo : str
        Título del gráfico.
    bins : int
        Cantidad de intervalos del histograma.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data debe ser un np.ndarray")
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError("histograma_rgb() espera una imagen 3D con 3 canales (RGB)")
 
    colores = ["red", "green", "blue"]
    nombres = ["Canal R", "Canal G", "Canal B"]
 
    plt.figure(figsize=(10, 4))
    for i, (color, nombre) in enumerate(zip(colores, nombres)):
        plt.subplot(1, 3, i + 1)
        plt.hist(data[:, :, i].ravel(), bins=bins, color=color, alpha=0.8, edgecolor="none")
        plt.title(nombre)
        plt.xlabel("Intensidad")
        plt.ylabel("Frecuencia")
 
    plt.suptitle(titulo)
    plt.tight_layout()
    plt.show()
 
 
def histograma_comparacion(
    data_original: np.ndarray,
    data_procesada: np.ndarray,
    titulo: str = "Comparación de histogramas",
    bins: int = 256,
) -> None:
    """
    Muestra los histogramas de dos imágenes lado a lado para comparar
    la distribución de intensidades antes y después de una transformación.
 
    Parámetros
    ----------
    data_original : np.ndarray
        Imagen original (2D).
    data_procesada : np.ndarray
        Imagen procesada (2D).
    titulo : str
        Título del gráfico.
    bins : int
        Cantidad de intervalos del histograma.
    """
    for nombre, data in [("data_original", data_original), ("data_procesada", data_procesada)]:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{nombre} debe ser un np.ndarray")
        if data.ndim != 2:
            raise ValueError(f"{nombre} debe ser una imagen 2D")
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    for ax, data, subtitulo, color in zip(
        axes,
        [data_original, data_procesada],
        ["Original", "Procesada"],
        ["steelblue", "darkorange"],
    ):
        ax.hist(data.ravel(), bins=bins, color=color, edgecolor="none", alpha=0.85)
        ax.set_title(subtitulo)
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Frecuencia")
 
    plt.suptitle(titulo)
    plt.tight_layout()
    plt.show()
 
 
def histograma_acumulado(
    data: np.ndarray,
    titulo: str = "Histograma acumulado",
    bins: int = 256,
) -> None:
    """
    Muestra el histograma acumulado de una imagen 2D.
    Útil para analizar la ecualización de histograma.
 
    Parámetros
    ----------
    data : np.ndarray
        Imagen 2D en escala de grises.
    titulo : str
        Título del gráfico.
    bins : int
        Cantidad de intervalos.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data debe ser un np.ndarray")
    if data.ndim != 2:
        raise ValueError("histograma_acumulado() espera una imagen 2D")
 
    plt.figure(figsize=(7, 4))
    plt.hist(
        data.ravel(),
        bins=bins,
        cumulative=True,
        color="steelblue",
        edgecolor="none",
        density=True,
    )
    plt.title(titulo)
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia acumulada (normalizada)")
    plt.tight_layout()
    plt.show()
 