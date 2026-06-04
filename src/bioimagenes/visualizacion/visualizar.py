from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
 
def mostrar_imagen(data: np.ndarray, titulo: str = "Imagen") -> None:
    """
    Muestra una imagen 2D (escala de grises) o 3D (RGB).
 
    Parámetros
    ----------
    data : np.ndarray
        Matriz de la imagen. Puede ser 2D o 3D.
    titulo : str
        Título que aparece sobre la imagen.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data debe ser un np.ndarray")
    if data.ndim not in [2, 3]:
        raise ValueError("data debe ser 2D o 3D")
 
    plt.figure(figsize=(6, 6))
 
    if data.ndim == 2:
        plt.imshow(data, cmap="gray")
    else:
        plt.imshow(data.astype(np.uint8))
 
    plt.title(titulo)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
 
 
def mostrar_comparacion(
    data_original: np.ndarray,
    data_procesada: np.ndarray,
    titulo_original: str = "Original",
    titulo_procesada: str = "Procesada",
) -> None:
    """
    Muestra dos imágenes lado a lado para comparar el antes y después
    de una transformación.
 
    Parámetros
    ----------
    data_original : np.ndarray
        Imagen original.
    data_procesada : np.ndarray
        Imagen luego de aplicar alguna transformación.
    titulo_original : str
        Título de la imagen izquierda.
    titulo_procesada : str
        Título de la imagen derecha.
    """
    for nombre, data in [("data_original", data_original), ("data_procesada", data_procesada)]:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{nombre} debe ser un np.ndarray")
        if data.ndim not in [2, 3]:
            raise ValueError(f"{nombre} debe ser 2D o 3D")
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    for ax, data, titulo in zip(
        axes,
        [data_original, data_procesada],
        [titulo_original, titulo_procesada],
    ):
        if data.ndim == 2:
            ax.imshow(data, cmap="gray")
        else:
            ax.imshow(data.astype(np.uint8))
        ax.set_title(titulo)
        ax.axis("off")
 
    plt.tight_layout()
    plt.show()
 
 
def mostrar_corte(
    data: np.ndarray,
    indice: int,
    eje: str = "axial",
    titulo: str = None,
) -> None:
    """
    Muestra un corte (slice) de un volumen 3D.
    Útil para imágenes de tomografía.
 
    Parámetros
    ----------
    data : np.ndarray
        Volumen 3D de forma (Z, Y, X) o (X, Y, Z).
    indice : int
        Índice del corte a visualizar.
    eje : str
        Eje de corte: "axial", "coronal" o "sagital".
    titulo : str, opcional
        Título del gráfico. Si no se pasa, se genera automáticamente.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data debe ser un np.ndarray")
    if data.ndim != 3:
        raise ValueError("data debe ser un volumen 3D")
 
    ejes_validos = {"axial", "coronal", "sagital"}
    if eje not in ejes_validos:
        raise ValueError(f"eje debe ser uno de {ejes_validos}")
 
    if eje == "axial":
        if indice < 0 or indice >= data.shape[2]:
            raise IndexError(f"indice {indice} fuera de rango para eje axial (0-{data.shape[2]-1})")
        corte = data[:, :, indice]
    elif eje == "coronal":
        if indice < 0 or indice >= data.shape[1]:
            raise IndexError(f"indice {indice} fuera de rango para eje coronal (0-{data.shape[1]-1})")
        corte = data[:, indice, :]
    else:  # sagital
        if indice < 0 or indice >= data.shape[0]:
            raise IndexError(f"indice {indice} fuera de rango para eje sagital (0-{data.shape[0]-1})")
        corte = data[indice, :, :]
 
    titulo_final = titulo if titulo else f"Corte {eje} — índice {indice}"
 
    plt.figure(figsize=(6, 6))
    plt.imshow(corte, cmap="gray")
    plt.title(titulo_final)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
 
 
def mostrar_mapa_calor(
    data: np.ndarray,
    titulo: str = "Mapa de calor",
    cmap: str = "hot",
    unidad: str = "",
) -> None:
    """
    Muestra una imagen como mapa de calor con barra de color.
    Útil para imágenes termográficas.
 
    Parámetros
    ----------
    data : np.ndarray
        Matriz 2D con los valores a visualizar.
    titulo : str
        Título del gráfico.
    cmap : str
        Colormap de matplotlib (por defecto "hot").
    unidad : str
        Etiqueta de la barra de color (ej: "°C").
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data debe ser un np.ndarray")
    if data.ndim != 2:
        raise ValueError("data debe ser una matriz 2D")
 
    plt.figure(figsize=(7, 6))
    im = plt.imshow(data, cmap=cmap)
    cbar = plt.colorbar(im)
    if unidad:
        cbar.set_label(unidad)
    plt.title(titulo)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
 
 
def mostrar_imagen_con_leyenda(
    data: np.ndarray,
    leyenda: dict,
    titulo: str = "Imagen segmentada",
) -> None:
    """
    Muestra una imagen RGB con una leyenda de colores.
    Útil para visualizar segmentaciones de tejidos en tomografías.
 
    Parámetros
    ----------
    data : np.ndarray
        Imagen RGB (3D).
    leyenda : dict
        Diccionario {etiqueta: (R, G, B)} con los colores y sus nombres.
        Ejemplo: {"Hueso": (255, 255, 255), "Grasa": (255, 255, 0)}
    titulo : str
        Título del gráfico.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data debe ser un np.ndarray")
    if data.ndim != 3:
        raise ValueError("data debe ser una imagen RGB (3D)")
 
    patches = [
        mpatches.Patch(
            color=[c / 255 for c in color],
            label=etiqueta,
        )
        for etiqueta, color in leyenda.items()
    ]
 
    plt.figure(figsize=(7, 6))
    plt.imshow(data.astype(np.uint8))
    plt.legend(handles=patches, loc="lower right", fontsize=9)
    plt.title(titulo)
    plt.axis("off")
    plt.tight_layout()
    plt.show()