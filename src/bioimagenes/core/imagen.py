import numpy as np
import matplotlib.pyplot as plt

from bioimagenes.core.info import Info
from bioimagenes.core.historial import Historial
from bioimagenes.filtros.filtro import Filtro

class Imagen:
    """
    Clase base para el manejo y procesamiento de imágenes digitales.
    Representa una imagen como una matriz de datos y proporciona herramientas para su manipulación, visualización y análisis.
    Permite aplicar operaciones como filtrado, recorte, conversión de escala de grises y ajustes de contraste o brillo.
    Además, integra metadatos mediante la clase Info y mantiene un registro de cambios a través de la clase Historial.
    
    """
    def __init__(self, data:np.ndarray, info: Info = None):

        """
        Inicializa una instancia de la clase Imagen.
        Parámetros
        ----------
        data : np.ndarray
        Matriz que contiene los valores de los píxeles de la imagen.
        Puede ser 2D (escala de grises) o 3D (RGB).
        info : Info
        Objeto que contiene los metadatos asociados a la imagen.
        Si no se proporciona, se genera uno por defecto.
        Retorna
        -------
        None
         """
        #Varificamos que data sea un array de numpy
        if not isinstance(data, np.ndarray):
            raise TypeError("data debe ser un np.ndarray")
        
        #Verificamos que tenga dimensiones válidas (2D o 3D)
        if data.ndim not in [2, 3]:
            raise ValueError("La imagen debe ser 2D (gris) o 3D (RGB)")
        
        #Convertimos los datos a float32 para facilitar cálculos
        self.data = data.astype(np.float32)

        #Inicialización de Info e Historial
        if info is None:
            #Creamos un historial vacío
            historial = Historial()

            #Creamos los metadatos básicos, tmaño, brillo, corte
            datos = {"dimenciones": data.shape, "brillo": np.mean(data), "cortada":False}

            #Creamos el objeto info
            self.info = Info(datos=datos, historial=historial)
        else:
            #Si ya viene un objeto Info, lo usamos directamente
            self.info = info
        
    #Métodos principales
    def aplicar_filtro(self, filtro: Filtro):

        #Validamos que el objeto sea de tipo Filtro
        if not isinstance(filtro, Filtro):
            raise TypeError("Debe pasar un objeto a Filtro")
        
        #Aplicamos el filtro (convolución)
        nueva_data = filtro.aplicar(self.data)

        #Actualizamos la imagen con los nuevos datos
        self.data = nueva_data

        #Actualizamos el brillo en los metadatos
        self.info.datos["brillo"] = np.mean(self.data)

        #Registramos la operación en el historial
        self.info.historial.modificar_historial(f"Filtro aplica: {filtro.tipo}")
    
    def bn(self):
        """
        Convierte imagen RGB a escala de grises
        """

        #Si ya es escala de grises, no hacemos nada
        if self.data.ndim == 2:
            return
        
        #Convertimos a gris promediando los canales RGB
        self.data = np.mean(self.data, axis=2)

        #Registramos el cambio en el historial
        self.info.historial.modificar_historial("Conversion a blanco y negro")
    
    def visualizar(self):
        """
        Muestra la imagen utilizando matplotlib.
        """
        plt.figure()

        #Si es imagen en escala de grises
        if self.data.ndim == 2:
            plt.imshow(self.data, cmap = "gray")
        
        #Si es imagen RGB
        else:
            plt.imshow(self.data.astype(np.uint8))
        
        plt.title("Imagen")
        plt.axis("off")
        plt.show()
    
    """
    Otros metodos:
    """

    def __str__(self):
        """
        Devuelve un resumen técnico de la imagen.
        """
        return (
            f"Imagen: dimensiones={self.data.shape}, "
            f"brillo={np.mean(self.data):.2f}"
        )

    def __len__(self):
        """
        Retorna la cantidad total de píxeles de la imagen.
        """
        return self.data.size

    def __getitem__(self, key):
        """
        Permite acceder a los valores de los píxeles mediante índices.

        Ejemplo:
        --------
        img[10, 20]
        """
        return self.data[key]
