import os
import cv2
from bioimagenes.medicas.imagen_radiografia import ImagenRadiografia

RUTA = os.path.join("data", "radiografias", "sample")

# Cargamos todas las radiografías
rutas = [os.path.join(RUTA, f) for f in os.listdir(RUTA) if f.endswith(".png")]
print(f"Imágenes encontradas: {len(rutas)}")

# Creamos una instancia con la primera imagen
img = cv2.imread(rutas[0], cv2.IMREAD_GRAYSCALE)
rx = ImagenRadiografia(data=img, modalidad="RX-tórax")

# Visualizamos los clusters pasando las rutas directamente
rx.visualizar_cluster(imagenes=rutas, n_clusters=3)