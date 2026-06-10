"""
bioimagenes.medicas
===================
Módulo con clases especializadas para modalidades de imagen médica.

Clases
------
ImagenTomografia    : Volúmenes CT con soporte HU, windowing y segmentación.
ImagenRadiografia   : Radiografías 2D con mejora de contraste y detección de bordes.
ImagenTermografica  : Imágenes térmicas con conversión a °C y mapas de calor.

Todas heredan de bioimagenes.core.Imagen.
"""

from bioimagenes.medicas.imagen_tomografia import ImagenTomografia
from bioimagenes.medicas.imagen_radiografia import ImagenRadiografia
from bioimagenes.medicas.imagen_termografica import ImagenTermografica

__all__ = ["ImagenTomografia", "ImagenRadiografia", "ImagenTermografica"]