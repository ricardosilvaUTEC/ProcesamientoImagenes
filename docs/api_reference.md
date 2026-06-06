# API Reference — bioimagenes

Documentación de referencia de todas las clases y métodos públicos de la librería.

---

## bioimagenes.core.Historial

Registra cronológicamente las operaciones aplicadas sobre una imagen.

### Constructor

```python
Historial(lista_cambios: list = None)
```

| Parámetro | Tipo | Descripción |
|---|---|---|
| `lista_cambios` | `list` | Lista inicial de cambios. Cada cambio es un dict con claves `operacion` y `fecha`. Opcional. |

### Propiedades

| Nombre | Tipo | Descripción |
|---|---|---|
| `lista_cambios` | `list` | Lista completa de cambios registrados. |
| `ultimo_cambio` | `dict \| None` | Último cambio registrado, o `None` si está vacío. |

### Métodos

| Método | Descripción |
|---|---|
| `modificar_historial(entrada: str)` | Añade una nueva entrada con la operación y la fecha actual. |

### Métodos nativos

| Método | Descripción |
|---|---|
| `__len__()` | Retorna la cantidad total de cambios. |
| `__iter__()` | Permite iterar sobre los cambios. |
| `__str__()` | Resumen del historial y último cambio. |

---

## bioimagenes.core.Info

Contenedor estructurado de metadatos asociados a una imagen. Funciona como diccionario.

### Constructor

```python
Info(datos: dict = None, historial: Historial = None)
```

| Parámetro | Tipo | Descripción |
|---|---|---|
| `datos` | `dict` | Metadatos iniciales. Ejemplo: `{"dimensiones": (100, 100), "brillo": 120.0, "cortada": False}`. |
| `historial` | `Historial` | Instancia de Historial asociada. Se crea una nueva si no se proporciona. |

### Propiedades

| Nombre | Tipo | Descripción |
|---|---|---|
| `dimensiones` | `tuple` | Resolución espacial y canales de la imagen. |
| `brillo` | `float` | Valor promedio de brillo. |
| `cortada` | `bool` | Indica si la imagen proviene de un recorte. |
| `historial` | `Historial` | Instancia de Historial asociada. |
| `datos` | `dict` | Diccionario completo de metadatos. |

### Métodos nativos

| Método | Descripción |
|---|---|
| `__contains__(clave)` | Verifica si una clave existe: `"brillo" in info`. |
| `__getitem__(clave)` | Accede a un valor: `info["brillo"]`. |
| `__setitem__(clave, valor)` | Asigna un valor: `info["brillo"] = 120`. |
| `__str__()` | Muestra todos los metadatos formateados. |

---

## bioimagenes.core.Imagen

Clase base para el manejo de imágenes digitales 2D y 3D.

### Constructor

```python
Imagen(data: np.ndarray, info: Info = None)
```

| Parámetro | Tipo | Descripción |
|---|---|---|
| `data` | `np.ndarray` | Matriz 2D (grises) o 3D (RGB) con los valores de píxeles. |
| `info` | `Info` | Metadatos. Se genera uno por defecto si no se proporciona. |

### Propiedades

| Nombre | Tipo | Descripción |
|---|---|---|
| `data` | `np.ndarray` | Matriz de datos de la imagen. |
| `info` | `Info` | Metadatos de la imagen. |

### Métodos

| Método | Descripción |
|---|---|
| `aplicar_filtro(filtro: Filtro)` | Aplica un filtro sobre la imagen mediante convolución. |
| `bn()` | Convierte imagen RGB a escala de grises usando pesos ITU-R BT.601. |
| `visualizar(titulo: str)` | Muestra la imagen con matplotlib. |
| `comparar(otra: Imagen, titulo_self, titulo_otra)` | Muestra dos imágenes lado a lado. |
| `histograma(bins: int)` | Muestra el histograma de intensidades. |

### Métodos nativos

| Método | Descripción |
|---|---|
| `__str__()` | Resumen técnico: shape, tipo, píxeles, brillo. |
| `__len__()` | Número total de píxeles. |
| `__getitem__(indice)` | Acceso a píxeles: `img[50, 50]`, `img[10:20, 10:20]`. |

---

## bioimagenes.filtros.Filtro

Filtro espacial por convolución con kernel 2D.

### Constructor

```python
Filtro(tipo: str, kernel: np.ndarray)
```

| Parámetro | Tipo | Descripción |
|---|---|---|
| `tipo` | `str` | Nombre del filtro (ej: `"Gaussiano"`, `"Sobel"`). |
| `kernel` | `np.ndarray` | Matriz 2D con los pesos del filtro. |

### Propiedades

| Nombre | Tipo | Descripción |
|---|---|---|
| `tipo` | `str` | Identificador del filtro. |
| `kernel` | `np.ndarray` | Matriz de pesos. |
| `tamaño` | `tuple` | Dimensiones del kernel (filas, columnas). |

### Métodos

| Método | Descripción |
|---|---|
| `convolucion(imagen: np.ndarray)` | Ejecuta la convolución sobre un array 2D o 3D. |
| `aplicar(imagen: np.ndarray)` | Alias de `convolucion()`. |

### Métodos nativos

| Método | Descripción |
|---|---|
| `__str__()` | Nombre del filtro y tamaño del kernel. |

---

## bioimagenes.medicas.ImagenTomografia

Hereda de `Imagen`. Manejo de volúmenes CT con soporte para unidades Hounsfield.

### Constructor

```python
ImagenTomografia(data: np.ndarray, info: Info = None, espaciado: tuple = (1.0, 1.0, 1.0))
```

| Parámetro | Tipo | Descripción |
|---|---|---|
| `data` | `np.ndarray` | Volumen 3D `(n_slices, H, W)` o corte 2D `(H, W)`. |
| `espaciado` | `tuple` | Tamaño físico del voxel en mm `(dz, dy, dx)`. |

### Propiedades

| Nombre | Tipo | Descripción |
|---|---|---|
| `espaciado` | `tuple` | Tamaño físico del voxel en mm. |
| `n_slices` | `int` | Número de cortes del volumen. |
| `en_hu` | `bool` | Indica si los datos están en unidades Hounsfield. |

### Métodos

| Método | Descripción |
|---|---|
| `convertir_a_hu(slope, intercept)` | Convierte valores de píxel a unidades Hounsfield. |
| `normalizar(rango_min, rango_max)` | Normaliza intensidades al rango [0, 1]. |
| `cargar_volumen(slices)` | Carga o reemplaza el volumen con un array 3D. |
| `get_slice(indice: int)` | Retorna un corte axial del volumen. |
| `visualizar_corte(indice, mostrar_referencia)` | Muestra el corte en grises y coloreado por tejido (aire, pulmón, grasa, tejido blando, sangre, hueso). |
| `ajustar_ventana(center, width)` | Aplica windowing médico al volumen. |
| `aplicar_ventana_predefinida(nombre)` | Aplica una ventana predefinida: `pulmón`, `hueso`, `tejido`, `cerebro`, `abdomen`. |

---

## bioimagenes.medicas.ImagenRadiografia

Hereda de `Imagen`. Manejo de radiografías 2D en escala de grises.

### Constructor

```python
ImagenRadiografia(data: np.ndarray, info: Info = None, modalidad: str = "RX")
```

| Parámetro | Tipo | Descripción |
|---|---|---|
| `data` | `np.ndarray` | Imagen 2D en escala de grises. |
| `modalidad` | `str` | Tipo de estudio (ej: `"RX-tórax"`). |

### Propiedades

| Nombre | Tipo | Descripción |
|---|---|---|
| `modalidad` | `str` | Tipo de estudio radiográfico. |

### Métodos

| Método | Descripción |
|---|---|
| `mejorar_contraste(clip_limit, tile_grid)` | Mejora el contraste con CLAHE (OpenCV). |
| `ecualizar_histograma()` | Ecualización global del histograma. |
| `invertir()` | Invierte la escala de intensidades. |
| `detectar_bordes(metodo)` | Detecta bordes: `"sobel"`, `"canny"`, `"laplacian"`. No modifica `data`. |
| `seleccionar_roi(fila_ini, fila_fin, col_ini, col_fin)` | Recorta una región de interés. Retorna nueva `ImagenRadiografia`. |
| `normalizar()` | Normaliza al rango [0, 1]. |
| `visualizar_cluster(imagenes, n_clusters)` | Agrupa imágenes con KMeans + PCA y muestra scatter 2D interactivo con miniatura al hover. |

---

## bioimagenes.medicas.ImagenTermografica

Hereda de `Imagen`. Manejo de imágenes térmicas con conversión a grados Celsius.

### Constructor

```python
ImagenTermografica(data: np.ndarray, info: Info = None, t_min: float = 30.0, t_max: float = 40.0)
```

| Parámetro | Tipo | Descripción |
|---|---|---|
| `data` | `np.ndarray` | Imagen 2D en escala de grises (0–255). |
| `t_min` | `float` | Temperatura mínima del rango de conversión en °C. |
| `t_max` | `float` | Temperatura máxima del rango de conversión en °C. |

### Propiedades

| Nombre | Tipo | Descripción |
|---|---|---|
| `t_min` | `float` | Temperatura mínima del rango. |
| `t_max` | `float` | Temperatura máxima del rango. |
| `en_temperatura` | `bool` | Indica si los datos ya están en °C. |

### Métodos

| Método | Descripción |
|---|---|
| `convertir_a_temperatura(t_min, t_max)` | Convierte escala de grises (0–255) a °C. |
| `normalizar()` | Normaliza al rango [0, 1]. |
| `mapa_calor(colormap)` | Visualiza la imagen como mapa de calor con colorbar. |
| `detectar_puntos_calientes(umbral)` | Retorna máscara booleana de píxeles sobre el umbral. |
| `segmentar_por_umbral(umbral_inf, umbral_sup)` | Segmenta píxeles dentro de un rango térmico. |
| `visualizar_segmentacion(umbral)` | Muestra imagen térmica con zonas calientes superpuestas en rojo. |

---

## Módulo visualizacion

### bioimagenes.visualizacion.visualizar

| Función | Descripción |
|---|---|
| `mostrar_imagen(data, titulo)` | Muestra imagen 2D o 3D. |
| `mostrar_comparacion(data_original, data_procesada, ...)` | Muestra dos imágenes lado a lado. |
| `mostrar_corte(data, indice, eje, titulo)` | Muestra un corte axial, coronal o sagital de un volumen 3D. |
| `mostrar_mapa_calor(data, titulo, cmap, unidad)` | Muestra imagen como mapa de calor con colorbar. |
| `mostrar_imagen_con_leyenda(data, leyenda, titulo)` | Muestra imagen RGB con leyenda de colores para segmentaciones. |

### bioimagenes.visualizacion.histogramas

| Función | Descripción |
|---|---|
| `histograma(data, titulo, bins)` | Histograma de una imagen en escala de grises. |
| `histograma_rgb(data, titulo, bins)` | Histograma por canal R, G, B. |
| `histograma_comparacion(...)` | Compara histogramas de dos imágenes. |
| `histograma_acumulado(data, titulo, bins)` | Histograma acumulado. |