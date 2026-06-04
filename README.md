## BioImagenes

## Objetivo

El objetivo del proyecto es implementar una biblioteca orientada a objetos que permita representar, procesar y analizar imágenes biomédicas mediante operaciones como:

- Conversión a escala de grises.
- Aplicación de filtros mediante convolución.
- Gestión de metadatos.
- Registro de cambios realizados sobre la imagen.
- Procesamiento de distintos tipos de imágenes médicas.

## Descripción

La biblioteca está diseñada siguiendo principios de programación orientada a objetos.

Cada imagen se representa mediante la clase `Imagen`, que almacena:

- Datos de la imagen.
- Metadatos.
- Historial de modificaciones.

Además, se incluyen clases especializadas para distintos tipos de imágenes médicas:

- Radiografías.
- Tomografías.
- Imágenes termográficas.

## Instalación

### Clonar el repositorio

```bash
git clone https://github.com/ricardosilvaUTEC/ProcesamientoImagenes.git
cd ProcesamientoImagenes
```

## Ejemplos






## Diagrama UML

![Diagrama UML](docs/uml/DiagramaUML.png)

## Arquitectura

```text
src/
└── bioimagenes/
    ├── core/
    │   ├── imagen.py
    │   ├── info.py
    │   └── historial.py
    │
    ├── filtros/
    │   └── filtro.py
    │
    ├── medicas/
    │   ├── imagen_radiografia.py
    │   ├── imagen_termografica.py
    │   └── imagen_tomografia.py
    │
    ├── utils/
    │
    ├── visualizacion/
    │   ├── histogramas.py
    │   └── visualizar.py
test/
   ├── test_filtro.py
   ├── test_historial.py
   ├── test_imagen.py
   ├── test_info.py
   ├── test_termografica.py
   ├── test_tomografia.py
   └── test_radiografia.py
```

### Componentes principales

- **Imagen**: clase base para representar imágenes digitales.
- **Info**: almacena metadatos de la imagen.
- **Historial**: registra las modificaciones realizadas sobre una imagen.
- **Filtro**: implementa filtros mediante convolución.
- **Imágenes médicas**: especializaciones para radiografías, tomografías e imágenes termográficas.

## Ejecutar pruebas

Para ejecutar todos los tests:

```bash
pytest
```

Para ejecutar un test específico:

```bash
pytest tests/test_imagen.py
```

## Integrantes

- Alexis González
- Ricardo Silva