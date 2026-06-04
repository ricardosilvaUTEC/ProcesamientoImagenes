"""
prueba_rgb.py
─────────────
Script de prueba para la clase base Imagen con imágenes RGB cargadas desde archivo.

Uso:
    python prueba_rgb.py                        # usa imagen sintética de ejemplo
    python prueba_rgb.py --ruta foto.jpg        # usa tu propia imagen
    python prueba_rgb.py --ruta foto.png --todo # corre todas las pruebas
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

# ── intentamos cargar cv2, si no está usamos matplotlib ──────────────────────
try:
    import cv2
    TIENE_CV2 = True
except ImportError:
    TIENE_CV2 = False

# ── importamos nuestra librería ───────────────────────────────────────────────
try:
    from bioimagenes.core.imagen import Imagen
    from bioimagenes.filtros.filtro import Filtro
except ImportError:
    print("[ERROR] No se encontró el paquete bioimagenes.")
    print("        Asegurate de haber instalado con:  pip install -e .")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de carga
# ─────────────────────────────────────────────────────────────────────────────

def cargar_imagen(ruta: str) -> np.ndarray:
    """Carga una imagen RGB desde disco. Usa cv2 si está disponible."""
    if TIENE_CV2:
        img = cv2.imread(ruta)
        if img is None:
            raise FileNotFoundError(f"No se pudo abrir: {ruta}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 carga en BGR
    else:
        img = plt.imread(ruta)
        if img is None:
            raise FileNotFoundError(f"No se pudo abrir: {ruta}")
        # plt.imread devuelve float [0,1] para png → convertimos a [0,255]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        return img.astype(np.uint8)


def imagen_ejemplo() -> np.ndarray:
    """Genera una imagen RGB sintética de 200×200 con degradados de color."""
    np.random.seed(42)
    h, w = 200, 200
    data = np.zeros((h, w, 3), dtype=np.uint8)
    data[:, :, 0] = np.linspace(0, 255, w)             # canal R: degradado horizontal
    data[:, :, 1] = np.linspace(0, 255, h)[:, None]    # canal G: degradado vertical
    data[:, :, 2] = np.random.randint(50, 200, (h, w)) # canal B: ruido
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Pruebas individuales
# ─────────────────────────────────────────────────────────────────────────────

def prueba_creacion(data: np.ndarray) -> Imagen:
    print("\n── 1. Creación de Imagen ──────────────────────────────────")
    img = Imagen(data=data)
    print(img)
    print(f"  Píxeles totales  : {len(img)}")
    print(f"  Pixel [0, 0]     : {img[0, 0]}")
    print(f"  Región [10:12, 10:12]:\n{img[10:12, 10:12]}")
    print(f"  Brillo promedio  : {img.info['brillo']:.2f}")
    print(f"  Dimensiones      : {img.info['dimensiones']}")
    print(f"  Historial vacío  : {len(img.info.historial)} cambios")
    return img


def prueba_visualizar(img: Imagen):
    print("\n── 2. Visualización ───────────────────────────────────────")
    print("  Mostrando imagen RGB original...")
    img.visualizar(titulo="Imagen RGB — original")


def prueba_filtros(img: Imagen) -> Imagen:
    print("\n── 3. Filtros ─────────────────────────────────────────────")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Filtro de suavizado promedio 5×5
    kernel_suave = np.ones((5, 5), dtype=np.float32) / 25.0
    f_suave = Filtro(tipo="Promedio 5×5", kernel=kernel_suave)
    img_suave = Imagen(data=img.data.copy())
    img_suave.aplicar_filtro(f_suave)
    print(f"  ✓ {f_suave}")

    # Filtro Gaussiano 3×3
    kernel_gauss = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ], dtype=np.float32) / 16.0
    f_gauss = Filtro(tipo="Gaussiano 3×3", kernel=kernel_gauss)
    img_gauss = Imagen(data=img.data.copy())
    img_gauss.aplicar_filtro(f_gauss)
    print(f"  ✓ {f_gauss}")

    # Filtro Sobel (detección de bordes horizontal)
    kernel_sobel = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1],
    ], dtype=np.float32)
    f_sobel = Filtro(tipo="Sobel", kernel=kernel_sobel)
    img_sobel = Imagen(data=img.data.copy())
    img_sobel.aplicar_filtro(f_sobel)
    print(f"  ✓ {f_sobel}")

    # Mostramos los 3 resultados en una figura comparativa
    def _show(ax, imagen, titulo):
        datos = np.clip(imagen.data / imagen.data.max(), 0, 1) if imagen.data.ndim == 3 else imagen.data
        ax.imshow(datos, cmap=None if imagen.data.ndim == 3 else "gray")
        ax.set_title(titulo)
        ax.axis("off")

    _show(axes[0], img_suave,  "Promedio 5×5")
    _show(axes[1], img_gauss,  "Gaussiano 3×3")
    _show(axes[2], img_sobel,  "Sobel")
    plt.suptitle("Comparación de filtros", fontsize=13)
    plt.tight_layout()
    plt.show()

    return img_suave


def prueba_bn(img: Imagen) -> Imagen:
    print("\n── 4. Conversión a Blanco y Negro ─────────────────────────")
    img_bn = Imagen(data=img.data.copy())
    print(f"  Antes  → ndim={img_bn.data.ndim}, shape={img_bn.data.shape}")
    img_bn.bn()
    print(f"  Después → ndim={img_bn.data.ndim}, shape={img_bn.data.shape}")
    print(f"  Historial: {img_bn.info.historial.ultimo_cambio}")
    return img_bn


def prueba_comparacion(original: Imagen, bn: Imagen):
    print("\n── 5. Comparación RGB vs BN ───────────────────────────────")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(np.clip(original.data / 255.0, 0, 1))
    axes[0].set_title("Original RGB")
    axes[0].axis("off")
    axes[1].imshow(bn.data, cmap="gray")
    axes[1].set_title("Blanco y Negro")
    axes[1].axis("off")
    plt.suptitle("Comparación RGB vs BN", fontsize=13)
    plt.tight_layout()
    plt.show()


def prueba_acceso_pixeles(img: Imagen):
    print("\n── 6. Acceso a píxeles con __getitem__ ────────────────────")
    print(f"  img[0, 0]         = {img[0, 0]}")
    print(f"  img[0, :3]        = {img[0, :3]}")       # primeras 3 columnas de fila 0
    print(f"  img[:3, :3].shape = {img[:3, :3].shape}") # región 3×3


def prueba_historial(img: Imagen):
    print("\n── 7. Historial de transformaciones ───────────────────────")
    h = img.info.historial
    print(f"  Total de cambios: {len(h)}")
    for cambio in h:
        print(f"    [{cambio['fecha']}]  {cambio['operacion']}")


def prueba_errores():
    print("\n── 8. Validaciones y errores esperados ────────────────────")

    casos = [
        ("data no es ndarray",  lambda: Imagen(data=[[1, 2], [3, 4]]),           TypeError),
        ("data 1D",             lambda: Imagen(data=np.ones(100)),                ValueError),
        ("data 4D",             lambda: Imagen(data=np.ones((2, 2, 2, 2))),       ValueError),
        ("info inválida",       lambda: Imagen(data=np.ones((10, 10)), info="x"), TypeError),
        ("filtro inválido",     lambda: Imagen(data=np.ones((10, 10))).aplicar_filtro("x"), TypeError),
        ("kernel no 2D",        lambda: Filtro(tipo="T", kernel=np.ones((3,3,3))), ValueError),
        ("tipo vacío en Filtro",lambda: Filtro(tipo="  ", kernel=np.ones((3,3))), TypeError),
    ]

    for nombre, accion, exc_esperada in casos:
        try:
            accion()
            print(f"  ✗ [{nombre}] — no lanzó excepción (revisar)")
        except exc_esperada as e:
            print(f"  ✓ [{nombre}] → {exc_esperada.__name__}: {e}")
        except Exception as e:
            print(f"  ? [{nombre}] — excepción inesperada {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prueba de Imagen RGB")
    parser.add_argument(
        "--ruta", type=str, default=None,
        help="Ruta a una imagen jpg/png. Si se omite, se usa una imagen sintética."
    )
    parser.add_argument(
        "--todo", action="store_true",
        help="Ejecuta todas las pruebas incluyendo comparación, acceso a píxeles y errores."
    )
    args = parser.parse_args()

    # ── Carga ─────────────────────────────────────────────────────────────────
    if args.ruta:
        print(f"\nCargando imagen desde: {args.ruta}")
        try:
            data = cargar_imagen(args.ruta)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
    else:
        print("\nNo se indicó --ruta. Usando imagen sintética de ejemplo (200×200 RGB).")
        data = imagen_ejemplo()

    print(f"  Shape: {data.shape} | dtype: {data.dtype}")

    # ── Pruebas ───────────────────────────────────────────────────────────────
    img          = prueba_creacion(data)
    prueba_visualizar(img)
    img_filtrada = prueba_filtros(img)
    img_bn       = prueba_bn(img)

    if args.todo:
        prueba_comparacion(img, img_bn)
        prueba_acceso_pixeles(img)
        prueba_historial(img_filtrada)
        prueba_errores()

    print("\n✓ Pruebas completadas.\n")


if __name__ == "__main__":
    main()