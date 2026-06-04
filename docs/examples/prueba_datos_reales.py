"""
prueba_datos_reales.py
──────────────────────
Script de prueba con los datos reales de la base de datos del TPI.

Uso:
    python prueba_datos_reales.py              # prueba las 3 modalidades
    python prueba_datos_reales.py --modo rx    # solo radiografías
    python prueba_datos_reales.py --modo ct    # solo tomografía
    python prueba_datos_reales.py --modo term  # solo termografías
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# ── importamos nuestra librería ───────────────────────────────────────────────
try:
    from bioimagenes.medicas.imagen_tomografia import ImagenTomografia
    from bioimagenes.medicas.imagen_radiografia import ImagenRadiografia
    from bioimagenes.medicas.imagen_termografica import ImagenTermografica
    from bioimagenes.filtros.filtro import Filtro
except ImportError:
    print("[ERROR] No se encontró el paquete bioimagenes.")
    print("        Asegurate de haber instalado con: pip install -e .")
    sys.exit(1)

# ── rutas a los datos ─────────────────────────────────────────────────────────
RUTA_CT      = os.path.join("data", "tomografia", "AC421363f.nii", "AC421363f.nii")
RUTA_TERM    = os.path.join("data", "termografias")
RUTA_RX      = os.path.join("data", "radiografias", "sample")


# ─────────────────────────────────────────────────────────────────────────────
# Tomografía CT
# ─────────────────────────────────────────────────────────────────────────────

def prueba_tomografia():
    print("\n" + "="*60)
    print("  TOMOGRAFÍA CT — AC421363f.nii")
    print("="*60)

    try:
        import nibabel as nib
    except ImportError:
        print("  [ERROR] nibabel no está instalado.")
        print("          Instalalo con: pip install nibabel")
        return

    if not os.path.exists(RUTA_CT):
        print(f"  [ERROR] No se encontró: {RUTA_CT}")
        return

    # ── Carga ─────────────────────────────────────────────────────────────────
    print("\n── 1. Cargando volumen .nii ───────────────────────────────")
    nii  = nib.load(RUTA_CT)
    data = nii.get_fdata().astype(np.float32)
    print(f"  Shape del volumen: {data.shape}")

    # nibabel carga en formato (X, Y, Z) → reordenamos a (Z, X, Y) para slices axiales
    if data.ndim == 3:
        data = np.transpose(data, (2, 0, 1))
    print(f"  Shape reordenado (slices, H, W): {data.shape}")

    ct = ImagenTomografia(data=data)
    print(ct)

    # ── Visualización básica ──────────────────────────────────────────────────
    print("\n── 2. Visualizando corte central ──────────────────────────")
    ct.visualizar(titulo="Tomografía CT — corte central")

    # ── Corte con segmentación de tejidos ─────────────────────────────────────
    print("\n── 3. Segmentación de tejidos en corte central ────────────")
    corte_central = data.shape[0] // 2
    print(f"  Aplicando visualizar_corte() en índice {corte_central}...")
    ct.visualizar_corte(indice=corte_central)

    # ── Ventana de visualización ──────────────────────────────────────────────
    print("\n── 4. Ventanas de visualización médica ────────────────────")
    for ventana in ["tejido", "pulmón", "hueso"]:
        resultado = ct.aplicar_ventana_predefinida(ventana)
        print(f"  ✓ Ventana '{ventana}': min={resultado.min():.1f}, max={resultado.max():.1f}")

    # ── Historial ─────────────────────────────────────────────────────────────
    print("\n── 5. Historial de operaciones ────────────────────────────")
    print(ct.info.historial)


# ─────────────────────────────────────────────────────────────────────────────
# Radiografías
# ─────────────────────────────────────────────────────────────────────────────

def prueba_radiografias():
    print("\n" + "="*60)
    print("  RADIOGRAFÍAS")
    print("="*60)

    try:
        import cv2
    except ImportError:
        print("  [ERROR] opencv-python no está instalado.")
        print("          Instalalo con: pip install opencv-python")
        return

    if not os.path.exists(RUTA_RX):
        print(f"  [ERROR] No se encontró: {RUTA_RX}")
        return

    archivos = [f for f in os.listdir(RUTA_RX) if f.endswith(".png")]
    if not archivos:
        print("  [ERROR] No se encontraron imágenes .png en la carpeta.")
        return

    print(f"\n  {len(archivos)} radiografías encontradas.")

    # Usamos la primera para las pruebas individuales
    ruta_rx = os.path.join(RUTA_RX, archivos[0])
    print(f"  Usando: {archivos[0]}")

    # ── Carga ─────────────────────────────────────────────────────────────────
    print("\n── 1. Cargando radiografía ────────────────────────────────")
    img = cv2.imread(ruta_rx, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [ERROR] No se pudo cargar la imagen.")
        return

    rx = ImagenRadiografia(data=img, modalidad="RX-tórax")
    print(rx)
    rx.visualizar(titulo="Radiografía original")

    # ── Mejora de contraste ───────────────────────────────────────────────────
    print("\n── 2. Mejora de contraste (CLAHE) ─────────────────────────")
    rx_contraste = ImagenRadiografia(data=img.copy(), modalidad="RX-tórax")
    rx_contraste.mejorar_contraste()
    rx_contraste.visualizar(titulo="Radiografía — contraste mejorado")

    # ── Ecualización ──────────────────────────────────────────────────────────
    print("\n── 3. Ecualización de histograma ──────────────────────────")
    rx_ecual = ImagenRadiografia(data=img.copy(), modalidad="RX-tórax")
    rx_ecual.ecualizar_histograma()
    rx_ecual.visualizar(titulo="Radiografía — ecualizada")

    # ── Detección de bordes ───────────────────────────────────────────────────
    print("\n── 4. Detección de bordes ─────────────────────────────────")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rx_bordes = ImagenRadiografia(data=img.copy(), modalidad="RX-tórax")

    for ax, metodo in zip(axes, ["sobel", "canny", "laplacian"]):
        bordes = rx_bordes.detectar_bordes(metodo=metodo)
        ax.imshow(bordes, cmap="gray")
        ax.set_title(f"Bordes — {metodo}")
        ax.axis("off")
        print(f"  ✓ {metodo}: shape={bordes.shape}")

    plt.suptitle("Detección de bordes en radiografía real", fontsize=13)
    plt.tight_layout()
    plt.show()

    # ── Inversión ─────────────────────────────────────────────────────────────
    print("\n── 5. Inversión de escala ─────────────────────────────────")
    rx_inv = ImagenRadiografia(data=img.copy(), modalidad="RX-tórax")
    rx_inv.invertir()
    rx_inv.visualizar(titulo="Radiografía — invertida")

    # ── ROI ───────────────────────────────────────────────────────────────────
    print("\n── 6. Selección de ROI ────────────────────────────────────")
    H, W = img.shape
    fi, ff = H // 4, 3 * H // 4
    ci, cf = W // 4, 3 * W // 4
    roi = rx.seleccionar_roi(fi, ff, ci, cf)
    print(f"  ROI shape: {roi.data.shape}")
    roi.visualizar(titulo="Radiografía — ROI central")

    # ── Historial ─────────────────────────────────────────────────────────────
    print("\n── 7. Historial ───────────────────────────────────────────")
    print(rx_contraste.info.historial)


# ─────────────────────────────────────────────────────────────────────────────
# Termografías
# ─────────────────────────────────────────────────────────────────────────────

def prueba_termografias():
    print("\n" + "="*60)
    print("  TERMOGRAFÍAS")
    print("="*60)

    try:
        import cv2
    except ImportError:
        print("  [ERROR] opencv-python no está instalado.")
        return

    if not os.path.exists(RUTA_TERM):
        print(f"  [ERROR] No se encontró: {RUTA_TERM}")
        return

    archivos = sorted([f for f in os.listdir(RUTA_TERM) if f.endswith(".jpg")])
    if not archivos:
        print("  [ERROR] No se encontraron imágenes .jpg en la carpeta.")
        return

    print(f"\n  {len(archivos)} imágenes térmicas encontradas: {archivos}")

    # Usamos la primera imagen
    ruta_term = os.path.join(RUTA_TERM, archivos[0])
    print(f"\n  Usando: {archivos[0]}")

    # ── Carga ─────────────────────────────────────────────────────────────────
    print("\n── 1. Cargando imagen térmica ─────────────────────────────")
    img = cv2.imread(ruta_term, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [ERROR] No se pudo cargar la imagen.")
        return

    # Rango de temperatura según el TPI
    T_MIN, T_MAX = 30.0, 36.5

    term = ImagenTermografica(data=img, t_min=T_MIN, t_max=T_MAX)
    print(term)

    # ── Mapa de calor (antes de conversión) ───────────────────────────────────
    print("\n── 2. Mapa de calor (escala gris original) ────────────────")
    term.mapa_calor(colormap="inferno")

    # ── Conversión a temperatura ───────────────────────────────────────────────
    print("\n── 3. Conversión a °C ─────────────────────────────────────")
    term.convertir_a_temperatura()
    print(f"  Temperatura mín: {term.data.min():.2f}°C")
    print(f"  Temperatura máx: {term.data.max():.2f}°C")
    print(f"  Temperatura media: {term.data.mean():.2f}°C")
    term.mapa_calor(colormap="inferno")

    # ── Detección de puntos calientes ─────────────────────────────────────────
    print("\n── 4. Detección de puntos calientes ───────────────────────")
    umbral = T_MIN + (T_MAX - T_MIN) * 0.75  # 75% del rango
    mascara = term.detectar_puntos_calientes(umbral=umbral)
    print(f"  Umbral: {umbral:.2f}°C")
    print(f"  Píxeles calientes: {mascara.sum()} ({100*mascara.mean():.1f}% de la imagen)")
    term.visualizar_segmentacion(umbral=umbral)

    # ── Segmentación por rango ────────────────────────────────────────────────
    print("\n── 5. Segmentación por rango térmico ──────────────────────")
    rango_inf = T_MIN + (T_MAX - T_MIN) * 0.4
    rango_sup = T_MIN + (T_MAX - T_MIN) * 0.7
    mascara_rango = term.segmentar_por_umbral(rango_inf, rango_sup)
    print(f"  Rango: [{rango_inf:.2f}°C, {rango_sup:.2f}°C]")
    print(f"  Píxeles en rango: {mascara_rango.sum()}")

    # ── Comparación de todas las imágenes ─────────────────────────────────────
    print("\n── 6. Comparación de todas las termografías ───────────────")
    n = len(archivos)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, archivo in zip(axes, archivos):
        ruta = os.path.join(RUTA_TERM, archivo)
        img_i = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img_i is not None:
            t = ImagenTermografica(data=img_i, t_min=T_MIN, t_max=T_MAX)
            t.convertir_a_temperatura()
            im = ax.imshow(t.data, cmap="inferno", vmin=T_MIN, vmax=T_MAX)
            ax.set_title(archivo, fontsize=7)
            ax.axis("off")

    fig.colorbar(im, ax=axes, label="Temperatura (°C)", shrink=0.8)
    plt.suptitle("Comparación de todas las termografías", fontsize=12)
    plt.tight_layout()
    plt.show()

    # ── Historial ─────────────────────────────────────────────────────────────
    print("\n── 7. Historial ───────────────────────────────────────────")
    print(term.info.historial)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prueba con datos reales del TPI")
    parser.add_argument(
        "--modo", type=str, default="all",
        choices=["all", "ct", "rx", "term"],
        help="Modalidad a probar: all, ct, rx, term. Por defecto: all"
    )
    args = parser.parse_args()

    if args.modo in ("all", "ct"):
        prueba_tomografia()

    if args.modo in ("all", "rx"):
        prueba_radiografias()

    if args.modo in ("all", "term"):
        prueba_termografias()

    print("\n✓ Pruebas con datos reales completadas.\n")


if __name__ == "__main__":
    main()
