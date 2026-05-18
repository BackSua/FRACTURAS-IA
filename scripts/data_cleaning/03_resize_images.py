"""
Script 03 — Redimensionar todas las imágenes a 640x640 píxeles.

¿Por qué 640x640?
YOLOv8 acepta imágenes de cualquier tamaño pero internamente las redimensiona
a un tamaño fijo. Usar 640x640 es el estándar porque:
1. Es el default de YOLO — no hay redimensionamiento extra en inferencia.
2. Es un balance entre detalle (más píxeles = más detalle) y velocidad.
3. Es potencia de 2 cercana, lo que optimiza el uso de GPU (memoria alineada).

¿Por qué se usa INTER_AREA para reducir e INTER_LINEAR para ampliar?
- INTER_AREA: Usa remuestreo por área. Al reducir, promedia los píxeles vecinos
  en lugar de interpolar, lo que da imágenes más nítidas sin aliasing.
- INTER_LINEAR: Interpolación bilineal. Al ampliar, suaviza la transición
  entre píxeles para no generar artefactos de bloque.

Las anotaciones YOLO normalizadas (valores 0.0-1.0) NO requieren ajuste
al redimensionar, porque son proporciones relativas al tamaño de la imagen.
"""

import os
import sys
import shutil
from pathlib import Path

import cv2


TARGET_SIZE = 640


def resize_image(src_path: Path, dst_path: Path, target_size: int = TARGET_SIZE) -> bool:
    """
    Redimensiona una imagen al tamaño objetivo.

    Args:
        src_path: Ruta de la imagen original.
        dst_path: Ruta donde guardar la imagen redimensionada.
        target_size: Tamaño objetivo (ancho y alto iguales).

    Returns:
        True si se procesó correctamente, False si hubo error.
    """
    img = cv2.imread(str(src_path))
    if img is None:
        return False

    h, w = img.shape[:2]

    # Elegir interpolación según si se reduce o amplía
    if h > target_size or w > target_size:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    resized = cv2.resize(img, (target_size, target_size), interpolation=interpolation)

    # Crear directorio destino si no existe
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), resized)
    return True


def copy_annotation(src_label: Path, dst_label: Path):
    """
    Copia el archivo de anotación sin modificarlo.

    Las coordenadas YOLO están normalizadas (0.0 a 1.0), así que son
    independientes del tamaño de la imagen. No necesitan ajuste.
    """
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src_label), str(dst_label))


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / 'dataset' / 'raw'
    processed_dir = project_root / 'dataset' / 'processed'

    if not raw_dir.exists():
        print(f"ERROR: No se encontró {raw_dir}")
        sys.exit(1)

    print("=" * 60)
    print(f"REDIMENSIONAMIENTO DE IMÁGENES A {TARGET_SIZE}x{TARGET_SIZE}")
    print("=" * 60)

    # Buscar dónde están las imágenes y labels en el dataset raw
    # Adaptarse a la estructura real (puede variar según versión del dataset)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    stats = {'processed': 0, 'errors': 0, 'labels_copied': 0}

    for root, _, files in os.walk(raw_dir):
        for filename in files:
            src_path = Path(root) / filename
            # Mantener la estructura relativa de carpetas
            rel_path = src_path.relative_to(raw_dir)

            if src_path.suffix.lower() in image_extensions:
                dst_path = processed_dir / rel_path
                if resize_image(src_path, dst_path):
                    stats['processed'] += 1
                    if stats['processed'] % 500 == 0:
                        print(f"  Procesadas: {stats['processed']} imágenes...")
                else:
                    stats['errors'] += 1

            elif src_path.suffix.lower() == '.txt':
                # Copiar anotaciones sin modificar
                dst_path = processed_dir / rel_path
                copy_annotation(src_path, dst_path)
                stats['labels_copied'] += 1

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"  Imágenes redimensionadas: {stats['processed']}")
    print(f"  Anotaciones copiadas:     {stats['labels_copied']}")
    print(f"  Errores:                  {stats['errors']}")
    print(f"  Directorio destino:       {processed_dir}")


if __name__ == "__main__":
    main()
