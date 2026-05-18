"""
Script 01 — Verificar integridad de imágenes del dataset.

Recorre todas las imágenes en dataset/raw/ e intenta cargarlas con OpenCV.
Las imágenes corruptas (que no pueden leerse) se eliminan automáticamente.

¿Por qué es necesario?
Una imagen corrupta durante el entrenamiento puede causar que YOLO lance
una excepción y detenga todo el proceso. Es más eficiente limpiarlas antes
que manejar errores durante horas de entrenamiento.
"""

import os
import sys
from pathlib import Path

import cv2


# Formatos de imagen que OpenCV puede leer
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def verify_images(dataset_dir: Path) -> dict:
    """
    Recorre todas las imágenes del directorio y verifica que sean legibles.

    Args:
        dataset_dir: Ruta al directorio raíz del dataset (dataset/raw/).

    Returns:
        Diccionario con estadísticas: válidas, corruptas, eliminadas.
    """
    stats = {'valid': 0, 'corrupt': 0, 'skipped': 0}
    corrupt_files = []

    # Recorrer recursivamente buscando archivos de imagen
    for root, _, files in os.walk(dataset_dir):
        for filename in files:
            filepath = Path(root) / filename

            # Solo procesar archivos con extensión de imagen
            if filepath.suffix.lower() not in VALID_EXTENSIONS:
                stats['skipped'] += 1
                continue

            # Intentar cargar la imagen con OpenCV
            # cv2.imread retorna None si el archivo está corrupto o no es imagen válida
            img = cv2.imread(str(filepath))

            if img is None:
                print(f"  [CORRUPTA] {filepath}")
                corrupt_files.append(filepath)
                stats['corrupt'] += 1
            else:
                stats['valid'] += 1

    # Eliminar archivos corruptos
    for filepath in corrupt_files:
        filepath.unlink()
        print(f"  [ELIMINADA] {filepath}")

    return stats


def main():
    # Ruta relativa desde la raíz del proyecto
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = project_root / 'dataset' / 'raw'

    if not dataset_dir.exists():
        print(f"ERROR: No se encontró el directorio del dataset: {dataset_dir}")
        print("Descarga el dataset de Kaggle y colócalo en dataset/raw/")
        sys.exit(1)

    print("=" * 60)
    print("VERIFICACIÓN DE INTEGRIDAD DE IMÁGENES")
    print("=" * 60)
    print(f"Directorio: {dataset_dir}")
    print("Procesando...\n")

    stats = verify_images(dataset_dir)

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"  Imágenes válidas:   {stats['valid']}")
    print(f"  Imágenes corruptas: {stats['corrupt']}")
    print(f"  Archivos omitidos:  {stats['skipped']}")
    print(f"  Total procesados:   {stats['valid'] + stats['corrupt']}")


if __name__ == "__main__":
    main()
