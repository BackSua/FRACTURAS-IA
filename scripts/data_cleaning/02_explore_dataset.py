"""
Script 02 — Explorar la estructura real del dataset descargado.

ESTE SCRIPT DEBE EJECUTARSE PRIMERO, ANTES DE CUALQUIER OTRO.

¿Por qué?
No se debe asumir cómo está organizado el dataset. Diferentes versiones
de GRAZPEDWRI-DX pueden tener estructuras distintas. Este script descubre
la estructura real para que los scripts siguientes se adapten correctamente.

Analiza:
- Estructura de directorios completa
- Cantidad de imágenes por carpeta
- Formatos de imagen presentes
- Formato de las anotaciones (YOLO .txt, PASCAL VOC .xml, COCO .json)
- Clases distintas en las anotaciones
- Tamaños de imagen más comunes
"""

import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

import cv2


VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
ANNOTATION_EXTENSIONS = {'.txt', '.xml', '.json'}


def print_directory_tree(root_dir: Path, prefix: str = "", max_depth: int = 4, current_depth: int = 0):
    """Imprime la estructura de directorios de forma visual (como el comando tree)."""
    if current_depth >= max_depth:
        return

    entries = sorted(root_dir.iterdir())
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]

    # Mostrar conteo de archivos si hay muchos
    if len(files) > 10:
        print(f"{prefix}[{len(files)} archivos]")
    else:
        for f in files:
            print(f"{prefix}{f.name}")

    for d in dirs:
        print(f"{prefix}{d.name}/")
        print_directory_tree(d, prefix + "    ", max_depth, current_depth + 1)


def count_images_by_folder(root_dir: Path) -> dict:
    """Cuenta imágenes por cada subdirectorio."""
    counts = defaultdict(int)
    for root, _, files in os.walk(root_dir):
        for f in files:
            if Path(f).suffix.lower() in VALID_IMAGE_EXTENSIONS:
                # Ruta relativa al dataset para que sea legible
                rel_path = Path(root).relative_to(root_dir)
                counts[str(rel_path)] += 1
    return dict(counts)


def analyze_image_sizes(root_dir: Path, sample_size: int = 100) -> Counter:
    """Analiza los tamaños de imagen más comunes (muestreo para velocidad)."""
    sizes = Counter()
    count = 0

    for root, _, files in os.walk(root_dir):
        for f in files:
            if Path(f).suffix.lower() in VALID_IMAGE_EXTENSIONS:
                filepath = Path(root) / f
                img = cv2.imread(str(filepath))
                if img is not None:
                    h, w = img.shape[:2]
                    sizes[(w, h)] += 1
                    count += 1
                    if count >= sample_size:
                        return sizes
    return sizes


def analyze_annotations(root_dir: Path) -> dict:
    """
    Analiza los archivos de anotación para entender el formato.

    El formato YOLO usa archivos .txt con una línea por objeto:
        class_id center_x center_y width height
    Todos los valores están normalizados entre 0.0 y 1.0.
    """
    stats = {
        'total_annotation_files': 0,
        'format_counts': Counter(),
        'class_ids': Counter(),
        'sample_content': None,
        'files_with_annotations': 0,
        'empty_annotation_files': 0,
    }

    for root, _, files in os.walk(root_dir):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in ANNOTATION_EXTENSIONS:
                filepath = Path(root) / f
                stats['total_annotation_files'] += 1
                stats['format_counts'][ext] += 1

                # Analizar contenido de archivos .txt (formato YOLO)
                if ext == '.txt':
                    content = filepath.read_text().strip()

                    if not content:
                        stats['empty_annotation_files'] += 1
                        continue

                    stats['files_with_annotations'] += 1

                    # Guardar un ejemplo del contenido
                    if stats['sample_content'] is None:
                        stats['sample_content'] = content[:500]

                    # Extraer class_ids (primer número de cada línea)
                    for line in content.split('\n'):
                        parts = line.strip().split()
                        if parts:
                            try:
                                class_id = int(parts[0])
                                stats['class_ids'][class_id] += 1
                            except ValueError:
                                pass

    return stats


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = project_root / 'dataset' / 'raw'

    if not dataset_dir.exists():
        print(f"ERROR: No se encontró el directorio: {dataset_dir}")
        print("Descarga el dataset de Kaggle y colócalo en dataset/raw/")
        sys.exit(1)

    print("=" * 70)
    print("EXPLORACIÓN DEL DATASET GRAZPEDWRI-DX")
    print("=" * 70)

    # 1. Estructura de directorios
    print("\n📁 ESTRUCTURA DE DIRECTORIOS:")
    print("-" * 40)
    print_directory_tree(dataset_dir)

    # 2. Imágenes por carpeta
    print("\n📊 IMÁGENES POR CARPETA:")
    print("-" * 40)
    img_counts = count_images_by_folder(dataset_dir)
    total_images = 0
    for folder, count in sorted(img_counts.items()):
        print(f"  {folder}: {count} imágenes")
        total_images += count
    print(f"  TOTAL: {total_images} imágenes")

    # 3. Formatos de imagen
    print("\n🖼️  FORMATOS DE IMAGEN:")
    print("-" * 40)
    format_counts = Counter()
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in VALID_IMAGE_EXTENSIONS:
                format_counts[ext] += 1
    for fmt, count in format_counts.most_common():
        print(f"  {fmt}: {count}")

    # 4. Tamaños de imagen (muestreo)
    print("\n📐 TAMAÑOS DE IMAGEN (muestreo de 100):")
    print("-" * 40)
    sizes = analyze_image_sizes(dataset_dir)
    for (w, h), count in sizes.most_common(10):
        print(f"  {w}x{h}: {count} imágenes")

    # 5. Anotaciones
    print("\n🏷️  ANOTACIONES:")
    print("-" * 40)
    ann_stats = analyze_annotations(dataset_dir)
    print(f"  Total archivos de anotación: {ann_stats['total_annotation_files']}")
    print(f"  Archivos con objetos: {ann_stats['files_with_annotations']}")
    print(f"  Archivos vacíos (sin objetos): {ann_stats['empty_annotation_files']}")

    print("\n  Formatos encontrados:")
    for fmt, count in ann_stats['format_counts'].most_common():
        print(f"    {fmt}: {count}")

    print(f"\n  Clases encontradas ({len(ann_stats['class_ids'])} distintas):")
    for class_id, count in sorted(ann_stats['class_ids'].items()):
        print(f"    Clase {class_id}: {count} objetos")

    if ann_stats['sample_content']:
        print("\n  Ejemplo de archivo de anotación:")
        print("  " + "-" * 30)
        for line in ann_stats['sample_content'].split('\n')[:5]:
            print(f"    {line}")

    # 6. Resumen
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"  Total imágenes: {total_images}")
    print(f"  Total anotaciones: {ann_stats['total_annotation_files']}")
    print(f"  Clases: {sorted(ann_stats['class_ids'].keys())}")
    print(f"  Formato de anotación principal: "
          f"{ann_stats['format_counts'].most_common(1)[0][0] if ann_stats['format_counts'] else 'N/A'}")
    print("\n  NOTA: Revisa la estructura anterior y adapta los scripts 03-07")
    print("  si la organización del dataset es diferente a la esperada.")


if __name__ == "__main__":
    main()
