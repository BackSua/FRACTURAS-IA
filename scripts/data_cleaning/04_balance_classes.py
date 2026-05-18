"""
Script 04 — Balancear clases del dataset.

¿Por qué balancear?
Si hay 8000 imágenes sin fractura y 2000 con fractura, el modelo aprende
que la respuesta "no fractura" es correcta el 80% del tiempo. Esto genera
un modelo con alto accuracy pero bajo recall: detecta pocas fracturas reales.

En contexto médico, un falso negativo (no detectar una fractura que sí existe)
es mucho más grave que un falso positivo (marcar fractura donde no hay).
Balancear las clases obliga al modelo a prestar igual atención a ambas.

Estrategia: Undersampling de la clase mayoritaria.
Se reduce la clase con más muestras para igualar la clase con menos.
Se prefiere undersampling sobre oversampling porque:
1. No introduce datos duplicados que podrían causar overfitting.
2. Reduce el tiempo de entrenamiento.
3. Es más simple y menos propenso a errores.
"""

import os
import sys
import random
from pathlib import Path
from collections import defaultdict


# Semilla fija para reproducibilidad
RANDOM_SEED = 42


def classify_images(images_dir: Path, labels_dir: Path) -> dict:
    """
    Clasifica imágenes en 'con fractura' y 'sin fractura'.

    Una imagen tiene fractura si su archivo de anotación .txt contiene
    al menos una línea (un bounding box). Si el archivo está vacío o
    no existe, se considera sin fractura.

    Args:
        images_dir: Directorio con las imágenes.
        labels_dir: Directorio con las anotaciones YOLO (.txt).

    Returns:
        Diccionario con listas de rutas: {'fracture': [...], 'normal': [...]}
    """
    classification = {'fracture': [], 'normal': []}
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}

    for img_file in images_dir.iterdir():
        if img_file.suffix.lower() not in image_extensions:
            continue

        # Buscar archivo de anotación correspondiente (mismo nombre, extensión .txt)
        label_file = labels_dir / (img_file.stem + '.txt')

        if label_file.exists():
            content = label_file.read_text().strip()
            if content:
                classification['fracture'].append(img_file)
            else:
                classification['normal'].append(img_file)
        else:
            classification['normal'].append(img_file)

    return classification


def undersample(file_list: list, target_count: int, seed: int = RANDOM_SEED) -> list:
    """
    Selecciona aleatoriamente target_count elementos de la lista.

    Usa semilla fija para que el resultado sea reproducible:
    ejecutar el script dos veces produce exactamente el mismo subset.
    """
    random.seed(seed)
    return random.sample(file_list, min(target_count, len(file_list)))


def remove_unselected(all_files: list, selected_files: set, labels_dir: Path):
    """Elimina las imágenes y sus anotaciones que no fueron seleccionadas."""
    removed = 0
    for img_file in all_files:
        if img_file not in selected_files:
            img_file.unlink(missing_ok=True)
            label_file = labels_dir / (img_file.stem + '.txt')
            label_file.unlink(missing_ok=True)
            removed += 1
    return removed


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / 'dataset' / 'processed'

    # Buscar directorios de imágenes y labels
    # La estructura puede variar — intentar rutas comunes
    images_dir = None
    labels_dir = None

    # Intentar estructura plana
    for candidate_img in [processed_dir / 'images', processed_dir]:
        if candidate_img.exists():
            images_dir = candidate_img
            break

    for candidate_lbl in [processed_dir / 'labels', processed_dir]:
        if candidate_lbl.exists():
            labels_dir = candidate_lbl
            break

    if images_dir is None:
        print(f"ERROR: No se encontró directorio de imágenes en {processed_dir}")
        print("Ejecuta primero 03_resize_images.py")
        sys.exit(1)

    print("=" * 60)
    print("BALANCEO DE CLASES")
    print("=" * 60)

    classification = classify_images(images_dir, labels_dir)

    print("\nESTADÍSTICAS ANTES DEL BALANCEO:")
    print(f"  Con fractura:  {len(classification['fracture'])}")
    print(f"  Sin fractura:  {len(classification['normal'])}")
    print(f"  Total:         {len(classification['fracture']) + len(classification['normal'])}")

    # Determinar clase minoritaria
    min_count = min(len(classification['fracture']), len(classification['normal']))

    if len(classification['fracture']) > len(classification['normal']):
        majority_class = 'fracture'
    else:
        majority_class = 'normal'

    print(f"\n  Clase mayoritaria: {majority_class}")
    print(f"  Target por clase:  {min_count}")

    # Hacer undersampling de la clase mayoritaria
    selected_majority = undersample(classification[majority_class], min_count)
    selected_majority_set = set(selected_majority)

    removed = remove_unselected(
        classification[majority_class],
        selected_majority_set,
        labels_dir
    )

    print(f"\nESTADÍSTICAS DESPUÉS DEL BALANCEO:")
    print(f"  Con fractura:  {min_count}")
    print(f"  Sin fractura:  {min_count}")
    print(f"  Total:         {min_count * 2}")
    print(f"  Eliminadas:    {removed}")


if __name__ == "__main__":
    main()
