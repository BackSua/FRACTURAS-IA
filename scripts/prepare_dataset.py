"""
Script unificado de preparación del dataset GRAZPEDWRI-DX.

Ejecuta todos los pasos de limpieza en secuencia:
1. Verificar integridad de imágenes
2. Redimensionar a 640x640
3. Dividir en train/val/test (70/20/10)
4. Generar data.yaml

Estructura del dataset descargado:
- dataset/raw/images_part1..4/  → imágenes PNG
- dataset/raw/folder_structure/yolov5/labels/ → archivos .txt YOLO
- 9 clases: boneanomaly, bonelesion, foreignbody, fracture, metal,
            periostealreaction, pronatorsign, softtissue, text
- Clase 3 = fracture
"""

import os
import sys
import random
import shutil
from pathlib import Path

import cv2

RANDOM_SEED = 42
TARGET_SIZE = 640
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / 'dataset' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'dataset' / 'processed'
LABELS_SRC = RAW_DIR / 'folder_structure' / 'yolov5' / 'labels'

CLASS_NAMES = [
    'boneanomaly', 'bonelesion', 'foreignbody', 'fracture',
    'metal', 'periostealreaction', 'pronatorsign', 'softtissue', 'text'
]


def collect_all_images() -> list:
    """Recopila todas las imágenes de las 4 partes."""
    images = []
    for part in range(1, 5):
        part_dir = RAW_DIR / f'images_part{part}'
        if part_dir.exists():
            images.extend(sorted(part_dir.glob('*.png')))
    return images


def verify_and_resize(images: list) -> list:
    """Verifica integridad y redimensiona imágenes. Retorna pares válidos."""
    valid_pairs = []
    corrupt = 0

    for i, img_path in enumerate(images):
        if (i + 1) % 2000 == 0:
            print(f"  Procesadas: {i+1}/{len(images)}...")

        # Verificar que la imagen se puede leer
        img = cv2.imread(str(img_path))
        if img is None:
            corrupt += 1
            continue

        # Verificar que existe su label correspondiente
        label_path = LABELS_SRC / (img_path.stem + '.txt')
        if not label_path.exists():
            continue

        valid_pairs.append((img_path, label_path))

    print(f"  Válidas: {len(valid_pairs)}, Corruptas: {corrupt}")
    return valid_pairs


def split_dataset(pairs: list) -> dict:
    """Divide pares en train/val/test con semilla fija."""
    random.seed(RANDOM_SEED)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    return {
        'train': shuffled[:train_end],
        'val': shuffled[train_end:val_end],
        'test': shuffled[val_end:],
    }


def process_and_save(pairs: list, split_name: str):
    """Redimensiona imágenes y copia labels al directorio destino."""
    img_dir = PROCESSED_DIR / 'images' / split_name
    lbl_dir = PROCESSED_DIR / 'labels' / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in pairs:
        # Redimensionar imagen
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        interp = cv2.INTER_AREA if (h > TARGET_SIZE or w > TARGET_SIZE) else cv2.INTER_LINEAR
        resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=interp)

        cv2.imwrite(str(img_dir / img_path.name), resized)

        # Copiar label (ya está en formato YOLO normalizado, no necesita ajuste)
        shutil.copy2(str(label_path), str(lbl_dir / label_path.name))


def generate_data_yaml():
    """Genera el archivo data.yaml para YOLO."""
    yaml_content = f"""# Dataset GRAZPEDWRI-DX para YOLOv8
# Generado por prepare_dataset.py

path: {PROCESSED_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}

names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"

    output_path = PROJECT_ROOT / 'ml' / 'configs' / 'data.yaml'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_content, encoding='utf-8')
    print(f"  data.yaml generado: {output_path}")


def main():
    print("=" * 60)
    print("PREPARACIÓN DEL DATASET GRAZPEDWRI-DX")
    print("=" * 60)

    # Crear directorio procesado (sobreescribe archivos existentes)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Paso 1: Recopilar imágenes
    print("\n[1/4] Recopilando imágenes...")
    all_images = collect_all_images()
    print(f"  Total imágenes encontradas: {len(all_images)}")

    # Paso 2: Verificar integridad
    print("\n[2/4] Verificando integridad y emparejando con labels...")
    valid_pairs = verify_and_resize(all_images)

    # Paso 3: Dividir dataset
    print("\n[3/4] Dividiendo dataset (70/20/10)...")
    splits = split_dataset(valid_pairs)
    for name, pairs in splits.items():
        print(f"  {name}: {len(pairs)} imágenes")

    # Paso 4: Procesar y guardar
    print("\n[4/4] Redimensionando y guardando...")
    for split_name, pairs in splits.items():
        print(f"  Procesando {split_name}...")
        process_and_save(pairs, split_name)

    # Generar data.yaml
    print("\nGenerando data.yaml...")
    generate_data_yaml()

    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    total = sum(len(p) for p in splits.values())
    print(f"  Total procesadas: {total}")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val:   {len(splits['val'])}")
    print(f"  Test:  {len(splits['test'])}")
    print(f"  Clases: {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES)})")
    print(f"  Tamaño: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"  Directorio: {PROCESSED_DIR}")
    print("\n¡Dataset listo para entrenamiento!")


if __name__ == "__main__":
    main()
