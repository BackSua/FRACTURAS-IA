"""
Script 05 — Dividir dataset en train/val/test (70/20/10).

¿Por qué dividir?
- Train (70%): El modelo aprende de estos datos. Ajusta sus pesos internos
  para minimizar el error en estas imágenes.
- Validation (20%): Se usa DURANTE el entrenamiento para medir si el modelo
  generaliza o solo memoriza. Early stopping se basa en val loss.
- Test (10%): Se usa UNA SOLA VEZ al final para evaluar el rendimiento real.
  Nunca se toca durante el entrenamiento.

Regla crítica: NUNCA debe haber una misma imagen en dos conjuntos.
Si una imagen aparece en train y test, el modelo ya la "vio" durante
el entrenamiento y las métricas de test serían artificialmente altas
(data leakage). Usamos semilla fija (42) para reproducibilidad.
"""

import sys
import shutil
import random
from pathlib import Path


TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
RANDOM_SEED = 42


def collect_image_label_pairs(images_dir: Path, labels_dir: Path) -> list:
    """
    Recopila pares (imagen, label) donde ambos archivos existen.

    Args:
        images_dir: Directorio con imágenes.
        labels_dir: Directorio con archivos .txt de anotación.

    Returns:
        Lista de tuplas (imagen_path, label_path).
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    pairs = []

    for img_file in sorted(images_dir.iterdir()):
        if img_file.suffix.lower() not in image_extensions:
            continue

        label_file = labels_dir / (img_file.stem + '.txt')
        # Incluir imagen aunque no tenga label (imagen sin objetos)
        pairs.append((img_file, label_file if label_file.exists() else None))

    return pairs


def split_list(items: list, train_r: float, val_r: float, seed: int) -> tuple:
    """
    Divide una lista en tres subconjuntos con las proporciones dadas.

    Usa shuffle con semilla fija para que la división sea siempre la misma.
    """
    random.seed(seed)
    shuffled = items.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_r)
    val_end = train_end + int(n * val_r)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def move_pairs(pairs: list, dest_images: Path, dest_labels: Path) -> int:
    """Mueve pares imagen/label al directorio destino."""
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    moved = 0
    for img_path, label_path in pairs:
        shutil.copy2(str(img_path), str(dest_images / img_path.name))
        if label_path and label_path.exists():
            shutil.copy2(str(label_path), str(dest_labels / label_path.name))
        else:
            # Crear label vacío para imágenes sin anotación
            (dest_labels / (img_path.stem + '.txt')).write_text('')
        moved += 1

    return moved


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / 'dataset' / 'processed'
    output_dir = processed_dir

    # Buscar directorios fuente (pueden variar según estructura del dataset)
    images_dir = None
    labels_dir = None

    for candidate in [processed_dir / 'images', processed_dir]:
        if candidate.exists() and any(candidate.glob('*.png')) or any(candidate.glob('*.jpg')):
            images_dir = candidate
            break

    for candidate in [processed_dir / 'labels', processed_dir]:
        if candidate.exists() and any(candidate.glob('*.txt')):
            labels_dir = candidate
            break

    if images_dir is None:
        print(f"ERROR: No se encontraron imágenes en {processed_dir}")
        print("Ejecuta primero los scripts 01-04.")
        sys.exit(1)

    if labels_dir is None:
        labels_dir = images_dir

    print("=" * 60)
    print("DIVISIÓN DEL DATASET (Train/Val/Test)")
    print("=" * 60)
    print(f"  Proporciones: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    print(f"  Semilla: {RANDOM_SEED}")

    pairs = collect_image_label_pairs(images_dir, labels_dir)
    print(f"  Total de pares imagen/label: {len(pairs)}")

    if len(pairs) == 0:
        print("ERROR: No se encontraron imágenes.")
        sys.exit(1)

    # Dividir
    train_pairs, val_pairs, test_pairs = split_list(pairs, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)

    print(f"\n  Train: {len(train_pairs)} ({len(train_pairs)/len(pairs)*100:.1f}%)")
    print(f"  Val:   {len(val_pairs)} ({len(val_pairs)/len(pairs)*100:.1f}%)")
    print(f"  Test:  {len(test_pairs)} ({len(test_pairs)/len(pairs)*100:.1f}%)")

    # Mover a directorios finales
    for split_name, split_pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        dest_img = output_dir / 'images' / split_name
        dest_lbl = output_dir / 'labels' / split_name
        moved = move_pairs(split_pairs, dest_img, dest_lbl)
        print(f"  Movidos a {split_name}: {moved}")

    # Verificar que no hay duplicados entre conjuntos
    train_names = {p[0].stem for p in train_pairs}
    val_names = {p[0].stem for p in val_pairs}
    test_names = {p[0].stem for p in test_pairs}

    overlap_tv = train_names & val_names
    overlap_tt = train_names & test_names
    overlap_vt = val_names & test_names

    if overlap_tv or overlap_tt or overlap_vt:
        print("\n⚠️  ADVERTENCIA: Se detectaron duplicados entre conjuntos!")
        print(f"  Train∩Val: {len(overlap_tv)}, Train∩Test: {len(overlap_tt)}, Val∩Test: {len(overlap_vt)}")
    else:
        print("\n✅ Verificación: NO hay imágenes duplicadas entre conjuntos.")

    print(f"\n  Directorio de salida: {output_dir}")


if __name__ == "__main__":
    main()
