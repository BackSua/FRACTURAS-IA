"""
Script 06 — Generar archivo data.yaml para YOLOv8.

Este archivo le dice a YOLO:
1. Dónde están las imágenes de train/val/test.
2. Cuántas clases hay y cómo se llaman.

YOLOv8 lee este archivo antes de empezar a entrenar para saber
qué datos usar y cuántas neuronas necesita en la capa de salida
(una por clase).
"""

import sys
from pathlib import Path


def detect_classes(labels_dir: Path) -> dict:
    """
    Detecta las clases presentes en los archivos de anotación.

    Lee los archivos .txt de labels y extrae los class_ids únicos.
    Retorna un mapeo {id: nombre}. Si solo se encuentra una clase,
    se asume que es 'fracture'.

    En GRAZPEDWRI-DX las clases comunes son:
    - 0: fracture (fractura)
    Pero pueden existir más clases dependiendo de la versión del dataset.
    """
    class_ids = set()

    for label_file in labels_dir.rglob('*.txt'):
        content = label_file.read_text().strip()
        for line in content.split('\n'):
            parts = line.strip().split()
            if parts:
                try:
                    class_ids.add(int(parts[0]))
                except ValueError:
                    continue

    # Mapear IDs a nombres
    # Ajustar estos nombres según lo que 02_explore_dataset.py revele
    default_names = {
        0: 'fracture',
        1: 'bone_anomaly',
        2: 'metal',
        3: 'periosteal_reaction',
        4: 'soft_tissue',
    }

    names = {}
    for cid in sorted(class_ids):
        names[cid] = default_names.get(cid, f'class_{cid}')

    return names


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = project_root / 'dataset' / 'processed'
    output_path = project_root / 'ml' / 'configs' / 'data.yaml'

    labels_dir = dataset_dir / 'labels'
    if not labels_dir.exists():
        print(f"ERROR: No se encontró {labels_dir}")
        print("Ejecuta primero los scripts 01-05.")
        sys.exit(1)

    print("=" * 60)
    print("GENERACIÓN DE data.yaml")
    print("=" * 60)

    # Detectar clases
    class_names = detect_classes(labels_dir)
    print(f"  Clases detectadas: {class_names}")

    # Construir contenido del YAML
    # Se usan rutas absolutas para evitar problemas con directorios de trabajo
    yaml_content = f"""# Configuración del dataset para YOLOv8
# Generado automáticamente por 06_generate_data_yaml.py

path: {dataset_dir.as_posix()}
train: images/train
val: images/val
test: images/test

# Número de clases
nc: {len(class_names)}

# Nombres de las clases (el índice debe coincidir con el class_id en los .txt)
names:
"""

    for cid, name in sorted(class_names.items()):
        yaml_content += f"  {cid}: {name}\n"

    # Escribir archivo
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_content, encoding='utf-8')

    print(f"\n  Archivo generado: {output_path}")
    print(f"\n  Contenido:")
    print("  " + "-" * 40)
    for line in yaml_content.strip().split('\n'):
        print(f"    {line}")


if __name__ == "__main__":
    main()
