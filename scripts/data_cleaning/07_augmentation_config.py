"""
Script 07 — Configurar pipeline de Data Augmentation con Albumentations.

¿Qué es Data Augmentation?
Es la técnica de crear variaciones artificiales de las imágenes de entrenamiento
para que el modelo vea más diversidad sin necesidad de recolectar más datos reales.

¿Por qué es necesario para radiografías?
Las radiografías pueden variar en:
- Rotación: el técnico puede posicionar la mano con ligera inclinación.
- Brillo/contraste: diferentes máquinas de rayos X producen imágenes con
  distintos niveles de exposición.
- Ruido: equipos más antiguos generan más ruido en la imagen.
- Zoom: la distancia entre la mano y el sensor varía.

Si el modelo solo ve imágenes "perfectas", fallará con imágenes reales
que tengan estas variaciones naturales. Augmentation simula estas variaciones.

IMPORTANTE: Las augmentaciones se aplican DURANTE el entrenamiento (on-the-fly),
no como un paso offline. Cada época ve versiones ligeramente diferentes de las
mismas imágenes, lo que actúa como regularización implícita contra overfitting.

Este script define la configuración. YOLO tiene su propio sistema de augmentation
integrado (hsv, degrees, scale, mosaic, mixup), pero Albumentations permite
transformaciones más específicas para imágenes médicas como CLAHE.
"""

import json
from pathlib import Path

import albumentations as A


def create_augmentation_pipeline() -> A.Compose:
    """
    Crea un pipeline de augmentaciones para radiografías.

    Cada transformación tiene una probabilidad (p) de aplicarse.
    No todas se aplican a cada imagen — la combinación aleatoria
    genera gran diversidad.

    Las transformaciones respetan los bounding boxes gracias a
    BboxParams con formato 'yolo'.

    Returns:
        Pipeline de Albumentations configurado para radiografías.
    """
    transform = A.Compose([
        # Rotación leve (±15°) — simula posicionamiento imperfecto de la mano
        A.Rotate(limit=15, p=0.5),

        # Volteo horizontal — una mano izquierda se ve como derecha y viceversa
        # Ambas son anatómicamente válidas para detección de fracturas
        A.HorizontalFlip(p=0.5),

        # Cambio de brillo y contraste — simula diferentes máquinas de rayos X
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Mejora el contraste local de la imagen. En radiografías, esto hace
        # más visibles las líneas de fractura finas que podrían perderse
        # en zonas muy oscuras o muy claras.
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

        # Ruido gaussiano — simula imperfecciones del sensor
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),

        # Desenfoque leve — simula movimiento del paciente durante la toma
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # Zoom aleatorio — simula diferentes distancias al sensor
        A.RandomScale(scale_limit=0.15, p=0.3),

    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,  # Descartar bbox si menos del 30% queda visible
    ))

    return transform


def save_config(output_path: Path):
    """Guarda la configuración como JSON para referencia."""
    config = {
        'description': 'Pipeline de augmentation para radiografías de mano/muñeca',
        'framework': 'albumentations',
        'transforms': [
            {'name': 'Rotate', 'limit': 15, 'p': 0.5},
            {'name': 'HorizontalFlip', 'p': 0.5},
            {'name': 'RandomBrightnessContrast', 'brightness_limit': 0.2, 'contrast_limit': 0.2, 'p': 0.5},
            {'name': 'CLAHE', 'clip_limit': 2.0, 'p': 0.3},
            {'name': 'GaussNoise', 'std_range': [0.02, 0.1], 'p': 0.3},
            {'name': 'GaussianBlur', 'blur_limit': [3, 5], 'p': 0.2},
            {'name': 'RandomScale', 'scale_limit': 0.15, 'p': 0.3},
        ],
        'bbox_params': {
            'format': 'yolo',
            'min_visibility': 0.3,
        },
        'note': 'YOLO tiene su propio augmentation integrado (mosaic, mixup, hsv). '
                'Este pipeline de Albumentations es complementario para '
                'transformaciones específicas de imágenes médicas (CLAHE).'
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    return config


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / 'ml' / 'configs' / 'augmentation_config.json'

    print("=" * 60)
    print("CONFIGURACIÓN DE DATA AUGMENTATION")
    print("=" * 60)

    # Crear y probar el pipeline
    pipeline = create_augmentation_pipeline()
    print(f"  Pipeline creado con {len(pipeline.transforms)} transformaciones")

    # Guardar configuración
    config = save_config(config_path)

    print(f"\n  Transformaciones configuradas:")
    for t in config['transforms']:
        print(f"    - {t['name']} (p={t['p']})")

    print(f"\n  Configuración guardada en: {config_path}")

    print("\n  NOTA: YOLO tiene augmentation integrado que se configura en train.py.")
    print("  Este pipeline de Albumentations es complementario y se puede usar")
    print("  en un dataloader personalizado si se necesita.")


if __name__ == "__main__":
    main()
