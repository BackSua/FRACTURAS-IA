"""
Script de predicción standalone para testing rápido.

Uso:
    python ml/predict.py --image path/to/radiografia.png
    python ml/predict.py --image path/to/radiografia.png --model ml/models/best.pt
    python ml/predict.py --image path/to/radiografia.png --conf 0.3

Muestra la imagen con bounding boxes y guarda la versión anotada.
"""

import argparse
import time
import logging
from pathlib import Path

import cv2
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / 'ml' / 'models' / 'best.pt'


def predict(image_path: str, model_path: str, confidence: float = 0.50):
    """
    Ejecuta predicción sobre una imagen y muestra resultados.

    Args:
        image_path: Ruta a la imagen de radiografía.
        model_path: Ruta al archivo .pt del modelo entrenado.
        confidence: Umbral de confianza mínimo (0.0 a 1.0).
    """
    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.exists():
        logger.error(f"No se encontró la imagen: {image_path}")
        return

    if not model_path.exists():
        logger.error(f"No se encontró el modelo: {model_path}")
        return

    # Cargar modelo
    model = YOLO(str(model_path))

    # Ejecutar inferencia
    start_time = time.time()
    results = model.predict(
        source=str(image_path),
        conf=confidence,
        iou=0.45,
        imgsz=640,
        verbose=False,
    )
    inference_time = (time.time() - start_time) * 1000  # ms

    result = results[0]

    # Mostrar detecciones
    logger.info("=" * 60)
    logger.info("RESULTADOS DE PREDICCIÓN")
    logger.info("=" * 60)
    logger.info(f"  Imagen: {image_path.name}")
    logger.info(f"  Tiempo de inferencia: {inference_time:.1f} ms")

    if len(result.boxes) == 0:
        logger.info("  Resultado: No se detectaron fracturas.")
    else:
        logger.info(f"  Detecciones: {len(result.boxes)}")
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            class_name = model.names[cls_id]
            logger.info(f"    [{i+1}] {class_name}: {conf:.2%} — "
                        f"bbox=({coords[0]:.0f}, {coords[1]:.0f}, "
                        f"{coords[2]:.0f}, {coords[3]:.0f})")

    # Guardar imagen anotada
    annotated = result.plot()
    output_path = image_path.parent / f"{image_path.stem}_annotated{image_path.suffix}"
    cv2.imwrite(str(output_path), annotated)
    logger.info(f"\n  Imagen anotada guardada en: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Predicción de fracturas con YOLOv8')
    parser.add_argument('--image', type=str, required=True, help='Ruta a la imagen de radiografía')
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL), help='Ruta al modelo .pt')
    parser.add_argument('--conf', type=float, default=0.50, help='Umbral de confianza (default: 0.50)')
    args = parser.parse_args()

    predict(args.image, args.model, args.conf)


if __name__ == "__main__":
    main()
