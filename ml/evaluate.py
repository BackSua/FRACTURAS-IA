"""
Script de evaluación del modelo YOLOv8 entrenado sobre el conjunto de test.

El conjunto de test NO se usa durante el entrenamiento. Son imágenes que el
modelo nunca ha visto, lo que permite medir su rendimiento real en producción.

=== MÉTRICAS DE EVALUACIÓN ===

Precision = TP / (TP + FP)
    De las fracturas que el modelo PREDIJO, ¿cuántas eran reales?
    Precision alta = pocas falsas alarmas.

Recall (Sensibilidad) = TP / (TP + FN)
    De las fracturas que REALMENTE EXISTEN, ¿cuántas detectó el modelo?
    Recall alto = pocas fracturas perdidas.
    EN CONTEXTO MÉDICO, RECALL ES LA MÉTRICA MÁS IMPORTANTE:
    un falso negativo (no detectar fractura) puede causar que el paciente
    no reciba tratamiento. Un falso positivo solo requiere una segunda revisión.

F1-Score = 2 · (Precision · Recall) / (Precision + Recall)
    Media armónica de Precision y Recall. Útil cuando hay desbalance de clases.
    F1 penaliza si una de las dos métricas es baja — no se puede "compensar"
    un Recall bajo con Precision alta.

IoU (Intersection over Union) = Área(pred ∩ gt) / Área(pred ∪ gt)
    Mide qué tan bien se alinea la caja predicha con la caja real.
    IoU=1.0 = cajas perfectamente superpuestas.
    IoU=0.0 = cajas sin superposición.

mAP@50 (mean Average Precision @ IoU=0.50)
    Promedio de AP (area bajo curva precision-recall) sobre todas las clases,
    considerando una detección como correcta si IoU ≥ 0.50.

mAP@50:95
    Promedio de mAP calculado con umbrales IoU de 0.50 a 0.95 (step 0.05).
    Es la métrica más estricta: exige que las cajas sean muy precisas.
"""

import json
import time
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'ml' / 'models' / 'best.pt'
DATA_YAML = PROJECT_ROOT / 'ml' / 'configs' / 'data.yaml'
OUTPUT_PATH = PROJECT_ROOT / 'ml' / 'training_results' / 'evaluation_results.json'


def evaluate():
    """Evalúa el modelo entrenado sobre el conjunto de test."""

    if not MODEL_PATH.exists():
        logger.error(f"No se encontró el modelo: {MODEL_PATH}")
        logger.error("Ejecuta primero: python ml/train.py")
        return

    if not DATA_YAML.exists():
        logger.error(f"No se encontró data.yaml: {DATA_YAML}")
        return

    logger.info("=" * 60)
    logger.info("EVALUACIÓN DEL MODELO SOBRE CONJUNTO DE TEST")
    logger.info("=" * 60)

    model = YOLO(str(MODEL_PATH))

    # Evaluar sobre el split 'test' definido en data.yaml
    start_time = time.time()
    metrics = model.val(
        data=str(DATA_YAML),
        split='test',
        imgsz=640,
        conf=0.50,
        iou=0.45,
        verbose=True,
    )
    eval_time = time.time() - start_time

    # Extraer métricas
    results = {
        'model': str(MODEL_PATH.name),
        'dataset': str(DATA_YAML),
        'evaluation_time_seconds': round(eval_time, 2),
        'metrics': {
            'mAP50': round(float(metrics.box.map50), 4),
            'mAP50_95': round(float(metrics.box.map), 4),
            'precision': round(float(metrics.box.mp), 4),
            'recall': round(float(metrics.box.mr), 4),
            'f1_score': round(
                2 * (float(metrics.box.mp) * float(metrics.box.mr))
                / (float(metrics.box.mp) + float(metrics.box.mr) + 1e-8),
                4
            ),
        },
        'per_class': {},
        'config': {
            'confidence_threshold': 0.50,
            'iou_threshold': 0.45,
            'image_size': 640,
        }
    }

    # Métricas por clase si están disponibles
    if hasattr(metrics.box, 'ap50') and metrics.box.ap50 is not None:
        class_names = model.names
        for i, name in class_names.items():
            if i < len(metrics.box.ap50):
                results['per_class'][name] = {
                    'ap50': round(float(metrics.box.ap50[i]), 4),
                    'ap50_95': round(float(metrics.box.ap[i]), 4) if i < len(metrics.box.ap) else None,
                }

    # Guardar resultados
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

    # Imprimir resumen
    logger.info("\nRESULTADOS:")
    logger.info(f"  mAP@50:     {results['metrics']['mAP50']}")
    logger.info(f"  mAP@50:95:  {results['metrics']['mAP50_95']}")
    logger.info(f"  Precision:  {results['metrics']['precision']}")
    logger.info(f"  Recall:     {results['metrics']['recall']}")
    logger.info(f"  F1-Score:   {results['metrics']['f1_score']}")
    logger.info(f"  Tiempo:     {results['evaluation_time_seconds']}s")
    logger.info(f"\nResultados guardados en: {OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate()
