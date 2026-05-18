"""
Script principal de entrenamiento del modelo YOLOv8 para detección de fracturas.

=== CONCEPTOS FUNDAMENTALES DE REDES NEURONALES ===

¿Qué es una Red Neuronal Convolucional (CNN)?
Es un tipo de red neuronal diseñada para procesar imágenes. Aprende a extraer
características (features) relevantes de la imagen — bordes, texturas, formas —
organizadas en capas jerárquicas: las primeras capas detectan patrones simples
(líneas, esquinas) y las capas profundas combinan esos patrones en conceptos
complejos (huesos, fracturas).

¿Qué es YOLO (You Only Look Once)?
YOLO es una arquitectura de detección de objetos que procesa toda la imagen
en un solo forward pass (una sola pasada por la red). A diferencia de métodos
como R-CNN que analizan la imagen por regiones, YOLO divide la imagen en una
grilla y predice simultáneamente las coordenadas, clase y confianza de todos
los objetos. Esto lo hace extremadamente rápido (~100ms por imagen).

¿Qué es Transfer Learning?
En vez de entrenar desde cero (millones de imágenes + semanas de GPU), se
toma un modelo ya entrenado en un dataset general (COCO, con 80 clases de
objetos comunes) y se "afina" con nuestro dataset específico (radiografías).
Las capas iniciales ya saben detectar bordes y texturas — solo necesitamos
que las capas finales aprendan qué es una fractura.

=== CONCEPTOS DE ENTRENAMIENTO ===

¿Qué es una Época (Epoch)?
Una pasada completa por TODAS las imágenes del dataset de entrenamiento.
Si tenemos 1000 imágenes, una época = 1000 imágenes procesadas.

¿Qué es un Batch?
Un grupo de imágenes que se procesan juntas antes de actualizar los pesos.
Con batch=16, se procesan 16 imágenes, se calcula el error promedio, y se
actualizan los pesos UNA vez. Batches más grandes dan gradientes más estables
pero requieren más memoria (RAM/VRAM).

¿Qué es un Forward Pass?
La imagen entra por las capas de la red (de izquierda a derecha) y se
obtiene una predicción. No se modifican pesos, solo se calcula la salida.

¿Qué es un Backward Pass (Backpropagation)?
Se calcula el error entre la predicción y la verdad (ground truth), y se
propaga ese error "hacia atrás" por la red para calcular cuánto contribuyó
cada peso al error total. Esto produce un gradiente por cada peso.

¿Qué es el Gradiente?
Es la derivada parcial del error respecto a cada peso. Indica la dirección
y magnitud en la que se debe ajustar cada peso para reducir el error.
Es como una brújula que señala "para reducir el error, mueve este peso
en esta dirección y esta cantidad".

=== OPTIMIZADOR AdamW ===

AdamW combina tres ideas:
1. Momentum (promedio móvil del gradiente): suaviza las actualizaciones
   para no oscilar en direcciones ruidosas.
2. RMSprop (promedio móvil del gradiente al cuadrado): adapta el learning
   rate por parámetro — pesos que cambian mucho se actualizan más lento.
3. Weight Decay desacoplado: penaliza pesos grandes directamente,
   a diferencia de L2 regularization que lo hace a través del gradiente.

Fórmulas de actualización:
    m_t = β1 · m_{t-1} + (1 − β1) · g_t            # Momento 1: promedio exponencial del gradiente
    v_t = β2 · v_{t-1} + (1 − β2) · g_t²            # Momento 2: promedio exponencial del gradiente²
    m̂_t = m_t / (1 − β1^t)                          # Corrección de sesgo del momento 1
    v̂_t = v_t / (1 − β2^t)                          # Corrección de sesgo del momento 2
    θ_{t+1} = θ_t − lr · m̂_t / (√v̂_t + ε) − lr · λ · θ_t

Donde:
    g_t = gradiente de la pérdida respecto al parámetro θ
    β1 = 0.9   → qué tan rápido "olvida" gradientes pasados (momento 1)
    β2 = 0.999 → qué tan rápido "olvida" magnitudes pasadas (momento 2)
    ε = 1e-8   → evita división por cero (estabilidad numérica)
    λ = weight_decay → fuerza de regularización (penaliza pesos grandes)
    lr = learning rate → tamaño del paso de actualización

=== FUNCIÓN DE PÉRDIDA DE YOLO ===

YOLO usa una pérdida compuesta:
    L_total = λ_box · L_CIoU + λ_cls · L_BCE + λ_dfl · L_DFL

1. CIoU Loss (Complete Intersection over Union) — Localización:
    IoU = Área(A ∩ B) / Área(A ∪ B)
    CIoU = IoU − ρ²(b, b_gt) / c² − αv
    donde:
        ρ = distancia euclidiana entre centros de las cajas
        c = diagonal de la caja envolvente mínima que contiene ambas
        v = (4/π²)(arctan(w_gt/h_gt) − arctan(w/h))²  (similitud de aspecto)
        α = v / (1 − IoU + v)
    CIoU penaliza no solo la superposición, sino también la distancia
    entre centros y la diferencia de proporciones. Esto hace que la red
    aprenda cajas más precisas más rápido que con IoU simple.

2. BCE Loss (Binary Cross-Entropy) — Clasificación:
    L_BCE = −[y · log(ŷ) + (1−y) · log(1−ŷ)]
    Mide qué tan segura está la predicción. Si la red dice 0.99 para
    fractura y la imagen tiene fractura (y=1), el loss es casi 0.
    Si dice 0.01 para fractura pero hay fractura, el loss es enorme.

3. DFL (Distribution Focal Loss) — Refinamiento de bordes:
    Modela la posición de los bordes del bounding box como una distribución
    de probabilidad, no como un punto fijo. Esto captura la incertidumbre
    en los bordes de la fractura.

=== NON-MAXIMUM SUPPRESSION (NMS) ===

Después de la predicción, YOLO puede detectar la misma fractura múltiples
veces con cajas ligeramente diferentes. NMS elimina las cajas redundantes:
1. Ordena todas las detecciones por confianza (mayor primero).
2. Toma la caja con mayor confianza y la mantiene.
3. Calcula IoU entre esa caja y todas las demás.
4. Elimina las cajas con IoU > umbral (0.45) — son duplicados.
5. Repite con la siguiente caja más confiable.
Resultado: una sola caja por objeto detectado.

=== EARLY STOPPING ===

Monitorea la pérdida en validación. Si después de N épocas (patience=20)
no mejora, detiene el entrenamiento. Previene overfitting: el modelo
podría seguir mejorando en train (memorizando) mientras empeora en val
(perdiendo capacidad de generalizar a imágenes nuevas).
"""

import shutil
import logging
from pathlib import Path

import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# === CONFIGURACIÓN ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / 'ml' / 'configs' / 'data.yaml'
OUTPUT_DIR = PROJECT_ROOT / 'ml' / 'models'
TRAINING_RESULTS_DIR = PROJECT_ROOT / 'ml' / 'training_results'

# Modelo base preentrenado en COCO (transfer learning)
# yolov8m = "medium" — balance entre velocidad y precisión
# Alternativas: yolov8n (nano, rápido), yolov8s (small), yolov8l (large), yolov8x (extra large)
BASE_MODEL = 'yolov8m.pt'

# --- Hiperparámetros ---
EPOCHS = 50            # Número de pasadas completas sobre el dataset (50 para RTX 3050 4GB)
BATCH_SIZE = 8         # Imágenes por lote (8 para GPUs con 4GB VRAM como RTX 3050)
IMAGE_SIZE = 640       # Tamaño de entrada en píxeles (estándar YOLO)
PATIENCE = 20          # Early stopping: épocas sin mejora antes de detener
OPTIMIZER = 'AdamW'    # Optimizador con weight decay desacoplado
LR_INITIAL = 0.001     # Learning rate inicial
LR_FINAL = 0.01        # Learning rate final (fracción del inicial)
MOMENTUM = 0.937       # Coeficiente de momentum para el optimizador
WEIGHT_DECAY = 0.0005  # Regularización L2: penaliza pesos grandes → previene overfitting
SEED = 42              # Semilla para reproducibilidad


def check_gpu() -> str:
    """Detecta si hay GPU disponible y retorna el device apropiado."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU detectada: {gpu_name} ({gpu_mem:.1f} GB)")
        return '0'
    else:
        logger.warning("No se detectó GPU. Entrenando en CPU (será lento).")
        logger.warning("Considera reducir EPOCHS a 30-50 y BATCH_SIZE a 8.")
        return 'cpu'


def train():
    """Ejecuta el entrenamiento completo del modelo YOLOv8."""

    # Verificar que existe data.yaml
    if not DATA_YAML.exists():
        logger.error(f"No se encontró {DATA_YAML}")
        logger.error("Ejecuta primero: python scripts/data_cleaning/06_generate_data_yaml.py")
        return

    device = check_gpu()

    logger.info("=" * 60)
    logger.info("INICIANDO ENTRENAMIENTO DE YOLOv8")
    logger.info("=" * 60)
    logger.info(f"  Modelo base: {BASE_MODEL} (transfer learning desde COCO)")
    logger.info(f"  Dataset: {DATA_YAML}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Épocas: {EPOCHS}")
    logger.info(f"  Batch: {BATCH_SIZE}")
    logger.info(f"  Imagen: {IMAGE_SIZE}x{IMAGE_SIZE}")

    # Cargar modelo preentrenado
    # Los pesos de COCO se descargan automáticamente (~50MB)
    model = YOLO(BASE_MODEL)

    # Entrenar
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        optimizer=OPTIMIZER,
        lr0=LR_INITIAL,
        lrf=LR_FINAL,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        save=True,
        save_period=10,
        device=device,
        seed=SEED,
        deterministic=True,
        project=str(PROJECT_ROOT / 'runs'),
        name='fracture_detection',
        exist_ok=True,

        # Data augmentation integrado en YOLO
        # Estas transformaciones se aplican on-the-fly durante el entrenamiento
        hsv_h=0.015,      # Variación de hue (tono de color)
        hsv_s=0.7,        # Variación de saturación
        hsv_v=0.4,        # Variación de valor (brillo)
        degrees=15.0,     # Rotación aleatoria ±15°
        translate=0.1,    # Traslación aleatoria ±10%
        scale=0.5,        # Zoom aleatorio ±50%
        fliplr=0.5,       # Volteo horizontal con 50% de probabilidad
        mosaic=0.0,       # Desactivado: crea imagen 1280x1280 que agota RAM en equipos con poca memoria
        mixup=0.0,        # Desactivado: evitar problemas de memoria
        workers=4,        # Reducir workers de datos para ahorrar RAM (default 8)
    )

    # Copiar el mejor modelo a la ubicación final
    runs_dir = PROJECT_ROOT / 'runs' / 'fracture_detection'
    best_pt = runs_dir / 'weights' / 'best.pt'

    if best_pt.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = OUTPUT_DIR / 'best.pt'
        shutil.copy2(str(best_pt), str(dest))
        logger.info(f"Modelo guardado en: {dest}")
    else:
        logger.error(f"No se encontró best.pt en {best_pt}")

    # Copiar resultados del entrenamiento para el dashboard
    TRAINING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_csv = runs_dir / 'results.csv'
    if results_csv.exists():
        shutil.copy2(str(results_csv), str(TRAINING_RESULTS_DIR / 'results.csv'))

    args_yaml = runs_dir / 'args.yaml'
    if args_yaml.exists():
        shutil.copy2(str(args_yaml), str(TRAINING_RESULTS_DIR / 'args.yaml'))

    confusion_matrix = runs_dir / 'confusion_matrix.png'
    if confusion_matrix.exists():
        shutil.copy2(str(confusion_matrix), str(TRAINING_RESULTS_DIR / 'confusion_matrix.png'))

    logger.info("Entrenamiento completado.")
    logger.info(f"Resultados copiados a: {TRAINING_RESULTS_DIR}")


if __name__ == "__main__":
    train()
