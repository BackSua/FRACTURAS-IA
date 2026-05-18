"""
Servicio de detección de fracturas con YOLOv8.

Equivalente a un @Service en Spring Boot: encapsula la lógica de negocio
de inferencia con YOLO. No conoce Django ni HTTP — recibe una ruta de
imagen y retorna un resultado estructurado.

El modelo se carga UNA SOLA VEZ al importar el módulo (singleton).
Cargar el modelo toma ~2-5 segundos; hacerlo por request haría la app
inutilizable. En Spring Boot sería un @Bean singleton.
"""

import time
import logging
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
from django.conf import settings

# Clases del dataset GRAZPEDWRI-DX que representan hallazgos patológicos reales.
# "text" es excluido — son marcadores radiográficos (letras L/R, fecha, etc.)
# colocados por el radiólogo en la placa, NO son fracturas.
PATHOLOGY_CLASSES = frozenset({
    'fracture', 'boneanomaly', 'bonelesion', 'foreignbody',
    'metal', 'periostealreaction', 'pronationsign', 'softtissue'
})

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Una detección individual de fractura."""
    class_name: str
    confidence: float
    bbox: list  # [x1, y1, x2, y2] en píxeles
    bbox_normalized: list  # [cx, cy, w, h] normalizado 0-1


@dataclass
class YOLOResult:
    """Resultado completo de una inferencia YOLO."""
    detections: list = field(default_factory=list)
    fracture_detected: bool = False
    max_confidence: float = 0.0
    annotated_image_path: str = ''
    original_image_path: str = ''
    inference_time_ms: float = 0.0
    error: str = ''


class YOLOService:
    """
    Servicio singleton que ejecuta inferencia YOLO sobre radiografías.

    Se instancia una sola vez a nivel de módulo. El modelo se mantiene
    en RAM durante toda la vida del proceso Django.
    """

    def __init__(self):
        self._model = None
        self._model_path = settings.YOLO_MODEL_PATH
        self._load_model()

    def _load_model(self):
        """Carga el modelo YOLOv8 desde el archivo .pt."""
        if not Path(self._model_path).exists():
            logger.warning(
                f"Modelo YOLO no encontrado en {self._model_path}. "
                f"La detección no estará disponible hasta que se entrene el modelo."
            )
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self._model_path))
            logger.info(f"Modelo YOLO cargado: {self._model_path}")
        except Exception as e:
            logger.error(f"Error cargando modelo YOLO: {e}")

    @property
    def is_available(self) -> bool:
        """Indica si el modelo está cargado y listo."""
        return self._model is not None

    def _preprocess_image(self, image_path: str) -> str:
        """
        Preprocesa la imagen para mejorar la compatibilidad con el modelo.

        Aplica conversión a escala de grises + CLAHE (ecualización de histograma
        adaptativa) para normalizar imágenes que provienen de fuentes externas
        (fotos de pantalla, JPEG, imágenes a color).

        Retorna la ruta de la imagen preprocesada (temporal).
        """
        img = cv2.imread(image_path)
        if img is None:
            return image_path  # No se pudo leer, devolver original

        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE: mejora contraste local, clave para radiografías
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Normalizar intensidad al rango 0-255
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

        # Volver a BGR para que YOLO lo procese correctamente
        bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # Guardar en archivo temporal junto a la imagen original
        tmp_path = str(Path(image_path).parent / f"_proc_{Path(image_path).name}")
        cv2.imwrite(tmp_path, bgr)
        return tmp_path

    def detect(self, image_path: str) -> YOLOResult:
        """
        Ejecuta detección de fracturas sobre una imagen.

        Args:
            image_path: Ruta absoluta a la imagen de radiografía.

        Returns:
            YOLOResult con las detecciones, imagen anotada y métricas.
        """
        result = YOLOResult(original_image_path=image_path)

        if not self.is_available:
            result.error = 'El modelo YOLO no está disponible. Entrene el modelo primero.'
            return result

        preprocessed_path = None
        try:
            start_time = time.time()

            # Preprocesar imagen (CLAHE + escala de grises)
            preprocessed_path = self._preprocess_image(image_path)

            # Ejecutar predicción sobre imagen preprocesada
            predictions = self._model.predict(
                source=preprocessed_path,
                conf=settings.YOLO_CONFIDENCE_THRESHOLD,
                iou=settings.YOLO_IOU_THRESHOLD,
                imgsz=settings.YOLO_IMAGE_SIZE,
                verbose=False,
            )

            result.inference_time_ms = (time.time() - start_time) * 1000

            pred = predictions[0]

            # Parsear dimensiones de la imagen original (no la preprocesada)
            img = cv2.imread(image_path)
            img_h, img_w = img.shape[:2] if img is not None else (640, 640)

            for box in pred.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detection = Detection(
                    class_name=self._model.names[cls_id],
                    confidence=conf,
                    bbox=[round(x1), round(y1), round(x2), round(y2)],
                    bbox_normalized=[
                        round((x1 + x2) / 2 / img_w, 4),
                        round((y1 + y2) / 2 / img_h, 4),
                        round((x2 - x1) / img_w, 4),
                        round((y2 - y1) / img_h, 4),
                    ]
                )
                result.detections.append(detection)

            # ── Determinar si hay fractura ──────────────────────────────────
            # Solo se considera fractura si hay detección de clase PATOLÓGICA.
            # La clase "text" son marcadores radiográficos (L, R, fecha),
            # no constituyen un hallazgo médico.
            pathology_detections = [
                d for d in result.detections if d.class_name in PATHOLOGY_CLASSES
            ]
            result.fracture_detected = len(pathology_detections) > 0
            if pathology_detections:
                result.max_confidence = max(d.confidence for d in pathology_detections)

            # ── Generar imagen anotada SOLO con hallazgos patológicos ─────────
            # Se usa la imagen ORIGINAL (no preprocesada) para mejor calidad visual.
            # Se dibujan manualmente solo las cajas de PATHOLOGY_CLASSES,
            # omitiendo los marcadores radiográficos ("text").
            annotated_img = cv2.imread(image_path)
            if annotated_img is None:
                annotated_img = pred.plot()  # fallback si no se puede leer
            else:
                # Paleta de colores por clase (BGR)
                COLOR_MAP = {
                    'fracture':          (0, 255, 255),   # Cyan
                    'boneanomaly':       (0, 165, 255),   # Naranja
                    'bonelesion':        (0, 0, 255),     # Rojo
                    'foreignbody':       (255, 0, 255),   # Magenta
                    'metal':             (128, 128, 128), # Gris
                    'periostealreaction':(255, 255, 0),   # Amarillo
                    'pronationsign':     (255, 128, 0),   # Azul claro
                    'softtissue':        (0, 255, 128),   # Verde claro
                }
                DEFAULT_COLOR = (0, 255, 255)

                for box in pred.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self._model.names[cls_id]

                    # Omitir marcadores de texto radiográfico
                    if class_name not in PATHOLOGY_CLASSES:
                        continue

                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    color = COLOR_MAP.get(class_name, DEFAULT_COLOR)

                    # Rectángulo con grosor 2
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

                    # Etiqueta con fondo sólido para legibilidad
                    label = f"{class_name} {conf:.2f}"
                    (lw, lh), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    label_y = max(y1 - 5, lh + 5)
                    cv2.rectangle(
                        annotated_img,
                        (x1, label_y - lh - baseline),
                        (x1 + lw, label_y + baseline),
                        color, cv2.FILLED
                    )
                    cv2.putText(
                        annotated_img, label,
                        (x1, label_y - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                    )

            results_dir = Path(settings.MEDIA_ROOT) / 'results'
            results_dir.mkdir(parents=True, exist_ok=True)

            annotated_filename = f"annotated_{Path(image_path).name}"
            annotated_path = results_dir / annotated_filename
            cv2.imwrite(str(annotated_path), annotated_img)
            result.annotated_image_path = str(annotated_path)

            logger.info(
                f"Detección completada: {len(result.detections)} hallazgos "
                f"({len(pathology_detections)} patológicos), "
                f"{result.inference_time_ms:.1f}ms"
            )

        except Exception as e:
            logger.error(f"Error en detección YOLO: {e}", exc_info=True)
            result.error = str(e)

        finally:
            # Limpiar imagen temporal preprocesada
            if preprocessed_path and preprocessed_path != image_path:
                try:
                    Path(preprocessed_path).unlink(missing_ok=True)
                except Exception:
                    pass

        return result


# Instancia singleton — se crea al importar el módulo
yolo_service = YOLOService()
