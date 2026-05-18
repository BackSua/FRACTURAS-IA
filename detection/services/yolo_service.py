"""
Servicio de detección de fracturas con YOLOv8 — inferencia ONNX Runtime.

Se usa onnxruntime-cpu en lugar de PyTorch para reducir el consumo de RAM
de ~650 MB (PyTorch + ultralytics) a ~250 MB (onnxruntime + numpy + OpenCV).
Esto permite desplegar en el plan gratuito de Render (512 MB).

La lógica de detección (preprocesamiento, NMS, postprocesamiento) se implementa
manualmente con numpy y OpenCV, sin dependencia de la librería ultralytics.
"""

import time
import logging
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
from django.conf import settings

# Clases del dataset GRAZPEDWRI-DX — mismo orden que en el entrenamiento.
# "text" = marcadores radiográficos (L/R, fecha), NO son fracturas.
CLASS_NAMES = {
    0: 'boneanomaly',
    1: 'bonelesion',
    2: 'foreignbody',
    3: 'fracture',
    4: 'metal',
    5: 'periostealreaction',
    6: 'pronatorsign',
    7: 'softtissue',
    8: 'text',
}

PATHOLOGY_CLASSES = frozenset({
    'fracture', 'boneanomaly', 'bonelesion', 'foreignbody',
    'metal', 'periostealreaction', 'pronatorsign', 'softtissue'
})

# Paleta de colores BGR por clase
COLOR_MAP = {
    'fracture':           (0, 255, 255),
    'boneanomaly':        (0, 165, 255),
    'bonelesion':         (0, 0, 255),
    'foreignbody':        (255, 0, 255),
    'metal':              (128, 128, 128),
    'periostealreaction': (255, 255, 0),
    'pronatorsign':       (255, 128, 0),
    'softtissue':         (0, 255, 128),
}
DEFAULT_COLOR = (0, 255, 255)

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: list          # [x1, y1, x2, y2] píxeles imagen original
    bbox_normalized: list  # [cx, cy, w, h] normalizado 0-1


@dataclass
class YOLOResult:
    detections: list = field(default_factory=list)
    fracture_detected: bool = False
    max_confidence: float = 0.0
    annotated_image_path: str = ''
    original_image_path: str = ''
    inference_time_ms: float = 0.0
    error: str = ''


class YOLOService:
    """
    Singleton que ejecuta inferencia YOLOv8 vía ONNX Runtime.
    El modelo se carga la primera vez que se llama a detect() (lazy loading).
    """

    # Tamaño de entrada esperado por el modelo ONNX
    INPUT_SIZE = 640

    def __init__(self):
        self._session = None
        self._model_path = Path(settings.BASE_DIR) / 'ml' / 'models' / 'best.onnx'

    # ── Carga ──────────────────────────────────────────────────────────────────

    def _load_model(self):
        """Carga la sesión ONNX Runtime (una sola vez)."""
        if not self._model_path.exists():
            logger.warning(f"Modelo ONNX no encontrado en {self._model_path}")
            return
        try:
            import onnxruntime as ort
            # Usar solo CPU para minimizar RAM
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(self._model_path),
                sess_options=opts,
                providers=['CPUExecutionProvider'],
            )
            logger.info(f"Modelo ONNX cargado: {self._model_path}")
        except Exception as e:
            logger.error(f"Error cargando modelo ONNX: {e}")

    def _ensure_loaded(self):
        if self._session is None:
            self._load_model()

    @property
    def is_available(self) -> bool:
        if self._session is not None:
            return True
        return self._model_path.exists()

    # ── Preprocesamiento ───────────────────────────────────────────────────────

    def _preprocess(self, image_path: str):
        """
        Lee la imagen, aplica CLAHE y la convierte al tensor de entrada ONNX.
        Retorna (tensor NCHW float32, escala_x, escala_y, img_original_BGR).
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")

        orig_h, orig_w = img.shape[:2]

        # CLAHE sobre escala de grises para mejorar contraste de radiografías
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # Resize a 640×640 manteniendo aspect ratio con letterbox
        scale = self.INPUT_SIZE / max(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized = cv2.resize(bgr, (new_w, new_h))

        # Pad hasta 640×640 con gris (114)
        canvas = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), 114, dtype=np.uint8)
        pad_top  = (self.INPUT_SIZE - new_h) // 2
        pad_left = (self.INPUT_SIZE - new_w) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        # BGR → RGB → NCHW float32 normalizado [0, 1]
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis]  # (1, 3, 640, 640)

        return tensor, scale, pad_top, pad_left, orig_h, orig_w, img

    # ── NMS ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _nms(boxes, scores, iou_thr=0.45):
        """Non-Maximum Suppression puramente numpy."""
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = np.where(union > 0, inter / union, 0)
            order = order[1:][iou <= iou_thr]
        return keep

    # ── Postprocesamiento ──────────────────────────────────────────────────────

    def _postprocess(self, output, conf_thr, iou_thr, scale, pad_top, pad_left, orig_h, orig_w):
        """
        Parsea la salida ONNX de YOLOv8.
        Output shape: (1, 4+num_classes, 8400)
        """
        pred = output[0]          # (4+num_classes, 8400)
        pred = pred.T             # (8400, 4+num_classes)

        boxes_xywh = pred[:, :4]  # cx, cy, w, h en escala INPUT_SIZE
        class_scores = pred[:, 4:]  # (8400, num_classes)

        max_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        mask = max_scores >= conf_thr
        if not mask.any():
            return []

        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        # cx,cy,w,h → x1,y1,x2,y2 en espacio INPUT_SIZE (con padding)
        cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Revertir letterbox → coordenadas imagen original
        boxes_orig = boxes_xyxy.copy()
        boxes_orig[:, [0, 2]] = (boxes_orig[:, [0, 2]] - pad_left) / scale
        boxes_orig[:, [1, 3]] = (boxes_orig[:, [1, 3]] - pad_top) / scale
        boxes_orig = np.clip(boxes_orig, 0, [orig_w, orig_h, orig_w, orig_h])

        # NMS por clase
        detections = []
        for cls_id in np.unique(class_ids):
            idx = np.where(class_ids == cls_id)[0]
            keep = self._nms(boxes_xyxy[idx], max_scores[idx], iou_thr)
            for k in keep:
                i = idx[k]
                class_name = CLASS_NAMES.get(int(cls_id), f'class_{cls_id}')
                bx1, by1, bx2, by2 = boxes_orig[i]
                cx_n = float((bx1 + bx2) / 2 / orig_w)
                cy_n = float((by1 + by2) / 2 / orig_h)
                w_n  = float((bx2 - bx1) / orig_w)
                h_n  = float((by2 - by1) / orig_h)
                detections.append(Detection(
                    class_name=class_name,
                    confidence=float(max_scores[i]),
                    bbox=[int(bx1), int(by1), int(bx2), int(by2)],
                    bbox_normalized=[round(cx_n, 4), round(cy_n, 4), round(w_n, 4), round(h_n, 4)],
                ))
        return detections

    # ── Anotación visual ───────────────────────────────────────────────────────

    def _annotate(self, img, detections):
        """Dibuja cajas de colores sobre la imagen original (solo clases patológicas)."""
        annotated = img.copy()
        for det in detections:
            if det.class_name not in PATHOLOGY_CLASSES:
                continue
            x1, y1, x2, y2 = det.bbox
            color = COLOR_MAP.get(det.class_name, DEFAULT_COLOR)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = max(y1 - 5, lh + 5)
            cv2.rectangle(annotated, (x1, label_y - lh - baseline), (x1 + lw, label_y + baseline), color, cv2.FILLED)
            cv2.putText(annotated, label, (x1, label_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return annotated

    # ── Detección principal ────────────────────────────────────────────────────

    def detect(self, image_path: str) -> YOLOResult:
        self._ensure_loaded()
        result = YOLOResult(original_image_path=image_path)

        if not self._session:
            result.error = 'Modelo ONNX no disponible.'
            return result

        try:
            start = time.time()
            conf_thr = float(getattr(settings, 'YOLO_CONFIDENCE_THRESHOLD', 0.50))
            iou_thr  = float(getattr(settings, 'YOLO_IOU_THRESHOLD', 0.45))

            tensor, scale, pad_top, pad_left, orig_h, orig_w, orig_img = self._preprocess(image_path)

            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: tensor})

            result.inference_time_ms = (time.time() - start) * 1000

            detections = self._postprocess(
                outputs[0], conf_thr, iou_thr,
                scale, pad_top, pad_left, orig_h, orig_w
            )
            result.detections = detections

            pathology = [d for d in detections if d.class_name in PATHOLOGY_CLASSES]
            result.fracture_detected = len(pathology) > 0
            if pathology:
                result.max_confidence = max(d.confidence for d in pathology)

            annotated = self._annotate(orig_img, detections)
            results_dir = Path(settings.MEDIA_ROOT) / 'results'
            results_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = results_dir / f"annotated_{Path(image_path).name}"
            cv2.imwrite(str(annotated_path), annotated)
            result.annotated_image_path = str(annotated_path)

            logger.info(
                f"Detección ONNX: {len(detections)} total, "
                f"{len(pathology)} patológicos, {result.inference_time_ms:.1f}ms"
            )

        except Exception as e:
            logger.error(f"Error en detección ONNX: {e}", exc_info=True)
            result.error = str(e)

        return result


# Singleton — carga lazy en primera llamada a detect()
yolo_service = YOLOService()
