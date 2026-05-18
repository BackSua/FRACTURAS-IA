"""
Vistas de Django para la app detection.

En Spring Boot, esto sería un @Controller con métodos @GetMapping/@PostMapping.
Django usa Class-Based Views (CBV) que son similares a los controllers de Spring:
cada clase maneja un endpoint, con métodos get() y post().

Flujo principal:
1. UploadXRayView: El médico sube una radiografía.
2. Se procesa con YOLO (detección) + LLM (interpretación).
3. ResultView: Se muestra el resultado al médico.
4. DashboardView: Gráficas estáticas del entrenamiento del modelo.
"""

import csv
import json
import logging
import uuid
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings
from django.shortcuts import render, redirect
from django.views import View

from .forms import XRayUploadForm
from .services.yolo_service import yolo_service
from .services.llm_service import llm_service

logger = logging.getLogger(__name__)


class UploadXRayView(View):
    """
    Vista para subir y analizar una radiografía.

    GET: Muestra el formulario de carga.
    POST: Recibe la imagen, ejecuta YOLO + LLM, redirige a resultado.
    """

    def get(self, request):
        form = XRayUploadForm()
        return render(request, 'detection/upload.html', {
            'form': form,
            'model_available': yolo_service.is_available,
        })

    def post(self, request):
        form = XRayUploadForm(request.POST, request.FILES)

        if not form.is_valid():
            return render(request, 'detection/upload.html', {
                'form': form,
                'model_available': yolo_service.is_available,
            })

        try:
            # Guardar imagen subida temporalmente
            uploaded_image = form.cleaned_data['image']
            image_path = self._save_uploaded_image(uploaded_image)

            # Validar que la imagen parece una radiografía antes de correr YOLO
            xray_error = self._validate_xray(image_path)
            if xray_error:
                import os
                try:
                    os.remove(image_path)
                except Exception:
                    pass
                return render(request, 'detection/upload.html', {
                    'form': form,
                    'model_available': yolo_service.is_available,
                    'error': xray_error,
                })

            # Ejecutar detección YOLO
            yolo_result = yolo_service.detect(str(image_path))

            # Generar interpretación LLM
            llm_result = llm_service.interpret(
                yolo_result.detections,
                yolo_result.fracture_detected,
            )

            # Construir datos del resultado para la sesión
            result_data = {
                'fracture_detected': yolo_result.fracture_detected,
                'max_confidence': round(yolo_result.max_confidence * 100, 1),
                'num_detections': sum(1 for d in yolo_result.detections if d.class_name != 'text'),
                'detections': [
                    {
                        'class_name': d.class_name,
                        'confidence': round(d.confidence * 100, 1),
                        'bbox': d.bbox,
                    }
                    for d in yolo_result.detections
                    if d.class_name != 'text'
                ],
                'inference_time_ms': round(yolo_result.inference_time_ms, 1),
                'annotated_image_url': self._get_media_url(yolo_result.annotated_image_path),
                'original_image_url': self._get_media_url(str(image_path)),
                'llm_interpretation': llm_result['text'],
                'llm_source': llm_result['source'],
                'llm_model': llm_result['model'],
                'llm_time_ms': llm_result['time_ms'],
                'patient_id': form.cleaned_data.get('patient_id', ''),
                'notes': form.cleaned_data.get('notes', ''),
                'error': yolo_result.error,
            }

            # Guardar en sesión para que ResultView pueda accederlo
            request.session['last_result'] = result_data

            return redirect('detection:result')

        except Exception as e:
            logger.error(f"Error procesando radiografía: {e}", exc_info=True)
            return render(request, 'detection/upload.html', {
                'form': form,
                'model_available': yolo_service.is_available,
                'error': f'Error al procesar la imagen: {str(e)}',
            })

    def _save_uploaded_image(self, uploaded_file) -> Path:
        """Guarda la imagen subida en media/uploads/ con nombre único."""
        uploads_dir = Path(settings.MEDIA_ROOT) / 'uploads'
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # Nombre único para evitar colisiones
        ext = Path(uploaded_file.name).suffix
        unique_name = f"{uuid.uuid4().hex[:12]}{ext}"
        dest_path = uploads_dir / unique_name

        with open(dest_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        return dest_path

    def _get_media_url(self, file_path: str) -> str:
        """Convierte una ruta absoluta a URL relativa para el template."""
        if not file_path:
            return ''
        try:
            rel_path = Path(file_path).relative_to(settings.MEDIA_ROOT)
            return f"{settings.MEDIA_URL}{rel_path.as_posix()}"
        except ValueError:
            return ''

    def _validate_xray(self, image_path: Path) -> str:
        """
        Valida que la imagen parece una radiografía de mano/muñeca.
        Retorna mensaje de error si no es válida, cadena vacía si es OK.

        Tres checks con numpy/OpenCV (sin dependencia extra):
          1. Colorfulness < 15  → imagen casi gris; rechaza fotos/caricaturas color
          2. StdDev > 20        → tiene contraste; rechaza imágenes uniformes
          3. AvgGradient < 22   → transiciones suaves; rechaza dibujos con bordes duros

        Las radiografías tienen gradientes promedio bajos (tejidos con transición
        gradual de grises). Los dibujos/grabados tienen muchos bordes abruptos
        que elevan el gradiente promedio aunque tengan grises intermedios.
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return ''  # Dejar que YOLO maneje el error de lectura

            # Reducir a 100×100 para análisis rápido
            small = cv2.resize(img, (100, 100)).astype(np.float32)
            b, g, r = small[:, :, 0], small[:, :, 1], small[:, :, 2]

            # Check 1: Colorfulness (canales R/G/B muy diferentes → imagen a color)
            avg_ch = (r + g + b) / 3.0
            colorfulness = (np.abs(r - avg_ch) + np.abs(g - avg_ch) + np.abs(b - avg_ch)).mean()
            if colorfulness > 15:
                return (
                    'Imagen rechazada: parece una fotografía o imagen a color. '
                    'Este sistema solo acepta radiografías en escala de grises '
                    'de mano y muñeca pediátrica.'
                )

            # Convertir a luminancia para checks 2 y 3
            gray = (r * 0.299 + g * 0.587 + b * 0.114)

            # Check 2: Imagen demasiado uniforme (blanco/negro/gris sólido)
            if gray.std() < 20:
                return (
                    'Imagen rechazada: la imagen parece estar en blanco, negro '
                    'o ser demasiado uniforme para ser una radiografía.'
                )

            # Check 3: Gradiente local promedio
            # X-rays → transiciones suaves → gradiente bajo (~5-18)
            # Dibujos/grabados → muchos bordes duros → gradiente alto (>22)
            grad_h = np.abs(np.diff(gray, axis=1))
            grad_v = np.abs(np.diff(gray, axis=0))
            avg_gradient = (grad_h.mean() + grad_v.mean()) / 2.0
            if avg_gradient > 22:
                return (
                    'Imagen rechazada: parece un dibujo, grabado o esquema gráfico. '
                    'Las radiografías tienen transiciones de gris suaves entre tejidos. '
                    'Sube una radiografía real de mano o muñeca.'
                )

            # Check 4: Ratio de píxeles extremos (negro puro + blanco puro)
            # Dibujos/logos B&W: >65% de píxeles en rango 0-30 o 225-255
            # Radiografías reales: huesos y tejidos producen grises intermedios
            #   → extremeRatio típicamente < 50%
            near_black = np.sum(gray < 30)
            near_white = np.sum(gray > 225)
            extreme_ratio = (near_black + near_white) / gray.size
            if extreme_ratio > 0.65:
                return (
                    'Imagen rechazada: parece un logo, ícono o dibujo en blanco y negro. '
                    'Las radiografías tienen variedad de tonos grises entre tejidos y huesos. '
                    'Sube una radiografía real de mano o muñeca.'
                )

            return ''  # Imagen válida

        except Exception as e:
            logger.warning(f"Error en validación de imagen: {e}")
            return ''  # Ante la duda, dejar pasar y que YOLO evalúe


class ResultView(View):
    """
    Vista que muestra el resultado del análisis.

    Lee los datos de request.session['last_result'] que fueron guardados
    por UploadXRayView después del procesamiento.
    """

    def get(self, request):
        result = request.session.get('last_result')

        if not result:
            return redirect('detection:upload')

        return render(request, 'detection/result.html', {'result': result})


class DashboardView(View):
    """
    Vista del dashboard con gráficas del entrenamiento.

    Los datos provienen de archivos estáticos generados durante el entrenamiento:
    - results.csv: Métricas por época (loss, mAP, precision, recall)
    - evaluation_results.json: Métricas finales sobre el test set

    Estos archivos se commitean en Git y están disponibles en producción.
    Chart.js en el frontend renderiza las gráficas — cero procesamiento
    gráfico en el servidor.
    """

    def get(self, request):
        training_data = self._load_training_data()
        eval_data = self._load_evaluation_data()
        args_data = self._load_training_args()

        context = {
            'training_data_json': json.dumps(training_data),
            'eval_data_json': json.dumps(eval_data),
            'args_data': args_data,
            'has_training_data': bool(training_data.get('epochs')),
            'has_eval_data': bool(eval_data),
        }

        return render(request, 'detection/dashboard.html', context)

    def _load_training_data(self) -> dict:
        """
        Lee results.csv generado por YOLO durante el entrenamiento.

        Este CSV contiene una fila por época con columnas como:
        epoch, train/box_loss, train/cls_loss, val/box_loss, val/cls_loss,
        metrics/precision, metrics/recall, metrics/mAP50, metrics/mAP50-95
        """
        csv_path = Path(settings.BASE_DIR) / 'ml' / 'training_results' / 'results.csv'

        if not csv_path.exists():
            logger.warning(f"No se encontró results.csv en {csv_path}")
            return {}

        data = {
            'epochs': [],
            'train_box_loss': [],
            'train_cls_loss': [],
            'val_box_loss': [],
            'val_cls_loss': [],
            'precision': [],
            'recall': [],
            'mAP50': [],
            'mAP50_95': [],
        }

        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Los nombres de columna pueden tener espacios — strip para limpiar
                    row = {k.strip(): v.strip() for k, v in row.items()}

                    data['epochs'].append(int(row.get('epoch', 0)))
                    data['train_box_loss'].append(round(float(row.get('train/box_loss', 0)), 4))
                    data['train_cls_loss'].append(round(float(row.get('train/cls_loss', 0)), 4))
                    data['val_box_loss'].append(round(float(row.get('val/box_loss', 0)), 4))
                    data['val_cls_loss'].append(round(float(row.get('val/cls_loss', 0)), 4))
                    data['precision'].append(round(float(row.get('metrics/precision(B)', 0)), 4))
                    data['recall'].append(round(float(row.get('metrics/recall(B)', 0)), 4))
                    data['mAP50'].append(round(float(row.get('metrics/mAP50(B)', 0)), 4))
                    data['mAP50_95'].append(round(float(row.get('metrics/mAP50-95(B)', 0)), 4))
        except Exception as e:
            logger.error(f"Error leyendo results.csv: {e}")
            return {}

        return data

    def _load_evaluation_data(self) -> dict:
        """Lee evaluation_results.json con métricas finales del modelo."""
        json_path = Path(settings.BASE_DIR) / 'ml' / 'training_results' / 'evaluation_results.json'

        if not json_path.exists():
            return {}

        try:
            return json.loads(json_path.read_text(encoding='utf-8'))
        except Exception as e:
            logger.error(f"Error leyendo evaluation_results.json: {e}")
            return {}

    def _load_training_args(self) -> dict:
        """Lee args.yaml con los hiperparámetros usados en el entrenamiento."""
        args_path = Path(settings.BASE_DIR) / 'ml' / 'training_results' / 'args.yaml'

        if not args_path.exists():
            return {}

        # Solo mostrar los parámetros relevantes para la presentación
        RELEVANT_KEYS = [
            'model', 'epochs', 'batch', 'imgsz', 'optimizer',
            'lr0', 'lrf', 'patience', 'device', 'seed',
            'weight_decay', 'warmup_epochs', 'augment',
            'mosaic', 'degrees', 'fliplr', 'amp',
        ]

        try:
            raw = {}
            content = args_path.read_text(encoding='utf-8')
            for line in content.split('\n'):
                if ':' in line and not line.strip().startswith('#'):
                    key, _, value = line.partition(':')
                    raw[key.strip()] = value.strip()
            # Devolver solo los parámetros relevantes que existan en el archivo
            return {k: raw[k] for k in RELEVANT_KEYS if k in raw}
        except Exception as e:
            logger.error(f"Error leyendo args.yaml: {e}")
            return {}
