"""
Servicio de interpretación médica con LLM (Qwen vía Ollama).

Equivalente a un @Service en Spring Boot que consume una API REST externa.
Recibe las detecciones del modelo YOLO y genera un texto médico descriptivo
usando un Large Language Model (LLM) local.

Si Ollama no está disponible (ej: en producción sin GPU), genera una
respuesta template basada en las detecciones sin hacer crash.
"""

import time
import logging

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# Prompt del sistema que define el rol del LLM
SYSTEM_PROMPT = """Eres un sistema de apoyo al diagnóstico radiológico especializado en \
radiografías de mano y muñeca pediátrica. Tu función es generar informes radiológicos \
detallados basados en los hallazgos de un modelo de detección YOLOv8.

ESTRUCTURA DEL INFORME (siempre usar este formato):

1. HALLAZGOS RADIOLÓGICOS
   - Para cada detección, indica:
     a) Estructura anatómica afectada (hueso específico: radio distal, cúbito, escafoides,
        semilunar, piramidal, pisiforme, grande, ganchoso, trapecio, trapezoide,
        1er-5to metacarpiano, falanges proximales/medias/distales)
     b) Región topográfica: epífisis, metáfisis, diáfisis, articular
     c) Tipo de hallazgo: fractura transversa, oblicua, espiral, conminuta, en rodete
        (torus), en tallo verde, avulsión; o anomalía ósea, lesión, reacción perióstica, etc.
     d) Nivel de confianza del modelo: bajo (<50%), moderado (50-75%), alto (>75%)
     e) Lateralidad si es identificable

2. EVALUACIÓN GLOBAL
   - Gravedad estimada (leve / moderada / grave)
   - Estructuras adyacentes en riesgo
   - Correlación clínica sugerida

3. RECOMENDACIONES
   - Proyecciones radiográficas adicionales si aplica
   - Necesidad de TAC u otros estudios

4. ADVERTENCIA MÉDICA OBLIGATORIA:
   "NOTA: Este análisis es generado por inteligencia artificial como apoyo al diagnóstico.
   No reemplaza el juicio clínico de un profesional médico certificado. Se recomienda
   evaluación por un radiólogo certificado antes de cualquier decisión terapéutica."

IMPORTANTE: Responde únicamente en español. Basa el análisis SOLO en los datos \
proporcionados. No inventes hallazgos. Si la confianza es baja, indícalo explícitamente.
Las coordenadas bbox normalizadas (cx, cy) indican: cx=0 lado izquierdo de imagen, \
cx=1 lado derecho; cy=0 parte superior (zona distal/dedos), cy=1 parte inferior \
(zona proximal/muñeca)."""


class LLMService:
    """
    Servicio que genera interpretaciones médicas usando un LLM local.

    Comunicación con Ollama via HTTP REST API (POST /api/generate).
    Si Ollama no está disponible, retorna un texto template.
    """

    def __init__(self):
        self._base_url = settings.OLLAMA_BASE_URL
        self._model = settings.OLLAMA_MODEL
        self._timeout = settings.OLLAMA_TIMEOUT

    @property
    def is_available(self) -> bool:
        """Verifica si Ollama está corriendo y responde."""
        try:
            response = requests.get(f"{self._base_url}/api/tags", timeout=3)
            return response.status_code == 200
        except requests.ConnectionError:
            return False
        except requests.Timeout:
            return False

    # ── Mapeo anatómico por coordenadas normalizadas ─────────────────────────
    # Basado en radiografía PA de muñeca/mano pediátrica GRAZPEDWRI-DX.
    # cy=0 → superior de imagen (dedos/falanges distales)
    # cy=1 → inferior de imagen (radio/cúbito proximal)
    # cx=0 → izquierda imagen | cx=1 → derecha imagen

    _REGIONS_Y = [
        (0.15, "falanges distales"),
        (0.32, "falanges medias y proximales"),
        (0.50, "región metacarpiana"),
        (0.68, "huesos del carpo (escafoides, semilunar, piramidal, pisiforme, grande, ganchoso, trapecio, trapezoide)"),
        (1.00, "radio distal y cúbito distal"),
    ]

    _REGIONS_X = [
        (0.22, "borde cubital — 5to metacarpiano/dedo meñique"),
        (0.40, "4to-5to metacarpiano"),
        (0.60, "3er metacarpiano — dedo medio (central)"),
        (0.78, "2do-3er metacarpiano"),
        (1.00, "borde radial — 1er metacarpiano/pulgar, escafoides, radio"),
    ]

    def _get_anatomical_location(self, cx: float, cy: float) -> str:
        """Mapea coordenadas normalizadas a región anatómica."""
        region_y = next(
            (r for threshold, r in self._REGIONS_Y if cy <= threshold),
            "zona proximal"
        )
        region_x = next(
            (r for threshold, r in self._REGIONS_X if cx <= threshold),
            "borde radial"
        )
        return f"{region_y} / {region_x}"

    def interpret(self, detections: list, fracture_detected: bool) -> dict:
        """
        Genera una interpretación médica de los hallazgos.

        Args:
            detections: Lista de objetos Detection del YOLOService.
            fracture_detected: Si se detectó al menos una fractura.

        Returns:
            Diccionario con 'text' (interpretación), 'source' ('llm' o 'template'),
            y 'time_ms' (tiempo de generación).
        """
        # Construir descripción de hallazgos para el LLM
        user_message = self._build_user_message(detections, fracture_detected)

        # Intentar usar Ollama
        try:
            start_time = time.time()
            response = self._call_ollama(user_message)
            elapsed_ms = (time.time() - start_time) * 1000

            if response:
                return {
                    'text': response,
                    'source': 'llm',
                    'model': self._model,
                    'time_ms': round(elapsed_ms, 1),
                }
        except Exception as e:
            logger.warning(f"Ollama no disponible, usando respuesta template: {e}")

        # Fallback: respuesta template sin LLM
        return {
            'text': self._generate_template_response(detections, fracture_detected),
            'source': 'template',
            'model': 'N/A',
            'time_ms': 0,
        }

    def _build_user_message(self, detections: list, fracture_detected: bool) -> str:
        """Construye el mensaje del usuario con los hallazgos formateados."""
        # Separar hallazgos patológicos de marcadores de texto radiográfico
        from .yolo_service import PATHOLOGY_CLASSES
        pathology = [d for d in detections if d.class_name in PATHOLOGY_CLASSES]
        markers = [d for d in detections if d.class_name not in PATHOLOGY_CLASSES]

        if not fracture_detected:
            nota_marcadores = ""
            if markers:
                nota_marcadores = (
                    f"\n(Se detectaron {len(markers)} marcador(es) de texto radiográfico "
                    f"[L/R/fecha], que no son hallazgos médicos.)"
                )
            return (
                "Analiza la siguiente radiografía PA de mano/muñeca pediátrica:\n"
                "El modelo de detección NO encontró hallazgos patológicos en la imagen."
                f"{nota_marcadores}\n"
                "Genera un informe indicando que la radiografía no presenta hallazgos "
                "anormales detectables por el modelo, manteniendo la estructura solicitada."
            )

        lines = [
            "Analiza la siguiente radiografía PA de mano/muñeca pediátrica.\n",
            f"El modelo YOLOv8 identificó {len(pathology)} hallazgo(s) patológico(s):\n",
        ]

        for i, det in enumerate(pathology, 1):
            cx, cy = det.bbox_normalized[0], det.bbox_normalized[1]
            anatomia = self._get_anatomical_location(cx, cy)
            confianza_nivel = (
                "ALTA" if det.confidence > 0.75 else
                "MODERADA" if det.confidence > 0.50 else "BAJA"
            )
            lines.append(
                f"  Hallazgo {i}:\n"
                f"    - Tipo: {det.class_name}\n"
                f"    - Confianza: {det.confidence:.1%} (nivel {confianza_nivel})\n"
                f"    - Posición en imagen: cx={cx}, cy={cy} "
                f"(bbox: {det.bbox[0]},{det.bbox[1]} → {det.bbox[2]},{det.bbox[3]})\n"
                f"    - Región anatómica estimada: {anatomia}\n"
                f"    - Tamaño del bbox: {det.bbox_normalized[2]:.3f}×{det.bbox_normalized[3]:.3f} "
                f"(relativo a imagen)\n"
            )

        if markers:
            lines.append(
                f"\nNota: Se detectaron {len(markers)} marcador(es) de texto radiográfico "
                f"(L/R/fecha) — no son hallazgos médicos, no los menciones en el informe.\n"
            )

        lines.append(
            "\nGenera un informe radiológico completo y estructurado según las instrucciones."
        )
        return "\n".join(lines)

    def _call_ollama(self, user_message: str) -> str:
        """Llama a la API REST de Ollama para generar texto."""
        payload = {
            'model': self._model,
            'prompt': user_message,
            'system': SYSTEM_PROMPT,
            'stream': False,
            'options': {
                'temperature': 0.3,  # Baja temperatura = respuestas más consistentes
                'num_predict': 512,  # Máximo de tokens a generar
            }
        }

        response = requests.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()

        data = response.json()
        return data.get('response', '')

    def _generate_template_response(self, detections: list, fracture_detected: bool) -> str:
        """
        Genera un informe médico en HTML estructurado cuando el LLM no está disponible.
        Retorna HTML que se renderiza con |safe en el template de Django.
        """
        from .yolo_service import PATHOLOGY_CLASSES

        pathology = [d for d in detections if d.class_name in PATHOLOGY_CLASSES]

        tipo_map = {
            'fracture':          ('Fractura',                        'danger'),
            'boneanomaly':       ('Anomalía ósea',                   'warning'),
            'bonelesion':        ('Lesión ósea',                     'warning'),
            'foreignbody':       ('Cuerpo extraño',                  'secondary'),
            'metal':             ('Material metálico (implante)',     'secondary'),
            'periostealreaction':('Reacción perióstica',             'info'),
            'pronationsign':     ('Signo de pronación',              'info'),
            'softtissue':        ('Anomalía tejidos blandos',        'primary'),
        }

        html = ['<div class="medical-report">']

        # ── Cabecera ────────────────────────────────────────────────────────
        html.append('''
        <div class="report-header d-flex align-items-center gap-2 mb-3">
            <i class="bi bi-file-earmark-medical fs-4 text-primary"></i>
            <div>
                <div class="fw-bold text-uppercase small text-muted tracking-wide">
                    Informe de Análisis Radiológico Automatizado
                </div>
                <div class="small text-muted">Sistema FractureAI · YOLOv8</div>
            </div>
        </div>''')

        # ── 1. Hallazgos ────────────────────────────────────────────────────
        html.append('''
        <div class="report-section mb-3">
            <div class="report-section-title">
                <span class="section-num">1</span>
                Hallazgos Radiológicos
            </div>''')

        if not fracture_detected or not pathology:
            html.append('''
            <div class="d-flex align-items-start gap-2 p-3 bg-success bg-opacity-10
                        border border-success border-opacity-25 rounded">
                <i class="bi bi-check-circle-fill text-success mt-1"></i>
                <div>
                    <div class="fw-semibold text-success">Sin hallazgos patológicos detectados</div>
                    <div class="small text-muted mt-1">
                        La radiografía no presenta fracturas, anomalías óseas ni lesiones
                        detectables con el umbral de confianza configurado (≥ 50%).
                    </div>
                </div>
            </div>''')
        else:
            for i, det in enumerate(pathology, 1):
                cx, cy = det.bbox_normalized[0], det.bbox_normalized[1]
                anatomia = self._get_anatomical_location(cx, cy)
                tipo_es, badge_color = tipo_map.get(det.class_name, (det.class_name.capitalize(), 'secondary'))
                w_px = det.bbox[2] - det.bbox[0]
                h_px = det.bbox[3] - det.bbox[1]

                if det.confidence > 0.75:
                    nivel, nivel_color = "ALTA", "danger"
                elif det.confidence > 0.50:
                    nivel, nivel_color = "MODERADA", "warning"
                else:
                    nivel, nivel_color = "BAJA", "secondary"

                conf_pct = int(det.confidence * 100)
                html.append(f'''
            <div class="finding-card border rounded p-3 mb-2">
                <div class="d-flex justify-content-between align-items-start mb-2">
                    <div class="d-flex align-items-center gap-2">
                        <span class="badge bg-{badge_color} px-2 py-1">
                            <i class="bi bi-exclamation-diamond me-1"></i>{tipo_es}
                        </span>
                        <span class="small fw-semibold">Hallazgo #{i}</span>
                    </div>
                    <span class="badge bg-{nivel_color} bg-opacity-75 small">
                        Confianza {nivel}: {conf_pct}%
                    </span>
                </div>
                <div class="progress mb-2" style="height:6px;">
                    <div class="progress-bar bg-{nivel_color}"
                         style="width:{conf_pct}%"></div>
                </div>
                <ul class="list-unstyled small mb-0 ps-1">
                    <li class="mb-1">
                        <i class="bi bi-geo-alt text-muted me-1"></i>
                        <strong>Región anatómica:</strong> {anatomia}
                    </li>
                    <li class="mb-1">
                        <i class="bi bi-bounding-box text-muted me-1"></i>
                        <strong>Extensión en imagen:</strong> {w_px}×{h_px} px
                        &nbsp;·&nbsp; bbox ({det.bbox[0]}, {det.bbox[1]}) → ({det.bbox[2]}, {det.bbox[3]})
                    </li>
                </ul>
            </div>''')

        html.append('</div>')  # cierra report-section hallazgos

        # ── 2. Evaluación global ────────────────────────────────────────────
        html.append('''
        <div class="report-section mb-3">
            <div class="report-section-title">
                <span class="section-num">2</span>
                Evaluación Global
            </div>''')

        if not fracture_detected or not pathology:
            html.append('''
            <p class="small mb-0 text-muted">
                <i class="bi bi-check2-all me-1 text-success"></i>
                Radiografía sin hallazgos anormales detectables por el modelo.
            </p>''')
        else:
            max_conf = max(d.confidence for d in pathology)
            n = len(pathology)
            gravedad = "Grave" if n >= 3 else ("Moderada" if n == 2 else "Leve a moderada")
            grav_color = "danger" if n >= 3 else ("warning" if n == 2 else "info")
            html.append(f'''
            <div class="row g-2 small">
                <div class="col-6">
                    <div class="stat-box border rounded p-2 text-center">
                        <div class="fw-bold fs-5">{n}</div>
                        <div class="text-muted">Zona(s) con hallazgos</div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="stat-box border rounded p-2 text-center">
                        <div class="fw-bold fs-5">{int(max_conf*100)}%</div>
                        <div class="text-muted">Confianza máxima</div>
                    </div>
                </div>
            </div>
            <div class="mt-2 d-flex align-items-center gap-2">
                <span class="badge bg-{grav_color}">Gravedad estimada: {gravedad}</span>
                <span class="small text-muted">
                    Se recomienda correlacionar con la clínica del paciente.
                </span>
            </div>''')

        html.append('</div>')

        # ── 3. Recomendaciones ──────────────────────────────────────────────
        html.append('''
        <div class="report-section mb-3">
            <div class="report-section-title">
                <span class="section-num">3</span>
                Recomendaciones
            </div>
            <ul class="small mb-0 ps-0 list-unstyled">''')

        if fracture_detected and pathology:
            recs = [
                ("bi-camera", "Solicitar proyecciones adicionales (lateral, oblicua) para confirmar hallazgos y evaluar desplazamiento."),
                ("bi-hospital", "Considerar TAC si se sospecha fractura intraarticular o lesión de carpo (escafoides, semilunar)."),
                ("bi-person-badge", "Evaluación urgente por ortopedia pediátrica."),
            ]
        else:
            recs = [
                ("bi-arrow-repeat", "Si persiste dolor o sospecha clínica, considerar proyecciones adicionales o repetir radiografía en 7-10 días."),
                ("bi-person-badge", "Consultar con especialista si los síntomas persisten."),
            ]

        for icon, text in recs:
            html.append(f'''
                <li class="d-flex gap-2 mb-1">
                    <i class="bi {icon} text-primary mt-1 flex-shrink-0"></i>
                    <span>{text}</span>
                </li>''')

        html.append('</ul></div>')

        # ── Advertencia médica ──────────────────────────────────────────────
        html.append('''
        <div class="alert alert-warning border-warning d-flex gap-2 mb-0 mt-2 small">
            <i class="bi bi-shield-exclamation fs-5 flex-shrink-0 text-warning"></i>
            <div>
                <strong>Advertencia médica:</strong>
                Este análisis fue generado automáticamente por inteligencia artificial
                (YOLOv8) como herramienta de apoyo diagnóstico.
                <strong>No reemplaza el juicio clínico</strong> de un profesional médico
                certificado. Se recomienda evaluación por un radiólogo certificado antes
                de cualquier decisión terapéutica.
            </div>
        </div>''')

        html.append('</div>')  # cierra medical-report
        return '\n'.join(html)


# Instancia singleton
llm_service = LLMService()
