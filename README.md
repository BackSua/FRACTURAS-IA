# FractureAI — Detección de Fracturas con Inteligencia Artificial

Sistema web para detección automática de fracturas en radiografías de mano y muñeca usando redes neuronales convolucionales (YOLOv8) con interpretación asistida por un modelo de lenguaje (Qwen/Ollama).

**Proyecto Final de Inteligencia Artificial** — CORHUILA, Ingeniería de Sistemas, Semestre 2026-1.

## 🚀 Aplicación en producción

**[https://fracturas-ia.onrender.com](https://fracturas-ia.onrender.com)**

> ⚠️ El servidor puede tardar ~30 segundos en responder la primera vez si estuvo inactivo (Render free tier).

---

## Tecnologías

| Componente | Tecnología |
|-----------|-----------|
| Lenguaje | Python 3.11 |
| Framework Web | Django 5.2 |
| Red Neuronal | YOLOv8 (Ultralytics) |
| LLM Local | Qwen 2.5 vía Ollama |
| Procesamiento de Imagen | OpenCV, Pillow |
| Data Augmentation | Albumentations |
| Gráficas (frontend) | Chart.js 4 (CDN) |
| Servidor WSGI | Gunicorn |
| Despliegue | Render.com |

## Dataset

**GRAZPEDWRI-DX** — Radiografías pediátricas de muñecas y manos con anotaciones de fracturas.  
Fuente: [Kaggle](https://www.kaggle.com/datasets/jasonroggy/grazpedwri-dx)

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd fractureai
```

### 2. Crear entorno virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con los valores apropiados
```

### 5. Ejecutar migraciones de Django

```bash
python manage.py migrate
```

### 6. Ejecutar la aplicación

```bash
python manage.py runserver
```

Abrir en el navegador: http://localhost:8000

---

## Preparación del Dataset

1. Descargar GRAZPEDWRI-DX desde Kaggle y colocar en `dataset/raw/`

2. Ejecutar los scripts de limpieza en orden:

```bash
python scripts/data_cleaning/02_explore_dataset.py   # Explorar estructura
python scripts/data_cleaning/01_verify_images.py      # Verificar integridad
python scripts/data_cleaning/03_resize_images.py      # Redimensionar a 640x640
python scripts/data_cleaning/04_balance_classes.py     # Balancear clases
python scripts/data_cleaning/05_split_dataset.py       # Dividir train/val/test
python scripts/data_cleaning/06_generate_data_yaml.py  # Generar config YOLO
python scripts/data_cleaning/07_augmentation_config.py # Config de augmentation
```

## Entrenamiento del Modelo

```bash
python ml/train.py
```

El entrenamiento genera `ml/models/best.pt` y copia las métricas a `ml/training_results/`.

### Evaluación

```bash
python ml/evaluate.py
```

### Predicción individual

```bash
python ml/predict.py --image path/to/radiografia.png
```

---

## Integración con LLM (Ollama)

Para la interpretación textual de hallazgos:

```bash
# Instalar Ollama (https://ollama.com)
ollama pull qwen2.5:7b
ollama serve
```

Si Ollama no está disponible, el sistema usa una respuesta template automática.

---

## Estructura del Proyecto

```
fractureai/
├── fracture_ai/          # Configuración global Django
├── detection/            # App principal
│   ├── services/         # YOLO + LLM services
│   ├── templates/        # HTML (base, upload, result, dashboard)
│   └── static/           # CSS, JS (Chart.js)
├── ml/                   # Machine Learning
│   ├── train.py          # Entrenamiento
│   ├── evaluate.py       # Evaluación
│   ├── predict.py        # Predicción standalone
│   ├── configs/          # data.yaml
│   ├── models/           # best.pt
│   └── training_results/ # Métricas (CSV, JSON)
├── scripts/
│   └── data_cleaning/    # Pipeline de limpieza (7 scripts)
├── dataset/              # Dataset (no incluido en Git)
└── requirements.txt
```

---

## Despliegue en Render

1. Push del repositorio a GitHub
2. Crear nuevo Web Service en Render
3. Configurar variables de entorno:
   - `DJANGO_SETTINGS_MODULE=fracture_ai.settings_prod`
   - `DJANGO_SECRET_KEY` (auto-generada)
   - `DEBUG=False`
4. Deploy automático desde el branch main

---

## Créditos

- **Dataset:** GRAZPEDWRI-DX (Medical University of Graz)
- **Modelo:** YOLOv8 por Ultralytics
- **LLM:** Qwen 2.5 por Alibaba Cloud, servido con Ollama
- **Framework:** Django Software Foundation
- **Gráficas:** Chart.js

---

## Licencia

Proyecto académico. Uso exclusivamente educativo.
