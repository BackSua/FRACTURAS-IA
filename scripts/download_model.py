"""
Script para descargar el modelo entrenado (best.pt) desde almacenamiento en la nube.

Uso en producción (Render):
    python scripts/download_model.py

Este script se ejecuta durante el build de Render si best.pt no está
en el repositorio (por ser muy grande para Git >100MB).

Si best.pt ya existe localmente, no descarga nada.

Configurar la variable de entorno MODEL_DOWNLOAD_URL con la URL directa
de descarga del archivo (Google Drive, Dropbox, etc.).
"""

import os
import sys
import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / 'ml' / 'models'
MODEL_PATH = MODEL_DIR / 'best.pt'

# URL de descarga del modelo (configurar en variables de entorno)
MODEL_URL = os.getenv('MODEL_DOWNLOAD_URL', '')


def download_model():
    """Descarga best.pt si no existe localmente."""

    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        logger.info(f"Modelo ya existe: {MODEL_PATH} ({size_mb:.1f} MB)")
        return

    if not MODEL_URL:
        logger.warning(
            "No se encontró MODEL_DOWNLOAD_URL. "
            "El modelo debe estar en ml/models/best.pt o configurar la URL de descarga."
        )
        return

    logger.info(f"Descargando modelo desde: {MODEL_URL}")

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        response = requests.get(MODEL_URL, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    if downloaded % (1024 * 1024) < 8192:  # Log cada ~1MB
                        logger.info(f"  Progreso: {percent:.1f}%")

        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        logger.info(f"Modelo descargado: {MODEL_PATH} ({size_mb:.1f} MB)")

    except requests.RequestException as e:
        logger.error(f"Error descargando modelo: {e}")
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        sys.exit(1)


if __name__ == "__main__":
    download_model()
