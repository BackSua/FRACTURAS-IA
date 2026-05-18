"""
Configuración de Django para el proyecto FractureAI.

Este archivo es el equivalente a application.properties en Spring Boot.
Centraliza toda la configuración: base de datos, apps instaladas,
middleware, archivos estáticos, media, logging, etc.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env (equivalente a @PropertySource en Spring)
load_dotenv()

# Directorio raíz del proyecto (donde está manage.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# --- SEGURIDAD ---
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'dev-secret-key-change-in-production')
DEBUG = os.getenv('DEBUG', 'True').lower() in ('true', '1', 'yes')
ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')


# --- APLICACIONES INSTALADAS ---
# Equivalente a las dependencias inyectadas por Spring Boot auto-configuration
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # App propia del proyecto
    'detection',
]


# --- MIDDLEWARE ---
# Equivalente a los filtros/interceptores de Spring (Filter, HandlerInterceptor)
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'fracture_ai.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'fracture_ai.wsgi.application'


# --- BASE DE DATOS ---
# Se usa SQLite por defecto solo para el sistema interno de Django.
# NO se crean modelos propios — los resultados se manejan en memoria/sesión.
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Sesiones basadas en cookies firmadas (no necesita tabla en BD)
SESSION_ENGINE = 'django.contrib.sessions.backends.signed_cookies'


# --- VALIDACIÓN DE CONTRASEÑAS ---
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]


# --- INTERNACIONALIZACIÓN ---
LANGUAGE_CODE = 'es-co'
TIME_ZONE = 'America/Bogota'
USE_I18N = True
USE_TZ = True


# --- ARCHIVOS ESTÁTICOS (CSS, JS, imágenes del tema) ---
# Equivalente a resources/static/ en Spring Boot
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = []


# --- ARCHIVOS MEDIA (subidos por usuarios) ---
# Equivalente a configurar un multipart resolver + directorio de uploads en Spring
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'


# --- TAMAÑO MÁXIMO DE ARCHIVO ---
# 10 MB máximo para radiografías
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024


# --- LOGGING ---
# Equivalente a configurar Logback/Log4j en Spring Boot
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{asctime}] {levelname} {name}: {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'fractureai.log',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
    'loggers': {
        'detection': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG' if DEBUG else 'INFO',
            'propagate': False,
        },
    },
}


# --- CONFIGURACIÓN DEL MODELO YOLO ---
YOLO_MODEL_PATH = BASE_DIR / 'ml' / 'models' / 'best.pt'  # local (desarrollo)
YOLO_ONNX_PATH  = BASE_DIR / 'ml' / 'models' / 'best.onnx'
YOLO_CONFIDENCE_THRESHOLD = 0.50
YOLO_IOU_THRESHOLD = 0.45
YOLO_IMAGE_SIZE = 640


# --- CONFIGURACIÓN DE OLLAMA (LLM) ---
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')
OLLAMA_TIMEOUT = 30


DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
