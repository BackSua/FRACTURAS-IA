"""
Configuración de producción para FractureAI (Render.com).

Hereda de settings.py y sobreescribe lo necesario para un entorno
seguro sin interfaz gráfica. Equivalente a application-prod.properties en Spring.
"""

import os
from .settings import *  # noqa: F401,F403

DEBUG = False

# Render asigna un subdominio único (fractureai.onrender.com).
# Acepta cualquier subdominio de onrender.com + localhost para health checks.
_allowed = os.getenv('ALLOWED_HOSTS', '.onrender.com')
ALLOWED_HOSTS = [h.strip() for h in _allowed.split(',')]
# Agregar explícitamente el subdominio de Render y wildcard para cubrir cualquier variante
ALLOWED_HOSTS += ['localhost', '127.0.0.1', 'fracturas-ia.onrender.com', '.onrender.com']

# WhiteNoise sirve archivos estáticos sin necesidad de Nginx
# Se inserta justo después de SecurityMiddleware
MIDDLEWARE.insert(1, 'whitenoise.middleware.WhiteNoiseMiddleware')
STORAGES = {
    'staticfiles': {
        # CompressedStaticFilesStorage: comprime archivos pero NO usa manifest.
        # Evita el 500 que ocurre cuando collectstatic corre sin settings_prod
        # y el manifest (staticfiles.json) no existe en runtime.
        'BACKEND': 'whitenoise.storage.CompressedStaticFilesStorage',
    },
}

# Seguridad HTTPS
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True

# En producción solo loguear a consola (stdout/stderr capturado por Render).
# El FileHandler de settings.py falla si el filesystem no tiene permisos de escritura.
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
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'detection': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}
