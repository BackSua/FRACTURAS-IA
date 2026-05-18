"""
Configuración de producción para FractureAI (Render.com).

Hereda de settings.py y sobreescribe lo necesario para un entorno
seguro sin interfaz gráfica. Equivalente a application-prod.properties en Spring.
"""

from .settings import *  # noqa: F401,F403

DEBUG = False
ALLOWED_HOSTS = ['.onrender.com']

# WhiteNoise sirve archivos estáticos sin necesidad de Nginx
# Se inserta justo después de SecurityMiddleware
MIDDLEWARE.insert(1, 'whitenoise.middleware.WhiteNoiseMiddleware')
STORAGES = {
    'staticfiles': {
        'BACKEND': 'whitenoise.storage.CompressedManifestStaticFilesStorage',
    },
}

# Seguridad HTTPS
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
