"""
WSGI config for fracture_ai project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

# Si no hay DJANGO_SETTINGS_MODULE explícito en el entorno,
# detectar producción por la ausencia de DEBUG=True y usar settings_prod.
# Esto garantiza que Render cargue el módulo correcto aunque el env var
# no esté configurado en el dashboard.
_is_dev = os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes')
_default_settings = 'fracture_ai.settings' if _is_dev else 'fracture_ai.settings_prod'
os.environ.setdefault('DJANGO_SETTINGS_MODULE', _default_settings)

application = get_wsgi_application()
