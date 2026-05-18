"""
Configuración de la app detection.

Equivalente a la anotación @SpringBootApplication que configura
el arranque y escaneo de componentes en Spring Boot.
"""

from django.apps import AppConfig


class DetectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detection'
    verbose_name = 'Detección de Fracturas'
