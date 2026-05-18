"""
URL router principal de FractureAI.

Equivalente a un DispatcherServlet o @RequestMapping a nivel de aplicación
en Spring Boot. Delega las rutas a la app detection.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView
from django.views.static import serve

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', RedirectView.as_view(url='/upload/', permanent=False)),
    path('', include('detection.urls')),
    # Servir archivos media tanto en desarrollo como en producción
    # (Render tiene filesystem efímero — las imágenes se pierden al reiniciar,
    # pero se sirven correctamente durante la sesión activa)
    path('media/<path:path>', serve, {'document_root': settings.MEDIA_ROOT}),
]
