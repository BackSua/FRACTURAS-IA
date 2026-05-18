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

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', RedirectView.as_view(url='/upload/', permanent=False)),
    path('', include('detection.urls')),
]

# En desarrollo, Django sirve archivos media directamente
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
