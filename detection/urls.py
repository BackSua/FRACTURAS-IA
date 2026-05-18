"""
Rutas URL de la app detection.

Equivalente a @RequestMapping en Spring Boot. Mapea cada URL a su vista.
"""

from django.urls import path
from . import views

app_name = 'detection'

urlpatterns = [
    path('upload/', views.UploadXRayView.as_view(), name='upload'),
    path('result/', views.ResultView.as_view(), name='result'),
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
]
