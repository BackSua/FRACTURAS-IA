"""
Modelos de Django para la app detection.

ESTE ARCHIVO ESTÁ INTENCIONALMENTE VACÍO.

¿Por qué no se usa base de datos?
Este proyecto funciona como una herramienta de análisis bajo demanda:
se sube una imagen, se analiza con YOLO + LLM, se muestra el resultado.
No hay usuarios registrados, no hay historial persistente, no hay
entidades que modelar.

En Spring Boot, sería como tener un @RestController sin @Entity ni
JPA Repository — un servicio stateless que recibe una imagen, la
procesa, y retorna un resultado.

Los resultados de cada análisis se manejan en memoria durante la sesión
del request. Las gráficas del dashboard se alimentan de archivos CSV/JSON
estáticos generados durante el entrenamiento, no de una base de datos.

Django mantiene su SQLite interna para sesiones y admin, pero no se
crean modelos propios.
"""
