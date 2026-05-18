web: python manage.py collectstatic --noinput && gunicorn fracture_ai.wsgi:application --workers 1 --timeout 120 --bind 0.0.0.0:$PORT
