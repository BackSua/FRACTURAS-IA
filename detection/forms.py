"""
Formularios de Django para la app detection.

Equivalente a un DTO con @Valid en Spring Boot: define qué datos
se esperan del usuario y cómo validarlos antes de procesarlos.
"""

from django import forms


# Formatos de imagen permitidos para radiografías
ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png']
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


class XRayUploadForm(forms.Form):
    """
    Formulario para subir una radiografía.

    No es ModelForm porque no hay modelo de BD asociado.
    Valida que el archivo sea una imagen JPEG/PNG de máximo 10MB.
    """

    image = forms.ImageField(
        label='Radiografía',
        help_text=f'Formatos aceptados: JPEG, PNG. Máximo {MAX_FILE_SIZE_MB}MB.',
        widget=forms.ClearableFileInput(attrs={
            'accept': 'image/jpeg,image/png',
            'class': 'form-control',
            'id': 'xray-input',
        })
    )

    patient_id = forms.CharField(
        label='ID del paciente',
        max_length=50,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Ej: PAC-001 (opcional)',
        })
    )

    notes = forms.CharField(
        label='Notas clínicas',
        max_length=500,
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Observaciones adicionales (opcional)',
        })
    )

    def clean_image(self):
        """Valida tipo y tamaño del archivo de imagen."""
        image = self.cleaned_data.get('image')

        if image:
            # Validar tipo MIME
            if image.content_type not in ALLOWED_IMAGE_TYPES:
                raise forms.ValidationError(
                    f'Formato no soportado: {image.content_type}. '
                    f'Use JPEG o PNG.'
                )

            # Validar tamaño
            if image.size > MAX_FILE_SIZE_BYTES:
                size_mb = image.size / (1024 * 1024)
                raise forms.ValidationError(
                    f'Archivo demasiado grande ({size_mb:.1f}MB). '
                    f'Máximo permitido: {MAX_FILE_SIZE_MB}MB.'
                )

        return image
