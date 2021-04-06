from django import forms
from .models import Image

class UploadForm(forms.Form):
    file = forms.ImageField()
    ignore_character_identification = forms.BooleanField(required=False)

