from django.db import models

# Create your models here.

class Image(models.Model):
    filepath = models.CharField(max_length=120)
    photo = models.ImageField()