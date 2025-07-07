from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    # Додаємо свої поля (опціонально)
    bio = models.TextField(blank=True, null=True)
    # Наприклад, можна ще: avatar = models.ImageField(...)

    def __str__(self):
        return self.username
