from django.urls import path
from .views import upload_audio, home  # Import the home function

urlpatterns = [
    path('upload/', upload_audio, name='upload_audio'),
    path('input/', home, name='input'),
]
