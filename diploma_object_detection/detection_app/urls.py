from django.urls import path
from .views import register_view, home_view, detect_objects, logout_view, video_view, process_video, process_webcam
from django.contrib.auth.views import LoginView


urlpatterns = [
    path("", home_view, name='home'),
    path('register/', register_view, name='register'),
    path('login/', LoginView.as_view(template_name='detection_app/login.html'), name='login'),
    path('detect/', detect_objects, name='detect_objects'),
    path('logout/', logout_view, name='logout'),
    path('video/', video_view, name='video_view'),
    path('process_video/', process_video, name='process_video'),
    path('process_webcam/', process_webcam, name='process_webcam'),
]

