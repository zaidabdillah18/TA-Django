from django.urls import path
from .controllers import wajah

urlpatterns = [
    path('hello', wajah.hello),
    path('predict', wajah.predict),
    path('multi-predict', wajah.multi_predict),
]
