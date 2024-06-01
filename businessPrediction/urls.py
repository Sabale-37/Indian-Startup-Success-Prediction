
from django.urls import path
from . import views

urlpatterns = [
   path('', views.prediction_business, name="prediction"),
   path('dashboard/', views.dashboard, name='dashboard')
]