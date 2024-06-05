
from django.urls import path
from . import views

urlpatterns = [
   path('', views.prediction_business, name="prediction"),
   path('dashboard/', views.dashboard, name='dashboard'),
   path('churn/', views.churn, name='churn'),
   path('churn_dashboard/', views.predict_employee_status, name='churn_dashboard'),
   path('sentiment/', views.sentiment, name='sentiment')

]