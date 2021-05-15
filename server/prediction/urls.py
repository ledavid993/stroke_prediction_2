from django.urls import path
import prediction.views as views

urlpatterns = [
    path('model/', views.api_add, name='api_add')
]
