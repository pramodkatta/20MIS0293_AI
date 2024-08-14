from django.urls import path
from .views import TrademarkClassPredictionView

urlpatterns = [
    path('predict/', TrademarkClassPredictionView.as_view(), name='predict'),
]
