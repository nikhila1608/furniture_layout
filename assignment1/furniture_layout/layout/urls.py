from django.urls import path
from . import views

urlpatterns = [
    # Existing route for optimize
    path('optimize/', views.predict_layout, name='optimize'),  # Update this to point to the correct view
]
