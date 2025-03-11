from django.contrib import admin
from django.urls import path, include  # Don't forget to include "include"

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('layout.urls')),  # Make sure you include the layout app's URLs here
]
