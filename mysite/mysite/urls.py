"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
#import mysite_root.beam_profile_check.views as views
from beam_profile_check import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('beam_profile_check.urls')),
    path('10x/', views.beam_energy_10x),
    path('10xplot/', views.beam_energy_10x_display_plot),
    path('10fff/', views.beam_energy_10fff),
    path('10fffplot/', views.beam_energy_10fff_display_plot)
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)