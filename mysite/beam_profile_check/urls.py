from django.urls import path

from .views import *

urlpatterns = [
    path('', Index.as_view(), name='index'),
    path('6x/', beam_energy_6x, name='beam_energy_6x'),
    path('6xplot/', beam_energy_6x_display_plot, name='beam_energy_6x_display_plot'),
    path('10x/,', beam_energy_10x, name='beam_energy_10x'),
    path('10xplot/,', beam_energy_10x_display_plot, name='beam_energy_10x_display_plot'),
    path('10fff/,', beam_energy_10fff, name='beam_energy_10fff'),
    path('10fffplot/,', beam_energy_10fff_display_plot, name='beam_energy_10fff_display_plot'),
    path('recent/', most_recent_plot, name='most_recent_plot'),
]