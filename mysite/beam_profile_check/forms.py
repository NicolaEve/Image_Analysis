from django import forms
from .models import *


class BeamEnergy6xForm(forms.ModelForm):
    class Meta:
        model = BeamEnergy6x
        fields = ('image', 'title')


class BeamEnergy10xForm(forms.ModelForm):
    class Meta:
        model = BeamEnergy10x
        fields = ('image', 'title')


class BeamEnergy10fffForm(forms.ModelForm):
    class Meta:
        model = BeamEnergy10fff
        fields = ('image', 'title')