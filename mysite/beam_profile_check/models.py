"""
Models defines the database objects for the app.

Author: Nicola Compton
Date: 24th May 2021
Contact: nicola.compton@ulh.nhs.uk
"""

from django.db import models
from .main import *
import numpy as np

#class Index(models.Model):
# need to make a model for the index page to avoid query improperly configured error


class BeamEnergy6x(models.Model):
    """ Class to process 6x file upload """
    image = models.FileField(upload_to='images/')
    title = models.CharField(max_length=200)

    def __str__(self):
        return self.title

    class Meta:
        db_table = "beam_profile_check_image_6x"


class BeamEnergy10x(models.Model):
    """ Class to process 10x file upload """
    image = models.FileField(upload_to='images/')
    title = models.CharField(max_length=200)

    def __str__(self):
        return self.title

    class Meta:
        db_table = "beam_profile_check_image_10x"


class BeamEnergy10fff(models.Model):
    """ Class to process 10fff file upload """
    image = models.FileField(upload_to='images/')
    title = models.CharField(max_length=200)

    def __str__(self):
        return self.title

    class Meta:
        db_table = "beam_profile_check_image_10fff"




