"""
Models defines the database objects for the app.

This script also contains code for symmetry and flatness metrics.

Author: Nicola Compton
Date: 24th May 2021
Contact: nicola.compton@ulh.nhs.uk
"""

from django.db import models
from django import forms
from . import run_calibration
from .main import Edges, Image, Profiles
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



class TransformView(models.Model):
    """class to apply the transformation matrices onto new images """

    def __init__(self, energy, filename):
        """ energy is which beam we are transforming, 6x, 10x, or 10xfff,
         filename is the newly uploaded image """
        self.energy = energy
        self.filename = filename

    def get_matrix(self):
        """ Extract the calibration matrix for the corresponding beam energy """
        if self.energy == "6x":
            matrix = run_calibration._6x_matrix
        if self.energy == "10x":
            matrix = run_calibration._10x_matrix
        if self.energy == "10fff":
            matrix = run_calibration._10fff_matrix

        return matrix

    def get_original_centre(self):
        """ Find centre of field in calibration image """
        if self.energy == "6x":
            centre = run_calibration._6x_sobel.get_centre()
        if self.energy == "10x":
            centre = run_calibration._10x_sobel.get_centre()
        if self.energy == "10fff":
            centre = run_calibration._10x_sobel.get_centre()

        return centre

    def normalise_again(self, x_array, y_array):
        """ Normalise the array by setting f(0)=1 i.e. dividing all values by the value at f(0)"""

        # find closest x value to 0 in the original array
        x_array = np.asarray(x_array)
        index = (np.abs(x_array - 0)).argmin()

        # divide all values in the array by f(0)
        normalised_array = [value / y_array[index] for value in y_array]

        return x_array, normalised_array

    def process_profile(self, profile, centre_cood):
        """ Input: profile = x or y profile; centre_cood = corresponding x or y central co-ordinate
            Output: the shifted (w.r.t. centre) and normalised arrays, measured in distance """

        # shift it to be centred at 0 and convert to cm, using MPC EPID resolution
        xs = np.linspace(0, len(profile), len(profile))
        shifted_xs = [(x - int(centre_cood)) * 0.0336 for x in xs]
        # normalise profile
        x_array, normalised_array= self.normalise_again(shifted_xs, profile)

        return x_array, normalised_array

    def core_80(self, x_array, profile):
        """ Get the central 80% of the field """
        # field width is at 50% of the max
        # middle 80% is 80% of this width (x axis)

        # find closest x values to 0.5 * max in the profile
        value = 0.5 * max(profile)
        half = int(len(profile) * 0.5)
        first_half = np.asarray(profile[0:half])
        index_1 = (np.abs(first_half - value)).argmin()
        second_half = np.asarray(profile[half:-1])
        index_2 = (np.abs(second_half - value)).argmin() + half

        # so we can define field width
        field_width = x_array[index_2] - x_array[index_1]

        # find the middle 80% of the field, in relation to the x axis index
        # values on x axis at 80% boundaries
        lwr = int((0.1 * field_width) + x_array[index_1])
        uppr = int(x_array[index_2] - (0.1 * field_width))

        # find the index in the original array closest to the upper and lower values,
        # which define the middle 80% of the field
        x_array = np.asarray(x_array)
        index_lwr = (np.abs(x_array - lwr)).argmin()
        index_uppr = (np.abs(x_array - uppr)).argmin()

        field_80 = profile[index_lwr:index_uppr]

        return field_80, index_lwr

    def symmetry(self, x_array, transformed_profile):
        """ Find the symmetry """
        # middle 80% of field
        field_80, index_lwr = self.core_80(x_array, transformed_profile)
        # we shifted to 0 by -centre so 0 is centre
        x_array = np.asarray(x_array)
        index = (np.abs(x_array - 0)).argmin()
        # find corresponding index in field_80
        index_centre = index - index_lwr

        # find the max difference for symmetric pairs
        symm =[]
        for i in range(int(len(field_80)*0.5)):
            right = index_centre + i
            left = index_centre - i
            cpd_percentage = 100 * (np.abs((field_80[right] - field_80[left])) / field_80[index_centre])
            symm.append(cpd_percentage)

        symmetry = max(symm)

        return symmetry


    def flattness(self, x_array, transformed_profile):
        """ Find the flatness of the field, in the middle 80% """

        # find percentage difference of the values in the middle 80%
        field_80, lrw = self.core_80(x_array, transformed_profile)
        # flatness is max absolute deviation from mean, expressed as percentage
        flat = []
        for x in field_80:
            diff = np.abs(x - np.mean(field_80))
            perc = 100 * (diff / np.mean(field_80))
            flat.append(perc)
        flatness = max(flat)
        return flatness

    def transform(self):
        """ Apply the calibration to the new image and return the transformed matrix,
         flatness and symmetry """

        # apply transformation matrix and plot

        # get central profiles of newly uploaded image
        img0 = Image(self.filename)
        image_sobel = Edges.sobel_edges(img0)
        # find centre
        new_centre = Profiles(image_sobel, [300, 900], [300, 900]).get_centre()
        # filter profiles
        central_profiles = Profiles(img0.gray(), [int(new_centre[0])], [int(new_centre[1])])
        profile_x, profile_y = central_profiles.filter()

        # normalise, shift, convert to distance
        x_array_x, normalised_array_x = self.process_profile(profile_x[0], new_centre[0])
        x_array_y, normalised_array_y = self.process_profile(profile_y[0], new_centre[1])

        # get calibration matrix
        transformation = self.get_matrix()
        # get centre of field matrix
        matrix_centre = self.get_original_centre()

        # multiply the central profiles by the matrix
        # corresponding to the centre
        transformed_profile_x = []
        for x in range(len(normalised_array_x)):
            transformed_profile_x.append(normalised_array_x[x] * transformation[int(matrix_centre[0])][x])
        transformed_profile_y = []
        for x in range(len(normalised_array_y)):
            transformed_profile_y.append(normalised_array_y[x] * transformation[x][int(matrix_centre[1])])

        symmetry_x = self.symmetry(x_array_x, transformed_profile_x)
        symmetry_y = self.symmetry(x_array_y, transformed_profile_y)

        flattness_x = self.flattness(x_array_x, transformed_profile_x)
        flattness_y = self.flattness(x_array_y, transformed_profile_y)

        return [x_array_x, transformed_profile_x, symmetry_x, flattness_x], \
               [x_array_y, transformed_profile_y, symmetry_y, flattness_y]
