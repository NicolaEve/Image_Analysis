"""
This script tests the functionality of the app
To run, cd into mysite and type 'nose2 -v' to the command line
A coverage report can be generated using nose2 --with-coverage   -vvv

Author: Nicola Compton
Date: 24th May 2021
Contact: nicola.compton@ulh.nhs.uk
"""


from django.test import TestCase

import unittest
from .main import *
import numpy as np
import matplotlib
from matplotlib import testing
from matplotlib.testing import decorators, compare, exceptions
from matplotlib.testing.decorators import image_comparison
import pytest
import cv2
import os
import pandas as pd
from .peakdetect import peakdetect
from numpy import diff

# directory where test images are stored
image_test_dir = os.path.join(os.getcwd(), "Images_for_tests")

# import test images
file_colour = os.path.join(image_test_dir, "Test_image_colour.png")
file_sobel = os.path.join(image_test_dir, "test_image_sobel.png")


class ImageTests(unittest.TestCase):
    """ Class to test the functionality of the image processing functions """

    def setUp(self):
        """Run prior to each test."""
        self.colour = Image(file_colour)
        self.sobel = Image(file_sobel)

    def test_sobel_equal(self):
        """Test that the Sobel edge detection function returns edges"""
        # save image as a .png so it can be tested using matplotlib compare
        out_img = Edges.sobel_edges(self.colour)
        global out_file
        out_file = os.path.join(image_test_dir, "output_image.png")
        cv2.imwrite(out_file, out_img)
        # test that they are equal, tolerance is 40 - bit iffy
        out = matplotlib.testing.compare.compare_images(out_file, file_sobel, 40, in_decorator=True)
        if str(out) != 'None':
            raise Exception('Edge detection using Sobel operator failed')

    def test_sobel_unequal(self):
        """Test the images are not equal"""
        out = matplotlib.testing.compare.compare_images(out_file, file_colour, 40)
        if str(out) == 'None':
            raise Exception('Sobel operator has not altered the image')
        # should compare_images raise the exception?

    def tearDown(self):
        """Run post each test."""
        pass


class ProfileTests(unittest.TestCase):
    """ Class to test the functionality of the profile processing functions """

    def setUp(self):
        """Run prior to each test."""
        image = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) # test array image
        self.profile = Profiles(image, [1], [1]) # test along these axis

    def test_get_profiles(self):
        """ Test that it retrieves the correct profiles """
        x_profile, y_profile = self.profile.get_profiles()
        np.testing.assert_array_equal(x_profile, np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
        np.testing.assert_array_equal(y_profile, np.array([[2, 2, 2]]))

    def tearDown(self):
        """Run post each test."""
        pass


class TransformTests(unittest.TestCase):
    """ Class tp test the functionality of the matrix transforms """

    def setUp(self):
        """Run prior to each test."""
        # set the test variables
        image = np.array([[1,2,3,4,5,6,7,8,9,10],
                        [1,2,3,4,5,6,7,8,9,10],
                        [1,2,3,4,5,6,7,8,9,10]])
        inline = [[-2,-1,0,1,2], [30,20,10,20,30]]
        crossline = [[-2,-1,0,1,2], [50,30,10,30,50]]
        df_list = [inline, crossline]
        profile_list = [[1,1,1,1,1], [1,1,1,1,1]]
        centre = 2, 2

        self.transform = Transform(df_list, profile_list, centre)

    def test_dose_matrix(self):
        """ Test the dose matrix function returns the correct ratios """
        dose_matrix = self.transform.dose_matrix()
        output = np.array([[15, 10, 5, 10, 15],
                          [9, 6, 3, 6, 9],
                          [3, 2, 1, 2, 3],
                          [9, 6, 3, 6, 9],
                          [15, 10, 5, 10, 15]])
        np.testing.assert_array_equal(dose_matrix, output)

    def tearDown(self):
        """Run post each test."""
        pass


class StaticTests(unittest.TestCase):
    """ Class to test static functions """

    def setUp(self):
        """ Run prior to each test."""
        self.normalise = normalise
        self.interpolate = interpolate
        self.core_80 = core_80
        self.symmetry = symmetry
        self.flatness = flatness
        self.line_intersection = line_intersection
        self.process_profile = process_profile

    def test_normalise(self):
        """ Test the normalising function """
        inline = [[-2, -1, 0, 1, 2], [30, 20, 10, 20, 30]]
        normalised = self.normalise(inline[0], inline[1])
        np.testing.assert_array_equal(normalised, np.array([3, 2, 1, 2, 3]))

    def test_interpolate(self):
        """ Test the interpolation function """
        df = [[-2, -1, 0, 1, 2], [10, 20, 30, 40, 50]]
        array = np.ones([10, 1])
        expected_xs = np.linspace(-2, 2, 10)
        expected_ys = np.linspace(10, 50, 10)
        xs, ys = self.interpolate(df, array)
        np.testing.assert_allclose(expected_xs, xs)
        np.testing.assert_allclose(expected_ys, ys)

    def test_core_80(self):
        """ Test the function which returns the central 80% of the field """
        profile = [1, 5, 10, 10, 10,
                   10, 10, 10, 10, 10,
                   10, 5, 1]
        x_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        exp_80 = profile[2:10]
        out_80, lwr_index = self.core_80(x_array, profile)
        self.assertEqual(lwr_index, 2)
        np.testing.assert_array_equal(exp_80, out_80)

    def test_symmetry(self):
        """ Test the symmetry function """
        profile = [1, 5, 10, 10, 10,
                   10, 10, 10, 10, 10,
                   10, 5, 1]
        x_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        symm = self.symmetry(x_array, profile)
        self.assertEqual(symm, 0)

        profile = [1, 5, 10, 6, 10,
                   10, 1, 10, 10, 10,
                   10, 5, 1]
        x_array = [0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 12]
        symm = self.symmetry(x_array, profile)
        self.assertEqual(symm, 90)

    def test_flatness(self):
        """ Test the flatness function """
        profile = [1, 5, 10, 6, 10,
                   10, 2, 10, 10, 10,
                   10, 5, 1]
        x_array = [0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 12]
        flat = self.flatness(x_array, profile)
        self.assertEqual(flat, 500)

        profile = [1, 5, 10, 10, 10,
                   10, 10, 10, 10, 10,
                   10, 5, 1]
        flat = self.flatness(x_array, profile)
        self.assertEqual(flat, 100)

    def test_lines_no_intersection(self):
        """ Test that the lines do not intersect """
        # check lines do not intersect error is raised
        line1 = [(0, 3), (3, 3)]
        line2 = [(0, 6), (3, 6)]
        # assert raises needs to be within a wrapper function to catch the error and not become a test error
        # lamba is a wrapper
        # with statement passes the error thrown by the function back into the assertRaises to catch it
        with self.assertRaises(Exception): self.line_intersection(line1, line2)

    def test_line_intersection(self):
        """ Test that the lines intersect at the correct point"""
        # check correct point of intersection is found
        line1 = [(0, 3), (3, 3)]
        line2 = [(0, 0), (0, 6)]
        self.assertTupleEqual(self.line_intersection(line1, line2), (0, 3))

    def test_process_profile(self):
        """ Test the profile function shifts to the centre, normalises and converts to distance """

        # set the input variables
        profile = [0, 1, 2, 3, 4, 3, 2, 1, 0]
        centre = 4

        # get the expected output by shifting the x axis
        shifted_xs = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        # convert to distance
        exp_xs = [x * 0.0336 for x in shifted_xs]
        # normalise profile
        exp_profile = [y / 4 for y in profile]

        out_xs, out_profile = self.process_profile(profile, centre)
        np.testing.assert_array_equal(out_xs, exp_xs)
        np.testing.assert_array_equal(out_profile, exp_profile)

        # is this using the same method as the function in main? is that robust?

    def tearDown(self):
        """ Run post each test."""
        pass


class FieldTests(unittest.TestCase):
    """ Check field size calculated correctly and matches """

    def setUp(self):
        """ Run prior to every test """

        # set the image
        image = np.random.randint(1, 50, size=(100, 100))
        image[10, :] = 100
        image[90, :] = 100
        image[:, 10] = 100
        image[:, 90] = 100

        profiles = [30, 70]

        self.sobel_profile = Profiles(image, profiles, profiles)

    def test_get_max_peaks(self):
        """ Test it finds the position and values of the peaks """

        # set up a testing array with known peaks
        x = np.linspace(-10, 10, 101)
        y1 = []
        for i in x:
            y1.append(-(i**2))

        # function usually accepts lists
        array_list = [y1, y1]
        positions, max_values = self.sobel_profile.get_max_peaks(array_list)

        # negative squared function has max value at -0, positioned in centre
        self.assertListEqual(positions, [(50,), (50,)])
        self.assertListEqual(max_values, [(-0,), (-0,)])

    def tearDown(self):
        """Run post each test."""
        pass


if __name__ == "__main__":
    unittest.main()
