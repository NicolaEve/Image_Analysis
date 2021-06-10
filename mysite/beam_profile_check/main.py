"""
Script containing methods used to perform image analysis including the following classes:
Image: basic image functions, read, guaussian blue, greyscale
Edges: sobel edge detection of an image
Profiles: finds peaks, field corners, centre and plot functions
Transform: functions to calculate matrix of offset dose ratios in order to calibrate water tank doses to MPC profiles
TransformView: functions to apply the calibration matrix to the newly generated MPC EPID image,
returning symmetry and flatness metrics
SNC: reads the data from the water tank snc from 1st April 2021, for calibration

The script maps the beam profile check data from the MPC from 1st April 2021 onto
corresponding values from the SNC water phantom acquired on 1st April 2021.
The script generates calibration matrices of dose points mapped to each other
for each beam energy, with respect to the centre of the field.

Author: Nicola Compton
Date: 24th May 2021
Contact: nicola.compton@ulh.nhs.uk
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from .peakdetect import peakdetect
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import os
import pandas as pd
from scipy.spatial import distance


def normalise(x_array, y_array):
    """ Normalise the array by setting f(0)=1 i.e. dividing all values by the value at f(0)"""

    # find closest x value to 0 in the original array
    x_array = np.asarray(x_array)
    index = (np.abs(x_array - 0)).argmin()

    # divide all values in the array by f(0)
    normalised_array = [value / y_array[index] for value in y_array]

    return normalised_array


def interpolate(df, profile):
    """ Interpolate the dataframe (x,y) to the number of points in the profile """
    xs = df[0]
    ys = df[1]
    number_points = len(profile)

    new_xs = np.linspace(min(xs), max(xs), number_points)
    new_ys = interp1d(xs, ys)(new_xs)

    return new_xs, new_ys


def core_80(x_array, profile):
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


def symmetry(x_array, transformed_profile):
    """ Find the symmetry """
    # middle 80% of field
    field_80, index_lwr = core_80(x_array, transformed_profile)
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
        cpd_percentage = 100 * (np.abs((field_80[right] - field_80[left])) / field_80[index_centre])# don't need?
        symm.append(cpd_percentage)

    symmetry = max(symm)

    return symmetry


def flatness(x_array, transformed_profile):
    """ Find the flatness of the field, in the middle 80% """

    # get the central 80% of the field
    field_80, lrw = core_80(x_array, transformed_profile)

    # flatness is the ratio of maximum to minimum value
    flat = 100 * (max(field_80) / min(field_80))

    return flat


def line_intersection(line1, line2):
    """ This function finds the intersection point between two lines
        Input: two lines as defined by two (x,y) co-ordinates i.e. line1=[(x,y),(x,y)]
        Output: (x,y) co-ordinate of the point of intersection """

    # differences in x and y for each line
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    # find the determinant
    def det(a, b):
        """ Returns the determinant of the vectors a and b """
        return a[0] * b[1] - a[1] * b[0]

    # take the determinant of the lines
    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    # find the co-ordinates of the point of intersection
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y


def process_profile(profile, centre_cood):
    """ Input: profile = x or y profile; centre_cood = corresponding x or y central co-ordinate
        Output: the shifted (w.r.t. centre) and normalised arrays, measured in distance """
    # this just puts the plot symmetric about 0 and normalised and in distance

    # shift it to be centred at 0 and convert to cm, using MPC EPID resolution
    xs = np.arange(0, len(profile), 1)
    shifted_xs = [(x - int(centre_cood)) * 0.0336 for x in xs]
    # normalise profile
    normalised_array = normalise(shifted_xs, profile)

    return shifted_xs, normalised_array


class Image:
    """ Class for processing the EPID image from MPC.
        Input: filename and path of the image in .png or .jpeg format """

    def __init__(self, filename):
        self.filename = filename

    def read(self):
        """ Read the file using open cv2 """
        read_img = cv2.imread(str(self.filename))
        return read_img

    def gray(self):
        """ Convert the image to grayscale """
        grey_img = cv2.cvtColor(self.read(), cv2.COLOR_BGR2GRAY)
        return grey_img

    def remove_noise(self):
        """ Remove noise from the grayscale image """
        unnoisy_img = cv2.GaussianBlur(self.gray(), (3, 3), 0)
        return unnoisy_img

    def sobel(self):
        """ Detects the edges of an image using a sobel operator using the blurred grayscale image """

        # find edges using sobel operator
        # convolute with proper kernels
        sobelx = cv2.Sobel(self.remove_noise(), cv2.CV_64F, 1, 0, ksize=9)  # x
        sobely = cv2.Sobel(self.remove_noise(), cv2.CV_64F, 0, 1, ksize=9)  # y
        mag = np.sqrt(sobelx ** 2 + sobely ** 2)  # magnitude
        mag *= 255.0 / np.max(mag)  # normalise
        return mag

    def filter_profiles(self, image, x_axis, y_axis):
        """ Take the profiles of the image along the axis lines
        and smooth profiles with Savitzky-Golay filter"""
        profile_x = [image[i, :] for i in x_axis]
        profile_y = [image[:, j] for j in y_axis]
        filtered_x = [savgol_filter(profile, 43, 3) for profile in profile_x]
        filtered_y = [savgol_filter(profile, 43, 3) for profile in profile_y]
        return filtered_x, filtered_y

    def noisy_profiles(self, image, x_axis, y_axis):
        """ Take the profiles of the image along the axis lines """
        profile_x = image[x_axis, :]
        profile_y = image[:, y_axis]
        return profile_x, profile_y

    def get_max_peaks(self, array_list):
        """ Find the peaks of profiles in array_list using peakdetect.py
        Return the position and value of the maximum peaks (i.e. field edges) """

        # find peaks
        peak_list = [peakdetect(profile, lookahead=10, delta=10) for profile in array_list]

        # unpack results, returning position and value of maximum peaks
        positions = []
        max_values = []
        for peak in peak_list:
            [max_peaks, min_peaks] = peak
            x, y = zip(*max_peaks)
            positions.append(x)
            max_values.append(y)
        return positions, max_values

    def plot_peaks(self):
        """ Plot the filtered profile with the peaks to check they are accurate """

        # sobel image
        image = self.sobel()
        # arbitary profiles
        x_axis = [300, 900]
        y_axis = [300, 900]
        filtered_x, filtered_y = self.filter_profiles(image, x_axis, y_axis)
        filtered = filtered_x + filtered_y
        positions, max_values = self.get_max_peaks(filtered)

        # sub plot
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.plot(np.linspace(1, len(filtered[i]), len(filtered[i])), filtered[i])
            plt.plot(positions[i:i+1], max_values[i:i+1], 'x')
        plt.suptitle("Filtered Profiles with Peaks")
        plt.show()

    def get_corners(self):
        """ Find the corners of the sobel image by finding points of intersection between peaks """

        # take profiles of the sobel image along arbitary lines
        image = self.sobel()
        x_axis = [300, 900]
        y_axis = [300, 900]
        filtered_x, filtered_y = self.filter_profiles(image, x_axis, y_axis)
        filtered = filtered_x + filtered_y

        # get peaks
        positions, max_values = self.get_max_peaks(filtered)
        position_list = [i for sub in positions for i in sub]  # concatanate tuples into list

        # find corners by drawing a line through the peaks
        top_line = [(x_axis[0], position_list[0]), (x_axis[1], position_list[2])]
        bottom_line = [(x_axis[0], position_list[1]), (x_axis[1], position_list[3])]
        left_line = [(position_list[4], y_axis[0]), (position_list[6], y_axis[1])]
        right_line = [(position_list[5], y_axis[0]), (position_list[7], y_axis[1])]

        corners = [line_intersection(top_line, left_line), line_intersection(top_line, right_line),
                   line_intersection(bottom_line, left_line), line_intersection(bottom_line, right_line)]

        return corners

    def get_centre(self):
        """ Find the centre of the field by finding intersection between corners """

        # get corners
        [top_left, top_right, bottom_left, bottom_right] = self.get_corners()

        # draw line between corners to find centre of the field
        diag_1 = [top_left, bottom_right]
        diag_2 = [top_right, bottom_left]
        x, y = line_intersection(diag_1, diag_2)
        return int(x), int(y)

    def central_profiles(self):
        """ Get the x and y profiles as an average of the central +/- 5mm = 15 pixels """
        # find the centre of the sobel image
        centre = self.get_centre()
        # use centre co-ordinates as axis from which to take the profile
        # take profiles from the grey image
        image = self.gray()
        # set up an  empty list from which to take the average
        x_axis = []
        y_axis = []
        for i in range(-15, 15, 1):
            x_axis.append(centre[0] + i)
            y_axis.append(centre[1] + i)
        # apply the filter along all the profiles
        filtered_x, filtered_y = self.filter_profiles(image, x_axis, y_axis)

        # then take the average of the filtered profiles
        # axis 0 so that we have an array
        central_profiles = np.mean(filtered_x, axis=0), np.mean(filtered_y, axis=0)
        return central_profiles

    def plot_sobel_corners(self):
        """ Plot the sobel image with peaks
            (peaks are the edges of the field) """

        # sobel image
        image = self.sobel()
        # arbitary profiles
        x_axis = [300, 900]
        y_axis = [300, 900]
        filtered_x, filtered_y = self.filter_profiles(image, x_axis, y_axis)
        filtered = filtered_x + filtered_y
        # get peaks
        positions, max_values = self.get_max_peaks(filtered)
        position_list = [i for sub in positions for i in sub]  # concatanate tuples into list

        # plot the sobel image
        plt.imshow(image)

        # plot the edges of the field
        edges= [(x_axis[0], position_list[0]), (x_axis[1], position_list[2]),
                (x_axis[0], position_list[1]), (x_axis[1], position_list[3]),
                (position_list[4], y_axis[0]), (position_list[6], y_axis[1]),
                (position_list[5], y_axis[0]), (position_list[7], y_axis[1])]

        # plot the edges of the field and their intersection at the corners
        corners = self.get_corners()
        for x, y in edges+corners:
            plt.plot(x, y, 'x')

        # plot centre of field
        centre = self.get_centre()
        plt.plot(centre[0],centre[1],'x')

        plt.show()

    def plot_central_profiles(self):
        """ Plot central x and y profiles with and without noise """

        # grey image
        image = self.gray()
        # get centre
        centre = self.get_centre()

        # get profiles
        noisy_x, noisy_y = self.noisy_profiles(image, centre[0], centre[1])
        filtered_x, filtered_y = self.central_profiles()

        # plot x profile
        plt.plot(np.linspace(1, len(noisy_x), len(noisy_x)), noisy_x, label="Noisy")
        plt.plot(np.linspace(1, len(filtered_x), len(filtered_x)), filtered_x, label="Filtered")
        plt.title("X Profile")
        plt.legend()
        plt.show()

        # plot y profile
        plt.plot(np.linspace(1, len(noisy_y), len(noisy_y)), noisy_y, label="Noisy")
        plt.plot(np.linspace(1, len(filtered_y), len(filtered_y)), filtered_y, label="Filtered")
        plt.title("Y Profile")
        plt.legend()
        plt.show()


class Calibrate:
    """ Class to calibrate the water phantom images to the MPC EPID images """

    def __init__(self):
        self.snc_dir = os.path.join(os.getcwd(), "Calibration_Data")
        self. mpc_dir = os.path.join(os.getcwd(), "Calibration_Data")

    def snc_data(self):
        """Read the dose tables from the MPC snctxt files stored in calibration data folder"""

        # declare variables
        _6x_inline = []
        _6x_crossline = []
        _10x_inline = []
        _10x_crossline = []
        _10fff_inline = []
        _10fff_crossline = []

        # loop through directory
        directory = self.snc_dir

        for file in os.listdir(directory):
            if file.endswith('snctxt'):

                # read the file into a pandas dataframe
                filename = os.path.join(directory, file)
                df = pd.read_table(filename, header=0, names=['X (cm)', 'Y (cm)', 'Z (cm)', 'Relative Dose (%)'])

                # determine if it's inline or crossline
                # set inline to binary, 0 for false
                inline = 0
                if filename.find("inline") != -1:
                    # inline uses y measurements
                    data = [df['Y (cm)'], df['Relative Dose (%)']]
                    inline = 1  # 1 for true
                if filename.find("crossline") != -1:
                    # crossline uses x measurements
                    data = [df['X (cm)'], df['Relative Dose (%)']]

                # determine the beam energy
                if filename.find("10fff") != -1:
                    if inline == 1:
                        _10fff_inline = data
                    else:
                        _10fff_crossline = data

                if filename.find("10x") != -1:
                    if inline == 1:
                        _10x_inline = data
                    else:
                        _10x_crossline = data

                if filename.find("6x") != -1:
                    if inline == 1:
                        _6x_inline = data
                    else:
                        _6x_crossline = data

        dataset = [_6x_inline, _6x_crossline,
                   _10x_inline, _10x_crossline,
                   _10fff_inline, _10fff_crossline]

        return dataset

    def mpc_data(self):
        # loop through images in directory
        directory = self.mpc_dir

        for file in os.listdir(directory):
            if file.endswith(".png") or file.endswith(".jpeg"):
                filename = os.path.join(directory, file)
                # read image
                img = Image(filename)
                # get the centre of the field
                centre = img.get_centre()
                # get the central profiles
                profile_x, profile_y = img.central_profiles()

                # find out which energy it is and create object for each energy
                if filename.find("6x") != -1:
                    _6x_mpc = profile_x, profile_y, centre
                if filename.find("10x") != -1:
                    _10x_mpc = profile_x, profile_y, centre
                if filename.find("10fff") != -1:
                    _10fff_mpc = profile_x, profile_y, centre

        return _6x_mpc, _10x_mpc, _10fff_mpc

    def centre(self, energy):
        """ return the centre of the original 6x mpc epid image """
        _6x_mpc, _10x_mpc, _10fff_mpc = self.mpc_data()
        if energy == "6x":
            centre = _6x_mpc[2]
        if energy == "10x":
            centre = _10x_mpc[2]
        if energy == "10fff":
            centre = _10fff_mpc[2]
        return centre

    def calibration_arrays(self, snc, mpc):
        """ Construct the matrix of dose ratios """
        # input: snc: a dataframe of inline and crossline dose from the water tank
        # mpc: central x, y profiles and the centre of the field, format profile x, profile y, centre

        # define the profiles
        profile_x = mpc[0]
        profile_y = mpc[1]
        centre = mpc[2]

        # get the inline and crossline data, interpolate and normalise
        inline_df = snc[0]
        crossline_df = snc[1]

        # get the normalised profiles and doses
        # inline is y
        new_xs, new_ys = interpolate(inline_df, profile_y)
        inline_dose = normalise(new_xs, new_ys)
        norm_profile_y = normalise(new_xs, profile_y)

        # crossline is x
        new_xs, new_ys = interpolate(crossline_df, profile_x)
        crossline_dose = normalise(new_xs, new_ys)
        norm_profile_x = normalise(new_xs, profile_x)

        # initialise 2 empty calibration lists
        calibration_x = []
        calibration_y = []
        # enter the ratio: water tank dose / EPID MPC pixel value
        for i in range(0, len(norm_profile_x), 1):
            calibration_x.append(crossline_dose[i] / norm_profile_x[i])
        for i in range(0, len(norm_profile_y), 1):
            calibration_y.append(inline_dose[i] / norm_profile_y[i])

        return calibration_x, calibration_y

    def get_calibrations(self):
        """ Get the calibration matrix for each beam energy """
        # get the mpc and snc data
        _6x_mpc, _10x_mpc, _10fff_mpc = self.mpc_data()
        [_6x_inline, _6x_crossline,
         _10x_inline, _10x_crossline,
         _10fff_inline, _10fff_crossline] = self.snc_data()

        # run the matrix function for each energy
        _6x_cal = self.calibration_arrays([_6x_inline, _6x_crossline], _6x_mpc)
        _10x_cal = self.calibration_arrays([_10x_inline, _10x_crossline], _10x_mpc)
        _10fff_cal = self.calibration_arrays([_10fff_inline, _10fff_crossline], _10fff_mpc)

        return _6x_cal, _10x_cal, _10fff_cal

    def energy_6x(self):
        """ Return _6x calibration arrays """
        _6x_cal, _10x_cal, _10fff_cal = self.get_calibrations()
        return _6x_cal

    def energy_10x(self):
        """ Return 10x calibration arrays """
        _6x_cal, _10x_cal, _10fff_cal = self.get_calibrations()
        return _10x_cal

    def energy_10fff(self):
        """ Return 10fff calibration arrays """
        _6x_cal, _10x_cal, _10fff_cal = self.get_calibrations()
        return _10fff_cal

    def plot(self, energy):
        """ Plot the water tank and EPID profiles and the calibration ratio between them
            Input: energy as 6x, 10x or 10fff """

        # get the data
        _6x_mpc, _10x_mpc, _10fff_mpc = self.mpc_data()
        [_6x_inline, _6x_crossline,
         _10x_inline, _10x_crossline,
         _10fff_inline, _10fff_crossline] = self.snc_data()

        if energy == "6x":
            # get the calibration ratio for 6x
            cal_x, cal_y = self.energy_6x()
            # interpolate and normalise the uncalibrated mpc and snc data
            profile_x, profile_y, centre = _6x_mpc
            new_xs, new_ys = interpolate(_6x_crossline, profile_x)
            crossline_y_axis = normalise(new_xs, new_ys)
            crossline_x_axis = new_xs
            norm_profile_x = normalise(new_xs, profile_x)

            new_xs, new_ys = interpolate(_6x_inline, profile_y)
            inline_y_axis = normalise(new_xs, new_ys)
            inline_x_axis = new_xs
            norm_profile_y = normalise(new_xs, profile_y)
            # set the title
            title = "6x"

        if energy == "10x":
            # get the calibration ratio for 6x
            cal_x, cal_y = self.energy_10x()
            # interpolate and normalise the uncalibrated mpc and snc data
            profile_x, profile_y, centre = _10x_mpc
            new_xs, new_ys = interpolate(_6x_crossline, profile_x)
            crossline_y_axis = normalise(new_xs, new_ys)
            crossline_x_axis = new_xs
            norm_profile_x = normalise(new_xs, profile_x)

            new_xs, new_ys = interpolate(_6x_inline, profile_y)
            inline_y_axis = normalise(new_xs, new_ys)
            inline_x_axis = new_xs
            norm_profile_y = normalise(new_xs, profile_y)
            # set the title
            title = "10x"

        if energy == "10fff":
            # get the calibration ratio for 6x
            cal_x, cal_y = self.energy_10fff()
            # interpolate and normalise the uncalibrated mpc and snc data
            profile_x, profile_y, centre = _10fff_mpc
            new_xs, new_ys = interpolate(_6x_crossline, profile_x)
            crossline_y_axis = normalise(new_xs, new_ys)
            crossline_x_axis = new_xs
            norm_profile_x = normalise(new_xs, profile_x)

            new_xs, new_ys = interpolate(_6x_inline, profile_y)
            inline_y_axis = normalise(new_xs, new_ys)
            inline_x_axis = new_xs
            norm_profile_y = normalise(new_xs, profile_y)
            # set the title
            title = "10FFF"

        plt.plot(crossline_x_axis, crossline_y_axis, label="Water Phantom")
        plt.plot(crossline_x_axis, norm_profile_x, label="EPID Profile")
        plt.plot(crossline_x_axis, cal_x, label="Calibration Ratio Dose/Pixel")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Normalised Dose")
        crossline_title = str(title + "  : Crossline")
        plt.title(crossline_title)
        plt.legend()
        plt.show()

        plt.plot(inline_x_axis, inline_y_axis, label="Water Phantom")
        plt.plot(inline_x_axis, norm_profile_y, label="EPID Profile")
        plt.plot(inline_x_axis, cal_y, label="Calibration Ratio Dose/Pixel")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Normalised Dose")
        inline_title = str(title + "  : Inline")
        plt.title(inline_title)
        plt.legend()
        plt.show()


class NewImages:
    """ Class for processing newly acquired MPC EPID images """

    def __init__(self, energy, filename):
        self.energy = energy
        self.filename = filename

    def apply_calibration(self):
        """ Apply the calibration to the new image """
        cal = Calibrate()

        if self.energy == "6x":
            cal_x, cal_y = cal.energy_6x()
        if self.energy == "10x":
            cal_x, cal_y = cal.energy_10x()
        if self.energy == "10fff":
            cal_x, cal_y = cal.energy_10fff()

        profile_x, profile_y = Image(self.filename).central_profiles()
        new_centre = Image(self.filename).get_centre()
        # normalise, shift to centre, convert to distance
        x_array_x, normalised_x = process_profile(profile_x, new_centre[0])
        x_array_y, normalised_y = process_profile(profile_y, new_centre[1])
        transformed_x = []
        transformed_y = []
        for i in range(len(cal_x)):
            transformed_x.append(cal_x[i] * normalised_x[i])
        for i in range(len(cal_y)):
            transformed_y.append(cal_y[i] * normalised_y[i])

        return [x_array_x, transformed_x],\
               [x_array_y, transformed_y]

    def centre_shift(self):
        """ Get the centre shift in distance """
        new_centre = Image(self.filename).get_centre()
        cal = Calibrate()
        if self.energy == "6x":
            original_centre = cal.centre(energy="6x")
        if self.energy == "10x":
            original_centre = cal.centre(energy="10x")
        if self.energy == "10fff":
            original_centre = cal.centre(energy="10fff")
        shift = (distance.euclidean(new_centre, original_centre)) * 0.336
        # convert to distance
        return shift

    def symmetry(self, transformed):
        """ Return the symmetry of the new image """
        if transformed is True:
            # use the transformed data, once the calibration has been applied
            crossline, inline = self.apply_calibration()
            crossline_symm = symmetry(crossline[0], crossline[1])
            inline_symm = symmetry(inline[0], inline[1])
        else:
            # use the raw data before applying calibration
            profile_x, profile_y = Image(self.filename).central_profiles()
            new_centre = Image(self.filename).get_centre()
            # normalise, shift to centre, convert to distance
            x_array_x, normalised_x = process_profile(profile_x, new_centre[0])
            x_array_y, normalised_y = process_profile(profile_y, new_centre[1])
            crossline_symm = symmetry(x_array_x, normalised_x)
            inline_symm = symmetry(x_array_y, normalised_y)

        return crossline_symm, inline_symm

    def flatness(self, transformed):
        """ Return the flatness of the new image """
        if transformed is True:
            crossline, inline = self.apply_calibration()
            crossline_flat = flatness(crossline[0], crossline[1])
            inline_flat = flatness(inline[0], inline[1])

        else:
            # use the raw data before applying calibration
            profile_x, profile_y = Image(self.filename).central_profiles()
            new_centre = Image(self.filename).get_centre()
            # normalise, shift to centre, convert to distance
            x_array_x, normalised_x = process_profile(profile_x, new_centre[0])
            x_array_y, normalised_y = process_profile(profile_y, new_centre[1])
            crossline_flat = flatness(x_array_x, normalised_x)
            inline_flat = flatness(x_array_y, normalised_y)

        return crossline_flat, inline_flat










