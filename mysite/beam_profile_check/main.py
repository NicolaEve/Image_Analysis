"""
Script containing methods used to perform image analysis including the following classes:
Image: basic image functions, read, guaussian blue, greyscale
Edges: sobel edge detection of an image
Profiles: finds peaks, field corners, centre and plot functions
Transform: functions to calculate matrix of offset dose ratios in order to calibrate water tank doses to MPC profiles

Author: Nicola Compton
Date: 24th May 2021
Contact: nicola.compton@ulh.nhs.uk
"""

# importing modules
import matplotlib.pyplot as plt
import numpy as np
import cv2
from .peakdetect import peakdetect
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from . import run_calibration

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


class Image:

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


class Edges(Image):

    def __init__(self):
        super().__init__(filename)

    def sobel_edges(self):
        """ Detects the edges of an image using a sobel operator using the blurred grayscale image """

        # find edges using sobel operator
        # convolute with proper kernels
        sobelx = cv2.Sobel(self.remove_noise(), cv2.CV_64F, 1, 0, ksize=9)  # x
        sobely = cv2.Sobel(self.remove_noise(), cv2.CV_64F, 0, 1, ksize=9)  # y
        mag = np.sqrt(sobelx ** 2 + sobely ** 2)  # magnitude
        mag *= 255.0 / np.max(mag)  # normalise
        return mag


class Profiles:

    def __init__(self, image, x_axis, y_axis):
        """ Arbitary lines across which to take the profiles, axis must be lists """
        self.image = image
        self.x_axis = x_axis
        self.y_axis = y_axis

    def get_profiles(self):
        """ Take the profiles of the image along the axis lines """
        profile_x = [self.image[i, :] for i in self.x_axis]
        profile_y = [self.image[:, j] for j in self.y_axis]
        return profile_x, profile_y

    def filter(self):
        """ Smooth profiles with Savitzky-Golay filter """
        profile_x, profile_y = self.get_profiles()
        filtered_x = [savgol_filter(profile, 43, 3) for profile in profile_x]
        filtered_y = [savgol_filter(profile, 43, 3) for profile in profile_y]
        return filtered_x, filtered_y

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

        filtered_x, filtered_y = self.filter()
        filtered = filtered_x + filtered_y
        positions, max_values = self.get_max_peaks(filtered)

        # sub plot
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.plot(np.linspace(1, len(filtered[i]), len(filtered[i])), filtered[i])
            plt.plot(positions[i:i+1], max_values[i:i+1], 'x')
        plt.suptitle("Filtered Profiles with Peaks")
        plt.show()

    def line_intersection(self, line1, line2):
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

    def get_corners(self):
        """ Find the corners by finding points of intersection between peaks """

        # get peaks
        filtered_x, filtered_y = self.filter()
        filtered = filtered_x + filtered_y
        positions, max_values = self.get_max_peaks(filtered)
        position_list = [i for sub in positions for i in sub]  # concatanate tuples into list

        # find corners by drawing a line through the peaks
        top_line = [(self.x_axis[0], position_list[0]), (self.x_axis[1], position_list[2])]
        bottom_line = [(self.x_axis[0], position_list[1]), (self.x_axis[1], position_list[3])]
        left_line = [(position_list[4], self.y_axis[0]), (position_list[6], self.y_axis[1])]
        right_line = [(position_list[5], self.y_axis[0]), (position_list[7], self.y_axis[1])]

        corners = [self.line_intersection(top_line, left_line), self.line_intersection(top_line, right_line),
                   self.line_intersection(bottom_line, left_line), self.line_intersection(bottom_line, right_line)]

        return corners

    def get_centre(self):
        """ Find the centre of the field by finding intersection between corners """

        # get corners
        [top_left, top_right, bottom_left, bottom_right] = self.get_corners()

        # draw line between corners to find centre of the field
        diag_1 = [top_left, bottom_right]
        diag_2 = [top_right, bottom_left]
        x, y = self.line_intersection(diag_1, diag_2)
        return x, y

    def field_size_pixels(self):
        """ Find the size of the field in number of pixels """

        # get corners
        [top_left, top_right, bottom_left, bottom_right] = self.get_corners()

        # get distance, assume no tilt
        distance_lr = top_right[0] - top_left[0]
        distance_ud = bottom_left[1] - top_left[1]

        field_size_pixels = [distance_lr, distance_ud]

        return field_size_pixels

    def field_size_cm(self):
        """ Find the size of the field in cm """

        # get corners
        [top_left, top_right, bottom_left, bottom_right] = self.get_corners()

        # get distance, assume no tilt
        distance_lr = top_right[0] - top_left[0]
        distance_ud = bottom_left[1] - top_left[1]

        # convert pixel to mm
        # one pixel = 	0.035? need to check this
        field_size_cm = [0.0336 * distance_lr, 0.0336 * distance_ud]

        return field_size_cm

    def plot_sobel_corners(self):
        """ Plot the sobel image with peaks
            (peaks are the edges of the field) """

        # get peaks
        filtered_x, filtered_y = self.filter()
        filtered = filtered_x + filtered_y
        positions, max_values = self.get_max_peaks(filtered)
        position_list = [i for sub in positions for i in sub]  # concatanate tuples into list

        # plot the sobel image
        plt.imshow(self.image)

        # plot the edges of the field
        edges= [(self.x_axis[0], position_list[0]), (self.x_axis[1], position_list[2]),
                (self.x_axis[0], position_list[1]), (self.x_axis[1], position_list[3]),
                (position_list[4], self.y_axis[0]), (position_list[6], self.y_axis[1]),
                (position_list[5], self.y_axis[0]), (position_list[7], self.y_axis[1])]

        # plot the edges of the field and their intersection at the corners
        corners = self.get_corners()
        for x, y in edges+corners:
            plt.plot(x, y, 'x')

        # plot centre of field
        centre = self.get_centre()
        plt.plot(centre[0],centre[1],'x')

        plt.show()

    def plot_profiles(self):
        """ Plot x and y profiles with and without noise """

        # get profiles
        noisy_x, noisy_y = self.get_profiles()
        filtered_x, filtered_y = self.filter()

        # plot x profile
        plt.plot(np.linspace(1, len(noisy_x[0]), len(noisy_x[0])), noisy_x[0], label="Noisy")
        plt.plot(np.linspace(1, len(filtered_x[0]), len(filtered_x[0])), filtered_x[0], label="Filtered")
        plt.title("X Profile")
        plt.legend()
        plt.show()

        # plot y profile
        plt.plot(np.linspace(1, len(noisy_y[0]), len(noisy_y[0])), noisy_y[0], label="Noisy")
        plt.plot(np.linspace(1, len(filtered_y[0]), len(filtered_y[0])), filtered_y[0], label="Filtered")
        plt.title("Y Profile")
        plt.legend()
        plt.show()


class Transform:

    def __init__(self, df_list, profile_list, centre):
        """ Input: df_list = The pandas dataframe from the water tank as [inline, crossline],
        profile_list = x and y profiles and the EPID image
         centre = central co-ordinate of the field """
        self.df_list = df_list
        self.profile_list = profile_list
        self.centre = centre

    def field_size(self, df, profile):
        """ Calculate field size as distance between inflection points """

        # first interpolate and normalise the df - inline/crossline
        xs, ys = interpolate(df, profile)
        normalised_ys = normalise(xs, ys)

        # find the two values where the normalised y = 0.5
        half = int(len(normalised_ys) * 0.5)

        first_half = np.asarray(normalised_ys[0:half])
        index_1 = (np.abs(first_half - 0.5)).argmin()

        second_half = np.asarray(normalised_ys[half:len(normalised_ys)])
        index_2 = half + (np.abs(second_half - 0.5)).argmin()

        # distance is the difference on the corresponding x axis
        distance = xs[index_2] - xs[index_1]

        # finding where second derivtive is 0 could be more accurate?
        # inflection at f"=0
        # but how to find derivative of a list?

        return distance

    def plot(self, inline):
        """ Plot the interpolated, normalised profiles and the ratio between them """
        # inline is first entry
        if inline is True:
            df = self.df_list[0]
            profile = self.profile_list[0]
            title = "Inline"
        else:
            df = self.df_list[1]
            profile = self.profile_list[1]
            title = "Crossline"

        new_xs, new_ys = interpolate(df, profile)
        x_axis = new_xs
        y_axis = normalise(new_xs, new_ys)
        norm_profile = normalise(new_xs, profile)
        ratio = []
        # the ratio is the normalised dose from water phantom / pixel value from EPID profile
        for i in range(len(x_axis)):
            ratio.append(y_axis[i] / norm_profile[i])

        plt.plot(x_axis, y_axis, label="Water Phantom")
        plt.plot(x_axis, norm_profile, label="EPID Profile")
        plt.plot(x_axis, ratio, label="Ratio Dose/Pixel")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Normalised Dose")
        plt.title(title)
        plt.legend()
        plt.show()

    def dose_matrix(self):
        """ Construct the matrix of dose ratios """

        # define the profiles
        profile_x = self.profile_list[0]
        profile_y = self.profile_list[1]

        # get the inline and crossline data, interpolate and normalise
        inline_df = self.df_list[0]
        crossline_df = self.df_list[1]

        # get the normalised profiles and doses
        # inline
        new_xs, new_ys = interpolate(inline_df, profile_x)
        inline_dose = normalise(new_xs, new_ys)
        norm_profile_x = normalise(new_xs, profile_x)

        # crossline
        new_xs, new_ys = interpolate(crossline_df, profile_y)
        crossline_dose = normalise(new_xs, new_ys)
        norm_profile_y = normalise(new_xs, profile_y)

        # initialise empty array = size of image
        dose_matrix = np.empty((len(norm_profile_x), len(norm_profile_y)), dtype=float)

        # get centre value
        x, y = self.centre

        # enter the ratio: normalised doses / normalised profile along the central axis
        # using central axis = centre of field
        for i in range(len(norm_profile_x)):
            dose_matrix[x, i] = inline_dose[i] / norm_profile_x[i]

        for j in range(len(norm_profile_y)):
            dose_matrix[j, y] = crossline_dose[j] / norm_profile_y[j]

        # set all values in the matrix to the product of corresponding dose ratios
        # needs to be relative to the centre of the field which may not be the centre of the matrix
        # in this way we're mapping to the field centre of the image
        for n in range(len(profile_x)):
            for m in range(len(profile_y)):
                if n != x and m != y:
                    dose_matrix[n, m] = dose_matrix[n][y] * dose_matrix[x][m]

        return dose_matrix


class TransformView:
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

    def process_profile(self, profile, centre_cood):
        """ Input: profile = x or y profile; centre_cood = corresponding x or y central co-ordinate
            Output: the shifted (w.r.t. centre) and normalised arrays, measured in distance """

        # shift it to be centred at 0 and convert to cm, using MPC EPID resolution
        xs = np.linspace(0, len(profile), len(profile))
        shifted_xs = [(x - int(centre_cood)) * 0.0336 for x in xs]
        # normalise profile
        normalised_array= normalise(shifted_xs, profile)

        return shifted_xs, normalised_array

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

    def flatness(self, x_array, transformed_profile):
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

        flatness_x = self.flatness(x_array_x, transformed_profile_x)
        flatness_y = self.flatness(x_array_y, transformed_profile_y)

        return [x_array_x, transformed_profile_x, symmetry_x, flatness_x], \
               [x_array_y, transformed_profile_y, symmetry_y, flatness_y]














