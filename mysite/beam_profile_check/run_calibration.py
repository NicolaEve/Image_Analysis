"""
copy of main for tidying purposes
"""


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
        profile_x = [image[i, :] for i in x_axis]
        profile_y = [image[:, j] for j in y_axis]
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

    def central_profiles(self):
        # find the centre of the sobel image
        centre = self.get_centre()
        # use centre co-ordinates as axis from which to take the profile
        x_axis = centre[0]
        y_axis = centre[1]
        # take profiles from the grey image
        image = self.gray()
        central_profiles = self.filter_profiles(image, x_axis, y_axis)
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
        filtered_x, filtered_y = self.filter_profiles(image, centre[0], centre[1])

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


class Calibrate:
    """ Class to calibrate the water phantom images to the MPC EPID images
        Input: full path and file location where .snctxt files and .. are stored """

    def __init__(self):
        self.snc_dir = os.path.join(os.getcwd(), "Calibration_Data")
        self. mpc_dir = os.path.join(os.getcwd(), "Calibration_Data")
        # snc and mpc is directory = os.path.join(os.getcwd(), "Calibration_Data")

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

    def matrix(self, snc, mpc):
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
        x = centre[0]
        y = centre[1]

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

    def get_matrices(self):
        """ Get the calibration matrix for each beam energy """
        # get the mpc and snc data
        _6x_mpc, _10x_mpc, _10fff_mpc = self.mpc_data()
        [_6x_inline, _6x_crossline,
         _10x_inline, _10x_crossline,
         _10fff_inline, _10fff_crossline] = self.snc_data()

        # run the matrix function for each energy
        _6x_cal_matrix = self.matrix([_6x_inline, _6x_crossline], _6x_mpc)
        _10x_cal_matrix = self.matrix([_10x_inline, _10x_crossline], _10x_mpc)
        _10fff_cal_matrix = self.matrix([_10fff_inline, _10fff_crossline], _10fff_mpc)

        return _6x_cal_matrix, _10x_cal_matrix, _10fff_cal_matrix

    def _6x(self):
        """ Return _6x calibration matrix """
        _6x_cal_matrix, _10x_cal_matrix, _10fff_cal_matrix = self.get_matrices()
        return _6x_cal_matrix

    def _10x(self):
        """ Return _6x calibration matrix """
        _6x_cal_matrix, _10x_cal_matrix, _10fff_cal_matrix = self.get_matrices()
        return _10x_cal_matrix

    def _10fff(self):
        """ Return _6x calibration matrix """
        _6x_cal_matrix, _10x_cal_matrix, _10fff_cal_matrix = self.get_matrices()
        return _10fff_cal_matrix


def process_profile(profile, centre_cood):
    """ Input: profile = x or y profile; centre_cood = corresponding x or y central co-ordinate
        Output: the shifted (w.r.t. centre) and normalised arrays, measured in distance """
    # this just puts the plot symmetric about 0 and normalised and in distance

    # shift it to be centred at 0 and convert to cm, using MPC EPID resolution
    xs = np.linspace(0, len(profile), len(profile))
    shifted_xs = [(x - int(centre_cood)) * 0.0336 for x in xs]
    # normalise profile
    normalised_array= normalise(shifted_xs, profile)

    return shifted_xs, normalised_array


def transform(energy, filename):
    """ Input: energy, filename of newly uploaded EPID image """
    if energy == "6x":
        """ Transform the 6x energy """
        cal = Calibrate._6x()
        centre = Image(filename).get_centre()
        profile_x, profile_y = Image(filename).central_profiles()
        # shift and convert to distance
        x_array_x, shifted_x = process_profile(profile_x, centre[0])
        x_array_y, shifted_y = process_profile(profile_y, centre[1])
        # apply matrix - need centre from the original













