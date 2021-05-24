"""
The script maps the beam profile check data from the MPC from 1st April 2021 onto
corresponding values from the SNC water phantom acquired on 1st April 2021.
The script generates calibration matrices of dose points mapped to each other
for each beam energy, with respect to the centre of the field.

Author: Nicola Compton
Date: 24th May 2021
Contact: nicola.compton@ulh.nhs.uk
"""

# Import classes from mains
from .main import Image, Edges, Profiles, Transform
import os
import pandas as pd

# open and read the mpc snc data for each energy
# this is the water phantom data

class SNC:

    """ Class for dealing with files exported from SNC i.e. water phantom"""

    def __init__(self):
        """ Define folder in which the SNC text files are stored"""
        pass

    def read_dose_tables():
        """Read the dose tables from the MPC snctxt files stored in calibration data folder"""

        # declare variables
        _6x_inline  = []
        _6x_crossline = []
        _10x_inline = []
        _10x_crossline =[]
        _10fff_inline =[]
        _10fff_crossline =[]

        # loop through directory
        directory = os.path.join(os.getcwd(), "Calibration_Data")

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
                    inline=1 # 1 for true
                if filename.find("crossline") != -1:
                    # crossline uses x measurements
                    data = [df['X (cm)'], df['Relative Dose (%)']]

                # determine the beam energy
                if filename.find("10fff") != -1:
                    if inline==1:
                        _10fff_inline = data
                    else:
                        _10fff_crossline = data

                if filename.find("10x") != -1:
                    if inline==1:
                        _10x_inline = data
                    else:
                        _10x_crossline = data

                if filename.find("6x") != -1:
                    if inline==1:
                        _6x_inline = data
                    else:
                        _6x_crossline = data

        dataset = [_6x_inline, _6x_crossline,
                    _10x_inline, _10x_crossline,
                    _10fff_inline, _10fff_crossline]

        return dataset


# run the above classes
# get the water phantom data
[_6x_inline, _6x_crossline,
_10x_inline, _10x_crossline,
_10fff_inline, _10fff_crossline] = SNC.read_dose_tables()

# get the EPID imager data from the MPC


# Get the directory where the XIM converted images are stored
directory = os.path.join(os.getcwd(), "Calibration_Data")


# loop through images in directory
for file in os.listdir(directory):
    if file.endswith(".png") or file.endswith(".jpeg"):
        filename = os.path.join(directory, file)
        # read image
        img0 = Image(filename)
        # detect edges
        img = Edges.sobel_edges(img0)
        # from the centre take x,y profiles in the original image
        x, y = Profiles(img, [300, 900], [300, 900]).get_centre()
        centre = int(x), int(y)

        # find out which energy it is and create object for each energy
        if filename.find("6x") != -1:
            _6x_central_profiles = Profiles(img0.gray(), [centre[0]], [centre[1]])
            _6x_sobel = Profiles(img, [300, 900], [300, 900])
            _6x_img = img0.gray()
            profile_x, profile_y = _6x_central_profiles.filter()
            # set an object which matches the water phantom and epid data for _6x energy
            _6x_mapping = Transform([_6x_inline, _6x_crossline], [profile_x[0], profile_y[0]], centre)
            # get the transform matrix
            _6x_matrix = _6x_mapping.dose_matrix()


        if filename.find("10x") != -1:
            _10x_central_profiles = Profiles(img0.gray(), [centre[0]], [centre[1]])
            _10x_sobel = Profiles(img, [300, 900], [300, 900])
            _10x_img = img0.gray()
            profile_x, profile_y = _10x_central_profiles.filter()
            _10x_mapping = Transform([_10x_inline, _10x_crossline], [profile_x[0], profile_y[0]], centre)
            # get the transform matrix
            _10x_matrix = _10x_mapping.dose_matrix()


        if filename.find("10fff") != -1:
            _10fff_central_profiles = Profiles(img0.gray(), [centre[0]], [centre[1]])
            _10fff_sobel = Profiles(img, [300, 900], [300, 900])
            _10fff_img = img0.gray()
            profile_x, profile_y = _10fff_central_profiles.filter()
            _10fff_mapping = Transform([_10fff_inline, _10fff_crossline], [profile_x[0], profile_y[0]], centre)
            # get the transform matrix
            _10fff_matrix = _10fff_mapping.dose_matrix()











