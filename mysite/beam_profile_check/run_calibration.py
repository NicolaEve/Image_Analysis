## Script which runs and gets data for image files
## map the Beam Profile check from the MPC onto the dose values from the SNC water phanton images

# Import classes from main
# import modules
from .main import Image, Edges, Profiles, SNC, Transform
import os

# profile x is inline or crossline ?
# assume x is inline, check this

# open and read the mpc snc data for each energy
# this is the water phantom data
[_6x_inline, _6x_crossline,
 _10x_inline, _10x_crossline,
 _10fff_inline, _10fff_crossline] = SNC.read_dose_tables()

# get the EPID imager data

# Convert xim to png / dicom by running ximmerAll.py -d 'name of directory where XIM files are stored'
# it will save the .png files in the folder 'XIMdata'

# Get the directory where the XIM converted images are stored
#directory = os.path.join(os.getcwd(), "XIMdata")
directory = r"C:\Users\NCompton\PycharmProjects\ImageAnalysis\XIMdata"

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
            # plot
            #_6x_mapping.plot(inline=True)
            #_6x_mapping.plot(inline=False)
            #_6x_sobel.plot_sobel_corners()
            #_6x_sobel.plot_peaks()

            # check field sizes are equal
            horizontal, vertical = _6x_sobel.field_size_cm()
            inline_size = _6x_mapping.field_size(_6x_inline, profile_x[0])
            crossline_size = _6x_mapping.field_size(_6x_crossline, profile_y[0])
            inline_diff = horizontal - inline_size
            crossline_diff = vertical - crossline_size
            #print(inline_diff, crossline_diff)


        if filename.find("10x") != -1:
            _10x_central_profiles = Profiles(img0.gray(), [centre[0]], [centre[1]])
            _10x_sobel = Profiles(img, [300, 900], [300, 900])
            _10x_img = img0.gray()
            profile_x, profile_y = _10x_central_profiles.filter()
            _10x_mapping = Transform([_10x_inline, _10x_crossline], [profile_x[0], profile_y[0]], centre)
            # get the transform matrix
            _10x_matrix = _10x_mapping.dose_matrix()
            # plot
            #_10x_mapping.plot(inline=True)
            #_10x_mapping.plot(inline=False)
            # check field sizes are equal
            horizontal, vertical = _10x_sobel.field_size_cm()
            inline_size = _10x_mapping.field_size(_10x_inline, profile_x[0])
            crossline_size = _10x_mapping.field_size(_10x_crossline, profile_y[0])
            inline_diff = horizontal - inline_size
            crossline_diff = vertical - crossline_size
            #print(inline_diff, crossline_diff)


        if filename.find("10fff") != -1:
            _10fff_central_profiles = Profiles(img0.gray(), [centre[0]], [centre[1]])
            _10fff_sobel = Profiles(img, [300, 900], [300, 900])
            _10fff_img = img0.gray()
            profile_x, profile_y = _10fff_central_profiles.filter()
            _10fff_mapping = Transform([_10fff_inline, _10fff_crossline], [profile_x[0], profile_y[0]], centre)
            # get the transform matrix
            _10fff_matrix = _10fff_mapping.dose_matrix()

            #plot
            #_10fff_mapping.plot(inline=True)
            #_10fff_mapping.plot(inline=False)


            # check field sizes are equal
            horizontal, vertical = _10fff_sobel.field_size_cm()
            inline_size = _10fff_mapping.field_size(_10fff_inline, profile_x[0])
            crossline_size = _10fff_mapping.field_size(_10fff_crossline, profile_y[0])
            inline_diff = horizontal - inline_size
            crossline_diff = vertical - crossline_size
            #print(inline_diff, crossline_diff)









