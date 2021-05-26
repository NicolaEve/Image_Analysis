""" This script will be run locally to extract the historic data from Aria image
Analysis using my calibration will be run and compared to MPC results
Then I can export this data for statistical analysis and reporting on bias,
sensitivity, specificity, reliability
The reason it will only run locally is that Aria image is password protected """

# import modules
from beam_profile_check import run_calibration, main
#from main import *
import os
import shutil
import datetime
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import csv
import pyodbc

# define the directories
media_directory = r"C:\Users\NCompton\PycharmProjects\ImageAnalysis_venv_2\mysite\media\images"
xim_directory = os.path.join(media_directory, "XIMdata")
dir = r"Y:\TDS\H192138\MPCChecks"

# these store results from the MPC
beam_output_change = []
beam_uniformity_change = []
beam_centre_shift = []

for folder in os.listdir(dir):
    path = os.path.join(dir, folder)
    filename = str(path)
    if filename.find("6x") != -1:

        # get the creation date of the file
        created = os.stat(path).st_ctime
        date = datetime.fromtimestamp(created)

        # get the beam profile check data
        for file in os.listdir(path):
            if str(file) == "BeamProfileCheck.xim":
                xim_file = os.path.join(path, file)
                # extract the data
                # convert xim to png, first copy the file into the media directory
                new_name = os.path.join(media_directory, "6xBeam.xim")
                # saving as the same name every time will overwrite the previous one
                shutil.copy(xim_file, new_name)
                # now run ximmer
                ximmer = os.path.join(media_directory, "ximmerAll.py")
                cmd = str("python " + ximmer + " -d " + media_directory)
                os.system(cmd)
                # get the filename of the image file
                for new_file in os.listdir(xim_directory):
                    if new_file.endswith(".png") or new_file.endswith(".jpeg"):
                        image_file = os.path.join(xim_directory, new_file)
                # apply the transformation matrix for the beam energy
                # this could need changing to include taking the average of +/-10 pixel rows
                # average from the profiles adjacent to centre
                # but is that assuming the adajcent 10 are the centre and then applying calibration
                # or applying the transformation matrix and then taking avergae?
                inline, crossline = main.TransformView("6x", image_file).transform()
                inline.append(date)
                crossline.append(date)


                # structure of inline and crossline = [x, y, symmetry, flatness, date]
                # add new data into the database
                pyodbc

                # compare to the MPC results
                # get symmetry and flatness from the raw data for comparison
                # extract the reported results in the excel file
            if str(file) == "Results.csv":
                results_file = os.path.join(path, file)
                f = open(results_file)
                csv_reader_object = csv.reader(f)

                # structure of these is [name, value, threshold, pass/fail, date]
                for line in csv_reader_object:
                    name = str(line[0])
                    if name.find("BeamOutputChange") != -1:
                        beam_output_change.append([line, date])
                    if name.find("BeamUniformityChange") != -1:
                        beam_uniformity_change.append([line, date])
                    if name.find("BeamCenterShift") != -1:
                        beam_centre_shift.append([line, date])


                # put that all in a database, so then can put into a spreadsheet
                # spc on the mpc results

    if filename.find("10x") != -1:
        if filename.find("10xFFF") != -1:
            for files in os.listdir(path):
                if str(files).find("BeamProfileCheck") != -1:
                    image_file = os.path.join(path, files)
                    #new_name = os.path.join(media_directory, str("10xfff_" + str(it_10fff) + ".xim"))
                    #shutil.copy(image_file, new_name)
                    #it_10fff = it_10fff + 1

        else:
            for files in os.listdir(path):
                if str(files).find("BeamProfileCheck") != -1:
                    image_file = os.path.join(path, files)
                    #new_name = os.path.join(media_directory, str("10x_" + str(it_10x) + ".xim"))
                    #shutil.copy(image_file, new_name)
                    #it_10x = it_10x + 1

# look for the folders with 10x, 10xFFF and 6x
# inside these search for BeamProfileCheck.xim
# copy files into media folder, rename to match the beam energy?

# apply what is done above:
# convert xim to png
# sobel image > find centre > filter profiles > normalise
# > apply transformation matrix > convert to distance > plot
# display symmetry

# what if ariaimage is password protected? it won't pull from a server level
# batch file to copy and put in credenitals -> security, can't be open source