""" Script to be run locally in order to extract historic data from Aria image
and write the MPC results into a database.
(The reason for running locally is Aria image is password protected and the database is local) """

# import modules
from beam_profile_check import main
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

# set up connection to the MPC database
cnxn_str = ("Driver={SQL Server Native Client 11.0};"
            "Server=IT049561\SQLEXPRESS;"
            "Database=MPC;"
            "Trusted_Connection=yes;")
dbo = pyodbc.connect(cnxn_str)
cursor = dbo.cursor()


# define a function to extract the data
def historic(energy, filepath):
    # we want to skip if there isn't a .xim file in this folder
    try:
        xim_file = os.path.join(filepath, "BeamProfileCheck.xim")
        results_file = os.path.join(filepath, "Results.csv")

        # get the creation date of one of the files
        created = os.stat(xim_file).st_ctime
        date = datetime.fromtimestamp(created)

        # get the most recent entry from the database
        statement = """declare @recent datetime; SELECT MAX(QA_Date) FROM MPC_Event as recent"""
        cursor.execute(statement)
        last_date = cursor.fetchval()

        # only extract data from files which have not already been written into the database
        cutoff = date > last_date
        if cutoff is True:

            # enter date and beam energy to database
            # return the mpc event id
            statement = """declare @mpc_event_id int;
                                       EXEC @mpc_event_id = Insert_MPC_Event ?,?;
                                       SELECT @mpc_event_id as mpc_event_id; """
            cursor.execute(statement, [energy, date])
            mpc_event_id = cursor.fetchval()
            cursor.commit()

            # extract the data from the xim file
            # convert xim to png, first copy the file into the media directory
            new_name = os.path.join(media_directory, "Beam.xim")
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

                    # get the calibrated symmetry, flatness and beam centre shift
                    obj = main.NewImages(energy, image_file)
                    crossline_symm, inline_symm = obj.symmetry(transformed=True)
                    crossline_flat, inline_flat = obj.flatness(transformed=True)
                    centre_shift = obj.centre_shift()
                    # enter into database
                    # Binary: inline is 1, transformed is 1
                    cursor.execute("EXEC Insert_Symm_Flat_Data ?,?,?,?,?",
                                   [mpc_event_id, 1, 1, inline_symm, inline_flat])
                    cursor.commit()
                    # repeat for crossline
                    cursor.execute("EXEC Insert_Symm_Flat_Data ?,?,?,?,?",
                                   [mpc_event_id, 0, 1, crossline_symm, crossline_flat])
                    cursor.commit()

                    # repeat for untransformed data, before calibration applied
                    crossline_symm, inline_symm = obj.symmetry(transformed=False)
                    crossline_flat, inline_flat = obj.flatness(transformed=False)
                    cursor.execute("EXEC Insert_Symm_Flat_Data ?,?,?,?,?",
                                   [mpc_event_id, 1, 0, inline_symm, inline_flat])
                    cursor.commit()
                    cursor.execute("EXEC Insert_Symm_Flat_Data ?,?,?,?,?",
                                   [mpc_event_id, 0, 0, crossline_symm, crossline_flat])
                    cursor.commit()

            # extract the MPC self-reported results in the excel file
            f = open(results_file)
            csv_reader_object = csv.reader(f)

            # structure of these is [name, value, threshold, pass/fail, date]
            for line in csv_reader_object:
                name = str(line[0])
                if name.find("BeamOutputChange") != -1:
                    cursor.execute("EXEC Insert_MPC_Output_Change_Data ?,?", [mpc_event_id, line[1]])
                if name.find("BeamUniformityChange") != -1:
                    cursor.execute("EXEC Insert_Uniformity_Change_Data ?,?", [mpc_event_id, line[1]])
                if name.find("BeamCenterShift") != -1:
                    cursor.execute("EXEC Insert_Beam_Center_Shift_Data ?,?,?", [mpc_event_id, line[1], centre_shift])
                cursor.commit()
    except FileNotFoundError:
        pass


# loop through files to extract the data
for folder in os.listdir(dir):
    path = os.path.join(dir, folder)
    filename = str(path)
    if filename.find("6x-") != -1:
        historic("6x", path)
    if filename.find("10x-") != -1:
        historic("10x", path)
    if filename.find("10xFFF-") != -1:
        historic("10fff", path)

