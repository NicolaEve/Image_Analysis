""" This script will be run locally to extract the historic data from Aria image
Analysis using my calibration will be run and compared to MPC results
Then I can export this data for statistical analysis and reporting on bias,
sensitivity, specificity, reliability
The reason it will only run locally is that Aria image is password protected """

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

for folder in os.listdir(dir):
    path = os.path.join(dir, folder)
    filename = str(path)
    if filename.find("6x-") != -1:

        # get the beam profile check data
        i = 0
        for file in os.listdir(path):
            if i == 0:
                # get the creation date of the first file in the folder
                first_path = os.path.join(path, file)
                created = os.stat(first_path).st_ctime
                date = datetime.fromtimestamp(created)

                # enter date and beam energy to database
                # return the mpc event id
                statement = """declare @mpc_event_id int;
                                    EXEC @mpc_event_id = Insert_MPC_Event ?,?;
                                    SELECT @mpc_event_id as mpc_event_id; """
                cursor.execute(statement, ["6x", date])
                mpc_event_id = cursor.fetchval()
                cursor.commit()
                i = 1
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
                        # apply the calibration for the beam energy

                        obj = main.NewImages("6x", image_file)
                        crossline, inline = obj.apply_calibration()
                        crossline_symm, inline_symm = obj.symmetry(transformed=True)
                        crossline_flat, inline_flat = obj.flatness(transformed=True)
                        centre_shift = obj.centre_shift()
                        # enter into database
                        cursor.execute("EXEC Insert_Symm_Flat_Data ?,?,?,?,?", [mpc_event_id, 1, 1, inline_symm, inline_flat])
                        cursor.commit()
                        # repeat for crossline
                        cursor.execute("EXEC Insert_Symm_Flat_Data ?,?,?,?,?", [mpc_event_id, 0, 1, crossline_symm, crossline_flat])
                        cursor.commit()

                        # repeat for untransformed data, before calibration applied
                        # 1 for inline
                        # 1 for transformed
                        crossline_symm, inline_symm = obj.symmetry(transformed=False)
                        crossline_flat, inline_flat = obj.flatness(transformed=False)
                        cursor.execute("EXEC Insert_Symm_Flat_Data ?,?,?,?,?",
                                       [mpc_event_id, 1, 0, inline_symm, inline_flat])
                        cursor.commit()
                        # repeat for crossline
                        cursor.execute("EXEC Insert_Symm_Flat_Data ?,?,?,?,?",
                                       [mpc_event_id, 0, 0, crossline_symm, crossline_flat])
                        cursor.commit()


                # compare to the MPC results
                # extract the reported results in the excel file
            if str(file) == "Results.csv":
                results_file = os.path.join(path, file)
                f = open(results_file)
                csv_reader_object = csv.reader(f)

                # structure of these is [name, value, threshold, pass/fail, date]
                for line in csv_reader_object:
                    name = str(line[0])
                    if name.find("BeamOutputChange") != -1:
                        # test_id = 1
                        cursor.execute("EXEC Insert_MPC_Output_Change_Data ?,?", [mpc_event_id, line[1]])
                    if name.find("BeamUniformityChange") != -1:
                        cursor.execute("EXEC Insert_Uniformity_Change_Data ?,?", [mpc_event_id, line[1]])
                        # test id = 2
                    if name.find("BeamCenterShift") != -1:
                        cursor.execute("EXEC Insert_Beam_Center_Shift_Data ?,?,?", [mpc_event_id, line[1], centre_shift])
                        # test id = 3
                    cursor.commit()

                # put that all in a database, so then can put into a spreadsheet
                # spc on the mpc results


# repeat for 10x and 10fff

  #  if filename.find("10x-") != -1:
     #   if filename.find("10xFFF-") != -1:
