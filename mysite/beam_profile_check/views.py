"""
Views processes user response and redirects user to new displays

Author: Nicola Compton
Date: 24th May 2021
Contact: nicola.compton@ulh.nhs.uk
"""

from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *
from plotly.offline import plot
from plotly.graph_objs import Scatter
from .models import *
from django.views.generic import ListView
from .run_calibration import *
from .main import *
import os
import datetime as dt
import shutil

# set the directory where uploaded images are saved to
media_directory = os.path.join(os.getcwd(), "media\images")
xim_directory = os.path.join(media_directory, "XIMdata")


class Index(ListView):
    template_name = 'templates/index.html'


def beam_energy_6x(request):
    if request.method == 'POST':
        form = BeamEnergy6xForm(request.POST, request.FILES)
        # delete all files saved in there before uploading new one to save storage and ensure we plot from the correct file!
        for file in os.listdir(media_directory):
            if file.endswith(".png") or file.endswith(".xim"):
                os.remove(os.path.join(media_directory, file))

        for file in os.listdir(xim_directory):
            if file.endswith(".png") or file.endswith(".xim"):
                os.remove(os.path.join(xim_directory, file))

        # then save the newly uploaded file
        if form.is_valid():
            form.save()
            return redirect(beam_energy_6x_display_plot)

    else:
        form = BeamEnergy6xForm()
    return render(request, 'BeamEnergy6x.html', {'form': form})


def beam_energy_10fff(request):
    if request.method == 'POST':
        form = BeamEnergy10fffForm(request.POST, request.FILES)
        # delete all files saved in there before uploading new one to save storage
        for file in os.listdir(media_directory):
            if file.endswith(".png") or file.endswith(".xim"):
                os.remove(os.path.join(media_directory, file))

        for file in os.listdir(xim_directory):
            if file.endswith(".png") or file.endswith(".xim"):
                os.remove(os.path.join(xim_directory, file))

        # then save the newly uploaded file
        if form.is_valid():
            form.save()
            return redirect(beam_energy_10fff_display_plot)

    else:
        form = BeamEnergy10fffForm()
    return render(request, 'BeamEnergy10fff.html', {'form': form})


def beam_energy_10x(request):
    if request.method == 'POST':
        form = BeamEnergy10xForm(request.POST, request.FILES)
        # delete all files saved in there before uploading new one to save storage
        for file in os.listdir(media_directory):
            if file.endswith(".png") or file.endswith(".xim"):
                os.remove(os.path.join(media_directory, file))

        for file in os.listdir(xim_directory):
            if file.endswith(".png") or file.endswith(".xim"):
                os.remove(os.path.join(xim_directory, file))

        # then save the newly uploaded file
        if form.is_valid():
            form.save()
            return redirect(beam_energy_10x_display_plot)

    else:
        form = BeamEnergy10xForm()
    return render(request, 'BeamEnergy10x.html', {'form': form})


def beam_energy_10x_display_plot(request):

    if request.method == 'GET':
        # there will only be one file in the media uploads
        # convert files in the media directory from xim to png
        ximmer = os.path.join(media_directory, "ximmerAll.py")
        cmd = str("python " + ximmer + " -d " + media_directory)
        os.system(cmd)

        # there will only be one file in the xim directory
        for file in os.listdir(xim_directory):
            if file.endswith(".png") or file.endswith(".jpeg"):
                filename = os.path.join(xim_directory, file)

        inline, crossline = TransformView("10x", filename).transform()

        plot_div = plot([Scatter(x=inline[0], y=inline[1],
                                 mode='lines', name='test',
                                 opacity=0.8)], output_type='div')
        plot_div_2 = plot([Scatter(x=crossline[0], y=crossline[1],
                                 mode='lines', name='test',
                                 opacity=0.8)], output_type='div')
        return render(request, 'BeamEnergy10xPlot.html',
                      context={'plot_div': plot_div, 'plot_div_2': plot_div_2,
                               'symm_x': inline[2], 'symm_y': crossline[2],
                               'flatness_x': inline[3], 'flatness_y': crossline[3]})


def beam_energy_6x_display_plot(request):

    if request.method == 'GET':
        # convert files in the media directory from xim to png
        ximmer = os.path.join(media_directory, "ximmerAll.py")
        cmd = str("python " + ximmer + " -d " + media_directory)
        os.system(cmd)

        # there will only be one file in the xim directory
        for file in os.listdir(xim_directory):
            if file.endswith(".png") or file.endswith(".jpeg"):
                filename = os.path.join(xim_directory, file)

        inline, crossline = TransformView("6x", filename).transform()

        plot_div = plot([Scatter(x=inline[0], y=inline[1],
                                 mode='lines', name='test',
                                 opacity=0.8)], output_type='div')
        plot_div_2 = plot([Scatter(x=crossline[0], y=crossline[1],
                                 mode='lines', name='test',
                                 opacity=0.8)], output_type='div')
        return render(request, 'BeamEnergy6xPlot.html',
                      context={'plot_div': plot_div, 'plot_div_2': plot_div_2,
                               'symm_x': inline[2], 'symm_y': crossline[2],
                               'flatness_x': inline[3], 'flatness_y': crossline[3]})


def beam_energy_10fff_display_plot(request):

    if request.method == 'GET':
        # convert files in the media directory from xim to png
        ximmer = os.path.join(media_directory, "ximmerAll.py")
        cmd = str("python " + ximmer + " -d " + media_directory)
        os.system(cmd)

        # there will only be one file in the xim directory
        for file in os.listdir(xim_directory):
            if file.endswith(".png") or file.endswith(".jpeg"):
                filename = os.path.join(xim_directory, file)

        inline, crossline = TransformView("10fff", filename).transform()

        plot_div = plot([Scatter(x=inline[0], y=inline[1],
                                 mode='lines', name='test',
                                 opacity=0.8)], output_type='div')
        plot_div_2 = plot([Scatter(x=crossline[0], y=crossline[1],
                                 mode='lines', name='test',
                                 opacity=0.8)], output_type='div')
        return render(request, 'BeamEnergy10fffPlot.html',
                      context={'plot_div': plot_div, 'plot_div_2': plot_div_2,
                               'symm_x': inline[2], 'symm_y': crossline[2],
                               'flatness_x': inline[3], 'flatness_y': crossline[3]})


def most_recent_plot(request):


    # view for displaying plots, pulled from the most recent MPCcheck in ariaimage
    if request.method == 'GET':

        # delete all files saved in there before finding new ones to save storage
        for file in os.listdir(media_directory):
            if file.endswith(".png") or file.endswith(".xim"):
                os.remove(os.path.join(media_directory, file))


        # get files created within the past day
        now = dt.datetime.now()
        ago = now - dt.timedelta(days=2)

        it_6 = 0
        it_10x = 0
        it_10fff = 0
        dir = r"Y:\TDS\H192138\MPCChecks"
        for file in os.listdir(dir):
            path = os.path.join(dir, file)
            st = os.stat(path)
            mtime = dt.datetime.fromtimestamp(st.st_mtime)
            if mtime > ago:
                filename = str(path)
                if filename.find("6x") != -1:
                    for files in os.listdir(path):
                        if str(files).find("BeamProfileCheck") != -1:
                            image_file = os.path.join(path, files)
                            # copy into media folder with a unique name
                            new_name = os.path.join(media_directory, str("6x_" + str(it_6) + ".xim"))
                            shutil.copy(image_file, new_name)
                            it_6 = it_6 + 1

                if filename.find("10x") != -1:
                    if filename.find("10xFFF") !=-1:
                        for files in os.listdir(path):
                            if str(files).find("BeamProfileCheck") != -1:
                                image_file = os.path.join(path, files)
                                new_name = os.path.join(media_directory, str("10xfff_" + str(it_10fff) + ".xim"))
                                shutil.copy(image_file, new_name)
                                it_10fff = it_10fff + 1

                    else:
                        for files in os.listdir(path):
                            if str(files).find("BeamProfileCheck") != -1:
                                image_file = os.path.join(path, files)
                                new_name = os.path.join(media_directory, str("10x_" + str(it_10x) + ".xim"))
                                shutil.copy(image_file, new_name)
                                it_10x = it_10x + 1


    return redirect(beam_energy_10x_display_plot)

        # look for the folders with 10x, 10xFFF and 6x
        # inside these search for BeamProfileCheck.xim
        # copy files into media folder, rename to match the beam energy?

        # apply what is done above:
        # convert xim to png
        # sobel image > find centre > filter profiles > normalise
        # > apply transformation matrix > convert to distance > plot
        # display symmetry
        # write a urls and and html to display all 3 plots and symmetry in one page
        # don't need a model or forms?

        # what if ariaimage is password protected? it won't pull from a server level
        # batch file to copy and put in credenitals -> security, can't be open source

