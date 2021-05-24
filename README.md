# Image_venv
Image analysis project within a virtual environment

The web-interface provides an area for the user to upload the EPID image from the MPC beam profile check in .xim format and returns a plots display of the central profiles in the inline and crossline direction. 
The dose measurements have been calibrated to a transformation matrix acquired by calculating the ratio between EPID pixel values and dose offset ratios from the SNC water tank.

To use:

git clone repository

activate the virtual environment by running
env\Scripts\activate

run the server
python manage.py runserver

navigate to localhost/6x for 6x beam energy, 10x or 10fff as required

Author: Nicola Compton
Date:
Contact: nicola.compton@ulh.nhs.uk
