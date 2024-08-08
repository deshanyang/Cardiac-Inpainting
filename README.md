# Cardiac-Inpainting

This repository contains a fully trained GAN model for the purpose of inpainting metal artifacts caused by ICD leads in cardiac CT images. The model is based on the Pix2Pix network whose README.md file can be found within the code folder and can also be accessed via this link: https://github.com/matlab-deep-learning/pix2pix. 

To use this model, first run the install.m file found in the /code folder. Once the install has been done add the code/Inpainting_code to your matlab path and open up the CardiacInpainting.m file found within that folder. The file is set up to run the model on the testing dataset used in the following paper: __________________. Additionally there is section below to apply the model to your own data, where all that is needed is the raw images and associated artifact contours. 
