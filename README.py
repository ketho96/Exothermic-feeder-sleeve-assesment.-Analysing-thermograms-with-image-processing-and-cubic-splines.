# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:05:11 2022

This file is the basis for the screenshots in the bachelor thesis. It is 
exclusively written in english so that an explanation of the most
importat parts of the code is provided. Please use the search function
to find explanations to code parts found in the other python skripts. I tried
my best to meet PEP8 code design standards in this file. I am sorry that I
didn't do it for the rest of the skripts. In the folder "examples" are all 
required files to run every skript to get a feeling of the capabilities. 

For further questions feel free to contact me via these email adresses:
   
kevinthomas@live.de (primary, private)
k.thomas@student.uni-kassel.de (university email valid till 03/2023)
backup: coolio962@live.de


GER:

Hier soll der Code sauber dargestellt und mit Kommentaren belegt werden


@author: Kevin Thomas
"""
import numpy as np
import pandas as pd
from scipy import LSQUnivariateSpline
import cv2


STD_BIN_SIZE = 20
CURVE_FIT = 100
FEEDER_COOLDOWN_CRITERIA = 100
QA_INTENSITY = 0.5
# =============================================================================
# Mockup variables
# =============================================================================
cap = 'mock'
list_all_frames = ['mocklist']
roiframe = 'mock'
mask_vol_mean = 'mock'
regionOfInterest = 'mock'
mask_spline_max = 'mock'

# =============================================================================
# Initialization
# =============================================================================
number_of_frames_above_FEEDER_COOLDOWN_CRITERIA_after_max_array = [0]
init_array = np.zeros(total_number_of_frames)
result_number_of_reacted_pixels = init_array
result_number_of_finished_pixels = init_array
result_sum_of_reacted_pixels = init_array
result_sum_of_finished_pixels = init_array
result_sum_of_total = init_array
point_of_ignition_image = np.zeros((row,column), 'uint16')
qa_textfile_container = ''



# =============================================================================
# Read the Videofile and create a 3-dimensional array from it
# =============================================================================

fn = 0
while(cap.isOpened()):
    
    # Read next frame
    ret, frame = cap.read()
    if ret:
        # As long as there are more frames to read
        uniframe=cv2.resize(frame, (800,600))
        # Reduce color channels from 3 to 1
        roigray = cv2.cvtColor(uniframe, cv2.COLOR_BGR2GRAY)
        # Accumulate all the frames in a list of 2D arrays
        list_all_frames.append(roiframe)
        fn += 1
    else:
        cap.release()
        cv2.destroyAllWindows()
pixel_vals_all_frames = np.array(list_all_frames)
# Reshape the array so that dim0 = rows, dim1 = columns, dim2 = time(frames)
pixel_vals_all_frames = pixel_vals_all_frames.transpose(1,2,0)
total_number_of_frames = pixel_vals_all_frames.shape[2]

# =============================================================================
# Iterate over every pixel and get the gray values over all frames
# =============================================================================

d = 0
r = 0
while d < pixel_vals_all_frames.shape[0]:
    r = 0
    while r < pixel_vals_all_frames.shape[1]:
        array_container = pixel_vals_all_frames[d,r,:]
        r += 1
    d += 1















    
# =============================================================================
# Divide the values into equally sized bins and calculate std. deviation    
# =============================================================================

# Reshape the array so that "STD_BIN_SIZE" number of elements are in one column
# STD_BIN_SIZE is a manually picked int defining the bin size
array_container_conv = np.reshape(array_container,
    (round(len(array_container)/STD_BIN_SIZE),STD_BIN_SIZE)) 
# Convert ndarray into pandas dataframe
df_array_container_conv = pd.DataFrame(array_container_conv)
df_array_container_conv[df_array_container_conv==0] = np.NaN 
# Calculate std. deviation along axis1 but ignore NaN
df_std = df_array_container_conv.std(axis=1, skipna=True) 
# Calculate the mean of all std. devs and save it to the result array
mask_vol_mean[d,r] = df_std.mean() 






# =============================================================================
# Create an appendix with the last measured value for 5*CURVE_FIT elements
# =============================================================================

# CURVE_FIT is a manually picked int defining the intervals of the knots
# Get the last measured value
appendix = array_container[-1] 
appendix_list=[]
i = 0
while i < CURVE_FIT*5: 
    appendix_list.append(appendix)
    i += 1
array_container = np.append(array_container, appendix_list)













# =============================================================================
# Create the spline
# =============================================================================

# CURVE_FIT is a manually picked int defining the intervals of the knots
len_array_container = array_container.shape[0]-1
# Create the knots from the end with CURVE_FIT intervals
knots = np.array(range((len_array_container-CURVE_FIT), CURVE_FIT,-CURVE_FIT)) 
# Reverse the knot order
knots = knots[::-1] 

# Create a x-value for every datapoint/frame
x_values = np.linspace(0, len_array_container,len_array_container) 
# Create a LSQUnivariateSpline object
spline = LSQUnivariateSpline(
    x_values, array_container[0:len_array_container], knots)
# Write the approximated values into the y_spline object
y_spline = spline(x_values)



# =============================================================================
# Background segmentation
# =============================================================================

# Initialize a 2D array with the size of 'region of interest'
mask = np.zeros((regionOfInterest[3] 
                 - regionOfInterest[2], regionOfInterest[1] 
                 - regionOfInterest[0])).astype('ubyte')

d = 0
r = 0
while d < pixel_vals_all_frames.shape[0]:
    r = 0
    while r < pixel_vals_all_frames.shape[1]:
        if (mask_vol_mean[d,r] == 0 or
            mask_spline_max[d,r] < 80 and
            mask_vol_mean[d,r] > 2 or
            mask_vol_mean[d,r] > 4 or
            mask_spline_max[d,r] < 70): 
            # Criteria defining what is NOT part of the feeder-surface
            r += 1
        else:
            # Set pixels belonging to the feeder-surface to 255 = white
            mask[d,r] = 255
            r += 1
    d += 1
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    

    















# =============================================================================
# Analyse thermophysical data
# =============================================================================

# QA_INTENSITY is a manually picked float (0 - 1) defining the color intensity
# FEEDER_COOLDOWN_CRITERIA is a manually selected condition
# for the time at which the value is considered to be cooled down 
# after the reaction

# Create a new UnivariateSpline object 
spline_deriv = spline.derivative()
# Write the approximated values into the y_spline_deriv object
y_spline_deriv = spline_deriv(x_values)

# [Legacy] Detection of the exothermic reaction 
frame_max_deriv = round((int((np.where(y_spline_deriv 
                                       == max(y_spline_deriv)))[0])))

# [Advanced] Detection of the exothermic reaction
max_spline_value = y_spline.max()
count = -1
# Sort derivative values ascending
sorted_deriv_values = np.sort(y_spline_deriv)
found = False
while not(found):
    # Check beginning with the largest derivative value
    deriv_value = sorted_deriv_values[count]
    frame_max_deriv = round((int((np.where(y_spline_deriv 
                                           == deriv_value))[0])))
    # Check if 70% of the maximum spline value has been reached
    if not(array_container[frame_max_deriv] >= 0.7*(max_spline_value)):
        # Check for the next lower derivative
        count -= 1
    else:
        found = True
try:
    # Document the point of the exothermic reaction 
    result_number_of_reacted_pixels[frame_max_deriv] += 1
    point_of_ignition_image[d,r] = frame_max_deriv
    # Document if the detection was successful
    qa_array_react[d,r,1] = 255*QA_INTENSITY #GREEN
except:
    qa_array_react[d,r,2] = 255*QA_INTENSITY #RED
    qa_textfile_container += (f'index is out of bounds.\
                              Reaction not found @ d:{d}/r:{r} RED\n')
# Frame where the spline reaches it's maxmimum value
frame_max_spline = round((int(((np.where(y_spline == max_spline_value)[0])[0]))))  
# Create a seperate section of the spline after the reaction
x_values_after_max_deriv = np.linspace(frame_max_deriv,len_array_container, 
                                       (len_array_container-frame_max_deriv)) 
y_spline_after_max_deriv = spline(x_values_after_max_deriv)
# Create a seperate section of the spline after the maximum was reached
x_values_after_max_spline_value = np.linspace(
    frame_max_spline, len_array_container, (len_array_container - frame_max_spline)) 
y_spline_after_max = spline(x_values_after_max_spline_value)



# Calculate the number of frames a pixel is considered reacting
number_of_frames_above_FEEDER_COOLDOWN_CRITERIA_after_max = len(
    y_spline_after_max[y_spline_after_max >= FEEDER_COOLDOWN_CRITERIA])
# Save the calculated values for every pixel in an array
number_of_frames_above_FEEDER_COOLDOWN_CRITERIA_after_max_array = np.vstack(
    number_of_frames_above_FEEDER_COOLDOWN_CRITERIA_after_max_array, 
    number_of_frames_above_FEEDER_COOLDOWN_CRITERIA_after_max) 
try:
    frame_below_FEEDER_COOLDOWN_CRITERIA_after_max = round(
        int(((np.where(y_spline_after_max < FEEDER_COOLDOWN_CRITERIA))[0])[0]))
    # Document the end of reaction
    result_number_of_finished_pixels[
        frame_below_FEEDER_COOLDOWN_CRITERIA_after_max + frame_max_spline] -= 1 
    # GREEN 255 -> success
    qa_array_finish[d,r,1] = 255*QA_INTENSITY 
# IndexError is raised, when FEEDER_COOLDOWN_CRITERIA is not met 
# within total_number_of_frames
except IndexError: 
    # Assume the end of reaction at the end of the test run
    result_number_of_finished_pixels[total_number_of_frames - 1] -= 1 
    qa_textfile_container += (f'Indexerror End of reaction not found,\
                              assumed at the end @ d:{d}/r:{r} RED\n')
    # RED 255 -> failure
    qa_array_finish[d,r,2] = 255*QA_INTENSITY
    if SHOW_PROGRESS:
        print(f'Indexerror End of reaction not found, assumed at the end @ d:{d}/r:{r} RED\n')
# If any other error is raised, print the error
except Exception as e: 
    print(e)
    # RED 255 -> failure
    qa_array_finish[d,r,0] = 255*QA_INTENSITY
    qa_textfile_container += (f'COMPLETE FAILURE @ d:{d}/r:{r} BLUE\n')
    if SHOW_PROGRESS:
        print(f'COMPLETE FAILURE @ d:{d}/r:{r}')
    pass

# =============================================================================
# Summarize the number_of_XXX_pixel arrays 
# =============================================================================

i=0
while i < len(result_number_of_reacted_pixels):
    result_sum_of_reacted_pixels[i]= sum(result_number_of_reacted_pixels[0:i+1])
    i+=1

i=0
while i < len(result_number_of_reacted_pixels):
    result_sum_of_finished_pixels[i]= sum(result_number_of_finished_pixels[0:i])
    i+=1

i=0
while i < len(result_number_of_reacted_pixels):
    result_sum_of_total[i]=result_sum_of_reacted_pixels[i]+result_sum_of_finished_pixels[i]
    i+=1 



# =============================================================================
# Read data and evaluate the point of ingnition
# =============================================================================

# Load the corresponding result file
result_container = np.load(file)
# Set all values that are smaller than THRESH_PERCENTAGE = 0
result_container[result_container<=((THRESH_PERCENTAGE/100)*mask_size)] = 0 
# Get the point of ignition in frames
point_of_ignition = result_container.size - np.count_nonzero(result_container) 
# Reduce the file name to the file_save_name
key = file.replace('__result_sum_of_reacted_pixels.npy','') 
# Assign each dataset a point_of_ignition
ignition_dict[f'{key}'] = point_of_ignition
# Convert absolute values into percentages 
result_container = (np.divide(result_container,mask_size))*100









































# =============================================================================
# Output video creation
# =============================================================================

out = cv2.VideoWriter(f'F:/{resultdir}/{file_save_name}\
                      Glättung{curve_fit_spline}_output_video.avi',
                      cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize) 
# Initialize a 3 channel image
area_image  = np.zeros((row,column,3), 'uint8')

fn = 0
while(cap.isOpened()):
    # As long as there are more frames to read (orig video)
    ret, frame = cap.read()
    if ret:
        # Segmented monochrome camera image
        roigray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Initialize an Image
        ignition_image  = np.zeros((row,column), 'uint8')
        # Identify all pixels which react at the current frame count fn 
        # and set them to 255 
        ignition_image[point_of_ignition_image == fn] = 255 
        # Convert the one channel image to a three channel image (BGR)
        ignition_image = cv2.cvtColor(ignition_image, cv2.COLOR_GRAY2BGR) 
        # Convert the one channel image back to a three channel image (BGR)
        roiframe_BGR = cv2.cvtColor(roigray, cv2.COLOR_GRAY2BGR) 
        # Set BLUE and GREEN channels to 0 which leaves only RED = 255
        ignition_image[:,:,0] = 0
        ignition_image[:,:,1] = 0
        area_image = area_image+ignition_image
        # Apply the segmentation mask
        area_image[uniSmask == 0] = 0
        # Overlay the red ignition visualization onto the original frame
        output_frame = cv2.addWeighted(output_frame,1,area_image,0.1,0)                            
        # Write the output frame to the video object
        out.write(output_frame) #schreibe in das video
        fn += 1
    else:
        cap.release()
        out.release()
        cv2.destroyAllWindows()















# =============================================================================
# Create boxplot and performance comparison data based on mask_spline_max
# =============================================================================

# Convert from float to int
array_container = np.load(file).astype(int) 
# Reduce dimensions from 2 to 1
array_container = np.ndarray.flatten(array_container) 
# Define boxplot settings
UPPER_LIMIT = 255
LOWER_LIMIT = 25 
STEP = 5 
bins = np.arange(LOWER_LIMIT,upper_limit,STEP)
# Assign the values of array_container to the specified bins
histogram_container, bins = np.histogram(array_container, bins = bins,
                                        range = (LOWER_LIMIT, upper_limit))
boxplot_array = histogram_container
# Stack the bin values for each measurement of one specific Feeder_Type; 
# this will be the basis for the boxplot graph
boxplot_array = np.vstack((boxplot_array,histogram_container)) 
# Calculate the mean for each bin for one Feeder_Type configuration 
# (i.e. fresh, aging, etc.)
boxplot_mean = np.mean(boxplot_array,axis=0)
# Create an array of all boxplot_mean arrays which will be the basis 
# for the performance_comparision
boxplot_mean_summarized = np.vstack((boxplot_mean_summarized, boxplot_mean)) 





