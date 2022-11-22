# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 20:06:56 2021

@author: User
"""








# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 14:09:44 2021


 parent v3
 change log:
     zusätzlich zum maximalen anstieg, muss zum zeitpunkt des maximalen anstiegs auch ein bestimmter prozentualer wert des maximums erreicht werden, sonst wird der nächst steilere anstieg genommen

@author: User
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os.path as os
import scipy 
from scipy.interpolate import  LSQUnivariateSpline
from scipy.interpolate import  UnivariateSpline
import scipy.optimize as spo
#from scipy.optimize import minimize_scalar
from scipy.optimize import basinhopping
from math import sin
import matplotlib

show_OVERVIEW = False
show_OVERVIEW_intervals = False

show_std = False
show_spline_deriv = False
deriv_selector = 0 # pick which grade of derivative u want to plot 0 = 1st deriv
show_result_sum_of_finished_pixels = False
show_values_after_max_deriv_only = False
show_raw_data = False

coordinatesSpeiser = {369:392,383:372,283:362}
coordinatesFlammen = {}
#VideoDateiName = 'Speisermessung_Foseco_Kalminex_ZF_7_9_Frisch_1.2.wmv'
VideoDateiName = 'Speisermessung_Chemex_CB_40_GK70_100_R_1.2.wmv'


duration = 9620
startFrame = 12
curve_fit_spline =100 #regler für den grad der glättung des splines, je größer desto gröber
curve_fit_vol = 20



videoName=VideoDateiName
coordinates_list = [coordinatesSpeiser,coordinatesFlammen]
o = 0
c = 'S' 
while o < 2:
    for x, y in coordinates_list[o].items():
        print(c,o)
        xS = x
        yS = y 
        values = np.load(f'{videoName}duration{duration}startframe{startFrame}valuesx{c}{xS}y{c}{yS}.npy')
        #Zündzeitpunkt = 6 # in Sekunden
        #Brenndauer =  7240 #gerade in frames sonst umschalten unten
        #Schwellwert = 50 # bit, kann auch in Grad Celsius sein, wenn der Ausdruck in tempthresh aktiviert wird
        #Glättungsfilter = 9 # siehe Erläuterungen
        #MorphologischerOperator = 9 # siehe Erläuterungen
        #threshold = 75 #grenzwert, den ein pixel über die versuchdurchführung mindestens erreichen muss, um als speiser gewertet zu werden
        
        
        
        
        
         
         
        #videoName = VideoDateiName
        #maskName = MaskenDateiName
        #startFrame = Zündzeitpunkt*25
        #endFrame = startFrame ++ Brenndauer
        #GB = Glättungsfilter
        #KO = MorphologischerOperator
        
           
        #duration = endFrame-startFrame #duration, dauer des videos
        
        
        
        
        #dataFolder = 'Data'
        
        #open(f'{videoName}duration{duration}startframe{startFrame}valuesxS{xS}yS{yS}.npy')
        #values = np.load(f'{videoName}duration{duration}startframe{startFrame}valuesxS{xS}yS{yS}.npy')
        
        
        
        len_values_init=values.shape[0]-1
        
        
        result_number_of_reacted_pixels = np.zeros(len(values))
        result_number_of_finished_pixels = np.zeros(len(values))
        result_sum_of_reacted_pixels = np.zeros(len(values))
        result_sum_of_finished_pixels = np.zeros(len(values))
        result_sum_of_total = np.zeros(len(values))
        
        x= 100
        i=0
        d=5
        #while i<= x:
            #values[(len_values-x)+i]=values[len_values-x]
            #i+=1
           
        appendix = values[-1]
        appendix_list=[]
        
        curve_fit = curve_fit_spline
        while i< curve_fit*5:
            appendix_list.append(appendix)
            
            i+=1
        
# =============================================================================
#         NO APPENDIX
# =============================================================================
        #values=np.append(values, appendix_list)
        
        
        print(values.shape[0])
        len_values=values.shape[0]-1
        print(len_values)
        #knots=np.array(range(curve_fit,len_values,curve_fit))
        knots=np.array(range((len_values-curve_fit),curve_fit-1,-curve_fit))
        #knots=np.array(range((len_values-curve_fit-x),curve_fit,-curve_fit))
        knots = knots[::-1]
        #print(knots)
        
        #x_values =   np.linspace(0,  len_values,len_values)
        #x_spline = x_values*100
        x_spline =   np.linspace(0, (len_values,len_values)*100)
        x_values10 = np.linspace(0, (len_values,len_values)*10)
        x_values=np.linspace(0, len_values,len_values)
        
        
        
        
        
        
        #knots=np.linspace(curve_fit, len_values-curve_fit,(round(len_values/curve_fit)))
        #print(knots[-2:])
        
        #start=knots[-1]
        #print(start)
        
        #end_knots=np.array(range(len_values-d,start,-d))
        
        #end_knots = end_knots[::-1]
        #print(knots)
        #print(end_knots)
        #knots= np.append(knots, end_knots)
        #print(knots)
        
        
        print(values.shape)
        print(x_values.shape)
        
        
        #print(len(x_values))
        #print(len(values))
        
        #spline = LSQUnivariateSpline(x_values, values[0:len_values], knots)
        spline = LSQUnivariateSpline(x_values, values[0:len_values], knots)
        #spline = UnivariateSpline(x_values, values[0:len_values], s=0)
        spline_deriv = spline.derivative()
        
        spline_deriv_antideriv = spline_deriv.antiderivative()
        
        
        #spline = UnivariateSpline(x_values, values,knots)
        x_spline = np.linspace(0, len_values,len_values*100)
        
        #y_spline = spline(x_spline)
        y_spline = spline(x_values)
        
        
        y_spline_deriv = spline_deriv(x_spline)
        y_spline_deriv = spline_deriv(x_values)
        y_spline_deriv_antideriv = spline_deriv_antideriv(x_spline)
        
        curve_fit= curve_fit_vol
        
        values_conv=values
        values_conv = np.reshape(values_conv,(round(len(values_conv)/curve_fit),curve_fit)) #schreibe immer curve_fit anzahl an werten in eine spalte 
        df_values_conv = pd.DataFrame(values_conv)
        df_values_conv[df_values_conv==0]=np.NaN #ändere alle nullen in NaN um diese bei der durchschnittsberechnung zu ignorieren
        df_std = df_values_conv.std(axis=1, skipna=True) #berechne die standardabweichung jeder spalte (also curve_fit anzahl an werten)
        df_std_mean = df_std.mean() #gib von allen standardabweichungen aller bins XXXXXX aus
        
        
        res =  [ele for ele in df_std for i in range(curve_fit)]
        x_std=range(len(res))    
        
        
        """
        fig = plt.figure()
        
        ax = fig.subplots()
        ax.plot(np.arange(values.shape[0]),values, color = 'b', alpha=0.5)
        ax.plot(x_values,y_spline, color = 'r') 
        #ax.plot(x_values,y_spline_deriv, color = 'g')
        
        fig = plt.figure()
        plt.title('std dev')
        #plt.title(f'Frame:{round(int(result[0]))/curve_fit},MaxDeriv:{max(spline_deriv(np.linspace(0, len(values),len(values)*10)))}')
        #plt.legend()
        plt.ylim(-10, 200)
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.grid()
        plt.show()
        """
        
        if show_std:
            fig = plt.figure()
            ax = fig.subplots()
            ax.plot(x_std,res, color = 'g') 
            plt.title(f'std devation. Mean = {df_std_mean} Type:{c} coordinates : {xS}:{yS}')
            plt.xlabel('Frame')
            plt.ylabel('Value')
            plt.grid()
            plt.savefig(f'std devation. Mean = {df_std_mean} Type={c} coordinates = {xS}%{yS}.png')
            plt.show()
        
        
        
        if show_spline_deriv:
            fig = plt.figure()
            ax = fig.subplots()
            
            spline_2nd_deriv = spline_deriv.derivative()
            spline_3rd_deriv = spline_2nd_deriv.derivative()
            
            y_spline_2nd_deriv = spline_2nd_deriv(x_values) #linear
            y_spline_3rd_deriv = spline_3rd_deriv(x_values) # constant
            
            deriv_list = [y_spline_deriv,y_spline_2nd_deriv, y_spline_3rd_deriv]
            
            
            ax.plot(x_values,deriv_list[deriv_selector], color = 'g') 
            plt.title(f'spline derivativeType:{c} coordinates : {xS}:{yS}')
            
            #plt.title(f'Frame:{round(int(result[0]))/curve_fit},MaxDeriv:{max(spline_deriv(np.linspace(0, len(values),len(values)*10)))}')
            #plt.legend()
            plt.ylim(-0.5, 0.5)
            plt.xlabel('Frame')
            plt.ylabel('Value')
            plt.grid()
            plt.savefig(f'spline derivativeType={c} coordinates = {xS}%{yS}.png')
            plt.show()
        
        array_container=values
        
        print(len(x_values))
        print(len(array_container))
        
        
        
        #array_container = values[d,r,:]
        #x_values=np.linspace(0, len(array_container),len(array_container))
        #spline = LSQUnivariateSpline(x_values, values[0:len_values], knots)
        #spline_deriv = spline.derivative()
        #x_spline = np.linspace(0, len(array_container)-1,len(array_container)*100)
        print(((np.where(y_spline_deriv == max(y_spline_deriv)))[0]))  
        
        frame_max_deriv = round((int((np.where(y_spline_deriv == max(y_spline_deriv)))[0])))
        
        print('frame_max_deriv',frame_max_deriv)
        
        
        
        
        
        
        
        
        
        
        
        #frame_max_after_max_deriv = round(int(result[0])/curve_fit)
        
        #Punkt nach dem Beginn der exothermen Reaktion finden, an dem der Wert maximal wird = Ende der Exothermen Reaktion
        x_values_after_max_deriv = np.linspace(frame_max_deriv,len_values,(len_values-frame_max_deriv))
        #x_values_after_max_deriv = np.linspace(round(int(result1[0])/curve_fit),len(array_container),(len(array_container)-round(int(result1[0])/curve_fit)))
        y_spline_after_max_deriv = spline(x_values_after_max_deriv)
        #frame_max_after_max_deriv = round(int((np.where(y_spline_after_max_deriv == max(y_spline_after_max_deriv)))[0]))
        
        
        
        """############################################################################################"""
        
        
        
        
        
        #Punkt nach dem Beginn der exothermen Reaktion finden, an dem der Wert 80% des Maximums erreicht = Ende der Exothermen Reaktion
        frame_max = round((int((np.where(y_spline == max(y_spline)))[0])))
        print('frame abs max value',frame_max)
        x_values_after_max = np.linspace(frame_max,len_values,(len_values-frame_max)) #schneide die Funktionswerte so zu, dass nur Werte nach frame_max_deriv ausgewertet werden
        print('x_values_after_max',x_values_after_max)
        #x_values_after_max_deriv = np.linspace(frame_max_deriv,len_values,(len_values-frame_max_deriv)) #schneide die Funktionswerte so zu, dass nur Werte nach frame_max_deriv ausgewertet werden
        #y_spline_after_max_deriv = spline(x_values_after_max_deriv)
        #y_spline_after_max = spline(x_values_after_max)
        maximum_value = y_spline.max()
        
        
        
        count = -1
        maximum_value = y_spline.max()
        sorted_deriv_values = np.sort(y_spline_deriv) #sortiere aufsteigend
        Found = False
        while not(Found):
            deriv_value = sorted_deriv_values[count]
            frame_max_deriv = round((int((np.where(y_spline_deriv == deriv_value))[0])))
            if not(values[frame_max_deriv] >= 0.7*(maximum_value)):
                count-=1
            else:
                Found=True
                
        result_number_of_reacted_pixels[frame_max_deriv]+=1 
        
        #frame_max = round((int((np.where(y_spline == maximum_value)[0]))))
        
        x_values_after_max_deriv = np.linspace(frame_max_deriv,len_values,(len_values-frame_max_deriv)) #schneide die Funktionswerte so zu, dass nur Werte nach frame_max_deriv ausgewertet werden
        y_spline_after_max_deriv = spline(x_values_after_max_deriv)
        y_spline_after_max = spline(x_values_after_max)
        
        
        
        
        threshold_list = [0.95*maximum_value,0.90*maximum_value,0.85*maximum_value,0.80*maximum_value]
        for threshold in threshold_list:
            number_of_frames_above_threshold_after_max = len(y_spline_after_max[y_spline_after_max >= threshold])
            try:
            
                first_frame_above_threshold = (np.where(y_spline_after_max >= threshold))[0][0]+frame_max
                print('first_frame_above_threshold = frame_max',first_frame_above_threshold)
            except IndexError:
                pass
            try:
                last_frame_above_threshold = (np.where(y_spline_after_max >= threshold))[0][-1]+frame_max
                print('last_frame_above_threshold',last_frame_above_threshold)
            except IndexError:
                pass
            print('number_of_frames_above_threshold',number_of_frames_above_threshold_after_max)
        
        
        
        
        
        
        
        
        
        y_spline_deriv = spline_deriv(x_spline)
        fig = plt.figure()
        ax = fig.subplots()
        plt.title(f'Overview of processing Type={c} coordinates = {xS}:{yS}')
        #plt.axline(y = 44.5, color = 'r', linestyle = '-')
        if show_OVERVIEW_intervals:
            for knot in knots:
                plt.axvline(knot, color ='g',linestyle = 'dashed')

        try:
            #plt.axvline(first_frame_above_threshold, color ='r',linestyle = '--')
            #plt.axvline(last_frame_above_threshold, color ='r',linestyle = '--')
            a=0
        except NameError:
            pass
        
        ax.plot(np.arange(values.shape[0]),values, color = 'b',alpha = 0.5, label = 'Data')
        ax.plot(x_values,y_spline, color = 'b', label ='Spline') 
        ax.plot(x_spline,y_spline_deriv*30, color = 'g', label = 'Spline derivative') 
        ax.plot(x_values_after_max_deriv,y_spline_after_max_deriv, color = 'r', label ='Spline exothermic reaction') 
        
        plt.ylim(-10, 200)
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.grid()
        plt.legend(loc = 'upper left')
        plt.axvline(frame_max_deriv, color = 'r', linestyle ='dashed', label = 'Detection of exothermic reaction')
        plt.savefig(f'Overview of processing Type={c} coordinates = {xS}%{yS} show_intervals={show_OVERVIEW_intervals}.png')
        matplotlib.rcParams.update({'font.size': 20})
        plt.show()
        
        
        
        
        
        print('np where value is below threshold ',(np.where(y_spline_after_max < 0.8*max(y_spline_after_max))[0])[0])
        frame_80pOf_max_after_max = round(int(((np.where(y_spline_after_max < 0.8*max(y_spline_after_max)))[0])[0]))
        print('umkehr',frame_80pOf_max_after_max+frame_max)
        result_number_of_finished_pixels[frame_80pOf_max_after_max+frame_max]-=1
        
        
        #print('frame_80pOf_max_after_max:', frame_80pOf_max_after_max)
        #try:
            #result_number_of_finished_pixels[frame_max_after_max+frame_max]-=1 #erniedrige den zähler der abgeklungenen pixel zum zeitpunkt/frame, der gefunden wurde
        #except:
            #result_number_of_finished_pixels[len_values_init-1]-=1
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
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
        
        
        
        
        
        
        
        
        
        
        if show_result_sum_of_finished_pixels:
            fig = plt.figure()
            ax = fig.subplots()
            #ax.plot(np.arange(values.shape[2]),values[yS,xS,:], color = 'r', alpha=0.5)
            #ax.plot(np.arange(values.shape[2]),values[yF,xF,:], color = 'b', alpha=0.5)
            #ax.plot(x_spline,y_spline_flamme, color = 'g') 
            #ax.plot(x_spline,y_spline_Speiser, color = 'g') 
            plt.title(f'pixel counting/reaction recognition. start = red, reaction = green, finish = blue Type={c} coordinates ={xS}:{yS}')
            ax.plot(np.arange(result_number_of_reacted_pixels.shape[0]),result_sum_of_total, color = 'g') 
            ax.plot(np.arange(result_number_of_reacted_pixels.shape[0]),result_number_of_finished_pixels, color = 'b') 
            ax.plot(np.arange(result_sum_of_reacted_pixels.shape[0]),result_number_of_reacted_pixels, color = 'r')
            
            
            #plt.title(f'X-Coordinate{c}')
            plt.ylim(-2, 2)
            plt.xlabel('Frame')
            plt.ylabel('Value')
            plt.grid()
            plt.savefig(f'pixel counting/reaction recognition. start = red, reaction = green, finish = blue Type={c} coordinates = {xS}%{yS}.png')
            plt.show()
            #print(ret1)
            #print(ret2)
        
        if show_values_after_max_deriv_only:
            fig = plt.figure()
            ax = fig.subplots()
            plt.title(f'values after max derivative only Type={c} coordinates = {xS}:{yS}')
            ax.plot(x_values_after_max,y_spline_after_max, color = 'g') 
            
            plt.ylim(-10, 200)
            plt.xlabel('Frame')
            plt.ylabel('Value')
            plt.grid()
            plt.savefig(f'values after max derivative only Type={c} coordinates = {xS}%{yS}.png')
            plt.show()
            
        if show_raw_data:
            fig = plt.figure()
            ax = fig.subplots()
            plt.title(f'Raw data Type={c} coordinates = {xS}:{yS}')
            ax.plot(values, color = 'b') 
            
            plt.ylim(-10, 200)
            plt.xlabel('Frame')
            plt.ylabel('Value')
            plt.grid()
            plt.savefig(f'Raw data Type={c} coordinates = {xS}%{yS}.png')
            plt.show()
        
        if show_OVERVIEW:
            y_spline_deriv = spline_deriv(x_spline)
            fig = plt.figure()
            ax = fig.subplots()
            plt.title(f'Overview of processing Type={c} coordinates = {xS}:{yS}')
            #plt.axline(y = 44.5, color = 'r', linestyle = '-')
            if show_OVERVIEW_intervals:
                for knot in knots:
                    plt.axvline(knot, color ='r',linestyle = '-')
            plt.axvline(knots[-2], color ='r',linestyle = '--')
            plt.axvline(knots[-3], color ='r',linestyle = '--')
            ax.plot(np.arange(values.shape[0]),values, color = 'b', label = 'Data')
            ax.plot(x_values,y_spline, color = 'r', label ='Spline') 
            ax.plot(x_spline,y_spline_deriv, color = 'g', label = 'Spline derivative') 
            ax.plot(x_values_after_max_deriv,y_spline_after_max_deriv, color = 'r') 
            plt.vlines(frame_max_deriv,0,255, color = 'cyan', linestyle = 'dashed')
            
            legend = plt.legend()
            # =============================================================================
            # Keine Duplikate in der Legende
            # https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
            # =============================================================================
            
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            
            
            plt.ylim(-10, 200)
            plt.xlabel('Frame')
            plt.ylabel('Value')
            plt.grid()
            
            plt.savefig(f'Overview of processing Type={c} coordinates = {xS}%{yS} show_intervals={show_OVERVIEW_intervals}.png')
            plt.show()
    c = 'F'    
    o+=1
