# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:53:40 2022

@author: Hiwi


Erstelle ein Boxplot für jede Messreihe und erstelle ein Graph in der die Durchschnittswerte der Histogramme der einzelnen Messreihen miteinander verglichen werden
"""

# https://stackoverflow.com/questions/10640114/overlay-two-same-sized-images-in-python

import os
from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

save_Y_N = True # Schalter, ob die übereinandergelegten Hsitogramme gespeichert werden sollen
Dont_Plot = False



#i=18
    

Feeder_Type_List = ['ASK','Feedex_Hoch','GK70','ZF9_12','KIM_10_13','Kalminex_7_9','L130','ASK_ADS_61','Feedex_Klein','KIM_30_95','Speisermessung_Kalminex_',
                    'ASK_KIM_10_13_Frisch','ASK_KIM_10_13_30_95_3G','ASK_KIM_10_13_R', #11-13
                    'Chemex_CB_40_GK70_100_R','Chemex_CB_40_GK70_100_30_95_7D','Chemex_CB_40_GK70_100_30_95_7G', #14-16
                    'Foseco_Kalminex_ZF_9_12_KLU_R','Foseco_Kalminex_ZF_9_12_KLU_Frisch','Foseco_Kalminex_ZF_9_12_KLU_30_95_7D', #17-19
                    'Foseco_Kalminex_ZF_7_9_Frisch','Foseco_Kalminex_ZF_7_9_30_95_7G','Foseco_Kalminex_ZF_7_9_R', #20-22
                    'Hofmann_L130_36','Hofmann_L130_37','Hofmann_L130_38','Hofmann_L130_39','Hofmann_L130_40', #23-27
                    'Foseco_Feedex_VSK_36_HD3','Foseco_Feedex_VSK_36_HDT','Foseco_Feedex_VSK_36_HD1', #28-30
                    'ASK_ADS_61ExF27_0','ASK_ADS_61ExF27_10','ASK_ADS_61ExF27_20','ASK_ADS_61ExF27_30', #31-34
                    'Chemex_EK_50_80W_CB22','Chemex_EK_50_80W_CB35','Chemex_EK_50_80W_CB43', #35-38      'Chemex_EK_50_80W_CB31' exkludiert
                    'Hofmann_LK95_R2909', 'Hofmann_LK95_R0601', 'Hofmann_LK95_R1102', 'Hofmann_LK95_R1301', 'Hofmann_LK95_R1310', #39-43
                    'Hofmann_LK95_R1401', 'Hofmann_LK95_R1410', 'Hofmann_LK95_R1510', 'Hofmann_LK95_R1901','Chemex_Tele_80-17_R1201_C_oben', #44-48
                    'Chemex_Tele_80-17_R1201_C_unten', 'Chemex_Tele_80-17_R1201_E_oben', 'Chemex_Tele_80-17_R1201_E_unten', 'Chemex_Tele_80-17_R1201_G_oben', #49-52
                    'Chemex_Tele_80-17_R1201_G_unten', 'Chemex_Tele_80-17_R1201_X_oben', 'Chemex_Tele_80-17_R1201_X_unten', 'Chemex_Tele_80-17_R1201_Y_oben', #53-56
                    'Chemex_Tele_80-17_R1201_Y_unten', 'Chemex_Tele_80-17_R2311_A_oben', 'Chemex_Tele_80-17_R2311_A_unten', 'Chemex_Tele_80-17_R2311_B_oben', #57-60
                    'Chemex_Tele_80-17_R2311_B_unten', 'Foseco_V38_R0601_A', 'Foseco_V38_R0601_B', 'Foseco_V38_R1105', 'Foseco_V38_R1505', 'Foseco_V38_R1610', #61-66
                    'Foseco_V38_R1811' ]#67
Feeder_Name_List = ['ASK_KIM_10_13','Chemex_CB_40_GK70_100','Foseco_Kalminex_ZF_9_12_KLU','Foseco_Kalminex_ZF_7_9','Hofmann_L130', #0-4
                    'Foseco_Feedex_VSK_36','ASK_ADS_61ExF27','Chemex_EK_50_80W','Hofmann_LK95','Foseco_V38'] #5-9



select_Feeder_Type = 20 # 99 = Alle Messriehen eines Typs

#ASK_KIM_10_13_List = ['ASK_KIM_10_13_Frisch','ASK_KIM_10_13_30_95_3G','ASK_KIM_10_13_R']


select_ALL_Feeder_Names = False # Schalter, ob alle Namen bearbeitet werden sollen oder nur ein bestimmter Typ
select_Feeder_Name = 8


for Feeder_Name_Selected in Feeder_Name_List:



    if not select_ALL_Feeder_Names:
        Feeder_Name_Selected = Feeder_Name_List[select_Feeder_Name]
    Feeder_Type_List_Selection = [] #Suche alle Messreihen eines Speisertyps
    for Feeder_Type in Feeder_Type_List:
        if Feeder_Name_Selected in Feeder_Type:
            Feeder_Type_List_Selection.append(Feeder_Type)
            
    print(Feeder_Type_List_Selection)
    
    
    
    
    Feeder_Type = Feeder_Type_List[select_Feeder_Type]
    
    substring = "mask_spline_max.npy" 
    
    rootdir = r'F:\test5'
    resultdir = f'F:/test5/{Feeder_Name_Selected}_Summary'
    print(resultdir)
    
    
    n=0
    labels = []
    histogram_list = []
    for Feeder_Type in Feeder_Type_List_Selection:
        i=0
        print('NEXT')
        
        
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                if substring in file: # Lade nur Dateien, die substring enthalten
                    if Feeder_Type in file:
                        os.chdir(subdir)
                        array_container = np.load(file).astype(int) #array kommt als float rein
                        
                        
                        array_container = np.ndarray.flatten(array_container) #2D array in 1 D array weil histogram funktion nur ein 1D array verarbeiten kann
                        #upper_limit = max(array_container)
                        upper_limit = 255
                        #sns.set() #Stil des diagramms
                        lower_limit = 25 #setzt die Untere Grenze für die Werte, die im Histogramm gezeigt werden, anfangs 25, für foseco auf  gesetzt
                        step = 5 #setzt die Schrittweite der Bins des Histogramms
                        
                        bins = np.arange(lower_limit,upper_limit,step) #erstellt ein array mit bins vom lower limit bis zum upper limit mit schrittweite step
                        
                        histogram_container,bins = np.histogram(array_container,bins = bins, range = (lower_limit,upper_limit))
                        
                        #Bis hierhin funktioniert es. In histogram container ist die anzahl der pixel pro bin gespeichert 1D Array
                        
                        if i==0:
                            Boxplot_Array = histogram_container
                            print('init')
                        else:
                        
                        
                            #print(histogram_container.shape)
                            #print(bins.shape)
                            
                            #plt.hist(array_container, bins = bins, align= 'left')  # 
                            #plt.show()
                
                            #histogram_list.append(histogram_container)
                            Boxplot_Array = np.vstack((Boxplot_Array,histogram_container)) # Stackt die 1D Arrays (entlang der neuen Achse sind die unterscheidlichen Durchläufe)
                        
                        
                            
                        i+=1
        
        
        
        #########speichere von jedem Versuch den Mittelwert der gebinnten maximalen Grauwerte (Entspricht den ordinaten-werten für die zu erzeugende Kurve)##########
        print('first column of Boxplot Array:',Boxplot_Array[:,0])
        Boxplot_mean = np.mean(Boxplot_Array,axis=0) #Durchschnitt der maximalen Grauwerte einer Messreihe
        print('first column mean:',Boxplot_mean[0])
        
        print('Boxplot mean shape',Boxplot_mean.shape)
        if n == 0:
            Boxplot_mean_summarized = Boxplot_mean
            print('Boxplot_mean_summarized.shape', Boxplot_mean_summarized.shape)
        else:
            dummy_array = np.zeros((1,45))
            Boxplot_mean_summarized = np.vstack((Boxplot_mean_summarized, Boxplot_mean)) #speichere jeden Versuch in einer neuen Reihe, durchschnitt der maximalen grauwerte aller Messreihen eines Speisertyps
            print('Boxplot_mean_summarized.shape', Boxplot_mean_summarized.shape)
        
        ######################## Erstelle die Boxplot Graphen ################################
        
        print(Boxplot_Array.shape)       
        #Boxplot_Array= Boxplot_Array.T #trabsponiere die matrix
        print(Boxplot_Array.shape)
        bins = bins[0:-1]          
        
        if Dont_Plot:
            plt.ioff()            
        plt.figure(figsize=(30, 15)) #größes des Plots definieren in inches
        plt.title(label = f'{Feeder_Type} Boxplot n={i}')   
        plt.xlabel('Maximum smoothed gray value')  
        plt.ylabel('Occurences')    
        plt.boxplot(Boxplot_Array, showfliers = True, showmeans= True,patch_artist=True, labels = bins)  #fliers sind die ausreißer  
        plt.ylim((0,25000))
        plt.grid(True)
        #plt.show() 
        resultdir.encode('unicode_escape')
        
        os.chdir(resultdir)
        plt.savefig(f'Boxplot_{Feeder_Type}.png')  
        plt.close()
        
        labels.append(f'{Feeder_Type}') 
        n+=1 
        
    ######### Erstelle die Übersichtsgrafik, in der der Durchschnittswert jeder Versuchreihe verglichen wird
    handles = []
    plt.figure(1)
    plt.figure(figsize=(30, 15)) #größes des Plots definieren in inches
    plt.xlim(20,250)
    plt.xticks(bins)
     
    for Messreihe in Boxplot_mean_summarized:
        print(Messreihe.shape)
        handle, = plt.plot(bins,Messreihe)
        handles.append(handle)
        
      
    print(labels)  
    
    plt.title(label = f'{Feeder_Name_Selected} comparison of mean maximum gray values')   
           
    plt.legend(handles = handles, labels = labels)
    plt.ylim(0,25000)
    plt.xlabel('Maximum smoothed gray value')  
    plt.ylabel('Average number of occurences')
    plt.grid(True)
    plt.savefig(f'Feeder_performance_comparision_Type_{Feeder_Name_Selected}.png') 
    plt.close()
    
    
                
      
    
    
    
