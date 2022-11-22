# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:53:40 2022

@author: Hiwi


Lege alle Histogramme mit einer Transparenz übereinander um die Streuung zu erkennen (mit abspeichern)
"""

# https://stackoverflow.com/questions/10640114/overlay-two-same-sized-images-in-python

import os
import numpy as np
import matplotlib.pyplot as plt

save_Y_N = False # Schalter, ob die übereinandergelegten Hsitogramme gespeichert werden sollen

    
""" AUSWAHL
Feeder_Type_List = ['ASK_KIM_10_13_Frisch','ASK_KIM_10_13_30_95_3G','ASK_KIM_10_13_R', #0-2
                    'Chemex_CB_40_GK70_100_R','Chemex_CB_40_GK70_100_30_95_7D','Chemex_CB_40_GK70_100_30_95_7G', #3-5
                    'Foseco_Kalminex_ZF_9_12_KLU_R','Foseco_Kalminex_ZF_9_12_KLU_Frisch','Foseco_Kalminex_ZF_9_12_KLU_30_95_7D', #6-8
                    'Foseco_Kalminex_ZF_7_9_Frisch','Foseco_Kalminex_ZF_7_9_30_95_7G','Foseco_Kalminex_ZF_7_9_R', #9-11
                    'Hofmann_L130_36','Hofmann_L130_38','Hofmann_L130_39','Hofmann_L130_40',#11-15
                    'Foseco_Feedex_VSK_36_HD3','Foseco_Feedex_VSK_36_HDT','Foseco_Feedex_VSK_36_HD1', #16-18
                    'ASK_ADS_61ExF27_0','ASK_ADS_61ExF27_10','ASK_ADS_61ExF27_20','ASK_ADS_61ExF27_30', #19-22
                    'Chemex_EK_50_80W_CB22','Chemex_EK_50_80W_CB31','Chemex_EK_50_80W_CB35','Chemex_EK_50_80W_CB43' #22-26
                    ]
"""
Feeder_Type_List = ['ASK_KIM_10_13_Frisch','ASK_KIM_10_13_30_95_3G','ASK_KIM_10_13_R', #0-2
                    'Chemex_CB_40_GK70_100_R','Chemex_CB_40_GK70_100_30_95_7D','Chemex_CB_40_GK70_100_30_95_7G', #3-5
                    'Foseco_Kalminex_ZF_9_12_KLU_R','Foseco_Kalminex_ZF_9_12_KLU_Frisch','Foseco_Kalminex_ZF_9_12_KLU_30_95_7D', #6-8
                    'Foseco_Kalminex_ZF_7_9_Frisch','Foseco_Kalminex_ZF_7_9_30_95_7G','Foseco_Kalminex_ZF_7_9_R', #9-11
                    'Hofmann_L130_36','Hofmann_L130_37','Hofmann_L130_38','Hofmann_L130_39','Hofmann_L130_40',#12-16
                    'Foseco_Feedex_VSK_36_HD3','Foseco_Feedex_VSK_36_HDT','Foseco_Feedex_VSK_36_HD1', #17-19
                    'ASK_ADS_61ExF27_0','ASK_ADS_61ExF27_10','ASK_ADS_61ExF27_20','ASK_ADS_61ExF27_30', #20-23
                    'Chemex_EK_50_80W_CB22','Chemex_EK_50_80W_CB31','Chemex_EK_50_80W_CB35','Chemex_EK_50_80W_CB43', #24-27
                    'Hofmann_LK95_R2909', 'Hofmann_LK95_R0601', 'Hofmann_LK95_R1102', 'Hofmann_LK95_R1301', 'Hofmann_LK95_R1310', #28-32
                    'Hofmann_LK95_R1401', 'Hofmann_LK95_R1410', 'Hofmann_LK95_R1510', 'Hofmann_LK95_R1901','Chemex_Tele_80-17_R1201_C_oben', #33-37
                    'Chemex_Tele_80-17_R1201_C_unten', 'Chemex_Tele_80-17_R1201_E_oben', 'Chemex_Tele_80-17_R1201_E_unten', 'Chemex_Tele_80-17_R1201_G_oben', #38-41
                    'Chemex_Tele_80-17_R1201_G_unten', 'Chemex_Tele_80-17_R1201_X_oben', 'Chemex_Tele_80-17_R1201_X_unten', 'Chemex_Tele_80-17_R1201_Y_oben', #42-45
                    'Chemex_Tele_80-17_R1201_Y_unten', 'Chemex_Tele_80-17_R2311_A_oben', 'Chemex_Tele_80-17_R2311_A_unten', 'Chemex_Tele_80-17_R2311_B_oben', #46-49
                    'Chemex_Tele_80-17_R2311_B_oben', 'Foseco_V38_R0601_A', 'Foseco_V38_R0601_B', 'Foseco_V38_R1105', 'Foseco_V38_R1505', 'Foseco_V38_R1610', #50-55
                    'Foseco_V38_R1811'
                    ]

select_Feeder_Type = 29

Feeder_Name_List = ['ASK_KIM_10_13','Chemex_CB_40_GK70_100','Foseco_Kalminex_ZF_9_12_KLU','Foseco_Kalminex_ZF_7_9','Hofmann_L130', #0-4
                    'Foseco_Feedex_VSK_36','ASK_ADS_61ExF27','Chemex_EK_50_80W','Hofmann_LK95','Foseco_V38'] #5-9

Feeder_Name_cooldown_criteria = [65,100,80,70,55,30,90,55,60,25]

Feeder_Type = Feeder_Type_List[select_Feeder_Type]

substring = "number_of_frames_above_threshold_after_max_array"

rootdir = f'F:/test11/{Feeder_Type}'

threshold_list = [0.95,0.90,0.85,0.80] #prozentual vom maximalwert als Abkühlkriterium
select_threshold = 3

i=0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if substring in file: # Lade nur Dateien, die histogram enthalten
            os.chdir(subdir)
            print(file)
            array_container = np.load(file,allow_pickle=True) #pickle kann eventuell gelöscht werden, weil das array nicht mehr als none initialisiert wird
            if i==0:
                raw_histogram_array = array_container
            else:
                raw_histogram_array = np.vstack()
            
print(raw_histogram_array.shape)           
chosen_histogram_data = raw_histogram_array[:,select_threshold]  
print(chosen_histogram_data.shape)  

print(max(chosen_histogram_data))
print(chosen_histogram_data[0])

plt.hist(chosen_histogram_data, bins = np.arange(0,max(chosen_histogram_data)))  # density=False would make counts
plt.ylabel('occurences')
plt.xlabel('Number of frames above threshold after maximum value')
plt.title(f'Threshold = {threshold_list[select_threshold]}')        
plt.show()        
            
                    
os.chdir("F:/test5")                   

            
  



