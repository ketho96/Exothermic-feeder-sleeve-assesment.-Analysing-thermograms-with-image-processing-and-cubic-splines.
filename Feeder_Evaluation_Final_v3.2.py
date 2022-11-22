# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:14:52 2022

@author: Hiwi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 17:23:46 2022

Parent: Feeder_Evaluation_Final_v2.2.VIDEOERSTELLENv2(?)py
 
Features:
    einheitliche Nomenklatur der Dateinamen, die gespeichert werden. Es werden gespeichert: mask_max numpy und png , mask_spline_max numpy und png,histogramm png,#
    result_number_of_reacted_pixels etc. numpy 
    Neue Nomenklatur: Speisermessung XXX + "was davon". vernachlässigung von curve fit start frame etc, da fix und oder automatisch generiert. Bei Änderung der Parameter
    neuen Lauf mit neuem Ordner machen -> innerhalb des Ordners eine Textdatei mit Parametern schreiben
    mask_max spline berechnung wieder integriert
    Das Ende der exothermen reaktion ist auf 0,8 vom maximum gesetzt
    Schalter, der die Ausgabe der graphen unterbindet (dont_plot)
    Schalter show_video um sich bei bedarf die verarbeitung anzeigen zu lassen
    speichern von zwischenständen rausgenommen
    SKalierung des Histogramms fixiert, um vergleichbarkeit zu gewährleisten 
        x-achse 255 (maximaler grauwert der überhaupt erreicht werden kann)
        y-achse 0 und maximale anzahl an weißen pixeln der maske (eventuell ein dritter oder viertel der gesamtanzahl? -> *0.2)
    Feeder Type Liste aktualisiert mit Vereinheitlichung
    gib kein reaction only mehr aus
    SUCCESS array
    QualityAssurance[QA] = SUCCESS und ERROR Array als RGB Image ausgeben (aus 3 arrays into one rgb image.py) es gibt 3 success fälle und 4 error fälle (255,150,60 green) (80,150,255 red black)
    QA Textfile -> Muss noch unter dem richtigen Pfad abgelegt werden -> String Variable erstellen
    Schalter um QA Image anzuzeigen
    Bei der Verarbeitung eines Pixels, wo am Ende ein Überhang gebildet werden muss war ein Fehler in der Verarbeitung. In v.2.2 versuche ich diesen Fehler zu beheben. Vermutlich wird der Anhang nicht richtig gebildet
    Das Errorhandling der Auswertung wurde angepasst, es wird jetzt immer ein Überhang gebildet (verfälscht das das Ergebnis?) 
    Es wird ein Video der Verarbeitung abgespeichert (?)
    
    Teile der Speisermaske sind nicht 255 also genau weiß. Alles, was in der Maske nicht 0 ist, wird auf 255 gesetzt
    die Funktion .max() ist schneller als max(x)
    Histogramme sind raus
    ##### !! ######Neben dem Kriterium der maximalen Steigung, muss zum Zeitpunkt der Steigung nun auch mindestens 70% des Maximalwertes vorliegen, sonst wird die nächstkleinere Steigung genommen, bis beide Kriterien erfüllt werden
    QA Arrays werden nciht erstellt/gespeichert
    number_of_frames_above_threshold_after_max_array wird berechnet (ein Array mit einer spalte für jeden threshold wert in einer threshold list (um schnell verschiedene schwellwerte zu vergleichen) und einer zeile für jeden pixel)
    
    
Change_Log:
    Für jeden Speisertyp wird ein fixwert für die Ablühlung als Kriterium genommen. Pixel, die diesen Grauwert erst gar nicht erreichen werden ignoriert
    
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#import csv
#import pandas as pd
#import os.path as os
from scipy.interpolate import  LSQUnivariateSpline
#import scipy.optimize as spo
#from scipy.optimize import minimize_scalar
#from scipy.optimize import basinhopping
import time
from fuzzywuzzy import fuzz
import seaborn as sns
from PIL import Image



"""Parameter"""

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


Feeder_Name_List = ['ASK_KIM_10_13','Chemex_CB_40_GK70_100','Foseco_Kalminex_ZF_9_12_KLU','Foseco_Kalminex_ZF_7_9','Hofmann_L130', #0-4
                    'Foseco_Feedex_VSK_36','ASK_ADS_61ExF27','Chemex_EK_50_80W','Hofmann_LK95','Foseco_V38'] #5-9

Feeder_Name_cooldown_criteria = [65,100,80,70,55,30,90,55,60,25]








dont_plot = True #zeige keine graphen von matplotlib an (nur speichern als png)
show_video = False #Zeigt die zwischenschritte der videoverarbeitung an
show_QA = False #Zeigt das RGB Image für die Quality Assurance an
create_video = False # Erstelle ein Video der Detektion
show_progress = False #Schalter, wenn man sehen will, wie weit der Fortschritt über die Pixel ist


resultdir = 'test11' #Unter diesm Pfad werden die Ergebnisse in den automatisch erstellten Ordnern abgelegt
rootdir = r'F:\02_Messreihen'



curve_fit_spline = 100 #regler für den grad der glättung des splines, je größer desto gröber     ursprünglich 100!
curve_fit_vol = 20 #regler für das intervall der volatilität je kleiner desto sensibler

regionOfInterest = [0, 800, 0, 600] # [xmin, xmax, ymin, ymax] 
row =  regionOfInterest[3] - regionOfInterest[2]#dimensionen des bildes
column = regionOfInterest[1] - regionOfInterest[0] #dimensionen des bildes

QA_intensity = 0.5 #Regler für die Helligkeit der Farben -> Auf 0.5 gesetzt, um zu sehen, ob ein Pixel doppelt ausgewertet wird -> teilweise über 100% im Plot


"""##########################"""

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith('.wmv'):
            os.chdir(subdir)
            file_dir =  os.path.join(subdir, file)
            print("current dir:",os.getcwd())
            file_save_name = file[:-4] + '_' #das .wmv aus dem Dateinamen entfernen wegen der Übersichtlichkeit
            print ('file:',file)
            index_list = []
            for Type in Feeder_Type_List:
                index_list.append(fuzz.partial_ratio(Type, file))#berechne für jedes file einen Grad der Übereinstimmung mit den Speisertypen/Speisertypenliste
            index = np.argmax(index_list) #fine den höchsten Grad der Übereinstimmung/dessen Index in der Feeder_type_List
            Feeder_Type = Feeder_Type_List[index] #Setze den Speisertypen des Files fest, um die entsprechende Maske zu laden
            
            
            index_list = []
            for Name in Feeder_Name_List:
                index_list.append(fuzz.partial_ratio(Name, file))#berechne für jedes file einen Grad der Übereinstimmung mit den Speisertypen/Speisertypenliste
            index = np.argmax(index_list) #fine den höchsten Grad der Übereinstimmung/dessen Index in der Feeder_type_List
            Feeder_Name = Feeder_Name_List[index] #Setze den Speisertypen des Files fest, um die entsprechende Maske zu laden
            Feeder_Cooldown_Criteria = Feeder_Name_cooldown_criteria[index]
            
            
            
            
            
            print('Feeder_Type:',Feeder_Type)
            print('Feeder_Name:', Feeder_Name)
            print('Feeder Cooldown Criteria:', Feeder_Cooldown_Criteria)
            try: # versuche die Textdatei zu lesen, die am Ende geschrieben wird, um die gleiche Datei nicht nochmal zu verarbeiten
                open(f'F:/{resultdir}/{Feeder_Type}/{file_save_name}Verarbeitet.txt')
                print('Resultat vorhanden')
                
            except FileNotFoundError:
                
                
                
                
                begin = time.time()
                print('Resultat nicht vorhanden')
                QA_Textfile_container = ''
                #read mask
                Smask=cv2.imread(f'F:/01_HotAreaSegmentation/Masken/Masken_neu/Maske_{Feeder_Type}.png')
                print(f'F:/01_HotAreaSegmentation/Masken/Masken_neu/Maske_{Feeder_Type}.png')
                uniSmask=cv2.resize(Smask, (800,600))
                uniSmask = cv2.cvtColor(uniSmask, cv2.COLOR_BGR2GRAY)
                uniSmask[uniSmask!=0]=255
                mask_size = np.count_nonzero(uniSmask) # gibt die fläche des speisers in der maske aus (anzahl pixel)
                print('mask size:',mask_size)
                
                
                #Histogramm erstellen => histogram_image_side_by_side.py
            
                fn=0
                
                cap = cv2.VideoCapture(file)
                
                #fn = 0
                
                
                """#############################################################################################"""
                """containerainers INIT"""
                """#############################################################################################"""
                
                list_all_frames = []
                
                """#############################################################################################"""
                """#############################################################################################"""
                """#############################################################################################"""
                """#############################################################################################"""
                """#############################################################################################"""
                
                
                while(cap.isOpened()):
                    
                    # nächtes Videobild laden
                    ret, frame = cap.read()
                
                    # Wenn das Video nicht zuende ist = (ret == true) (falls ret == false konnte kein (weiteres) Bild geladen werden)
                    if ret:
                        
                        #print(fn)
                        #if fn<startFrame or fn>endFrame:
                            #fn += 1
                            #continue
                        
                        uniframe=cv2.resize(frame, (800,600))
                        # Videobild in Grauwerte umwandeln
                        
                        roigray = cv2.cvtColor(uniframe, cv2.COLOR_BGR2GRAY)
                        
                        
                        #creating background image mask
                        roiframe=cv2.bitwise_and(roigray, uniSmask)
                        
                        
        
                        if show_video:
                            
                            
                            
                            # das aktuell eingelesene Videobild wird für das OutputVideo kopiert
                            output = roiframe.copy()
                            
                            # Zeige Output-Videobild an
                            cv2.imshow('Process', output)
                            
                            #shows the image read from the video
                            cv2.imshow('input frame',frame)
                    
                            #shows the mask that was read from the Data folder
                            cv2.imshow('input mask',uniSmask)
                            
                            #shows the result of the masking process = input frame for the areaSegmentation algorithm
                            #cv2.imshow('output', roiframe)
                            
                            cv2.waitKey(1)
                        
                        list_all_frames.append(roiframe)
                        
                        fn += 1
                
                    # Wenn das ret False (das Video ist zuende), beende die Schleife und löse die Videos aus dem Speicher
                    else:
                        #print('cap release')
                        cap.release()
                        #out.release()
                        cv2.destroyAllWindows()
                        
                pixel_vals_all_frames = np.array(list_all_frames)
                pixel_vals_all_frames = pixel_vals_all_frames.transpose(1,2,0)
                print('pixel_vals_all_frames.shape',pixel_vals_all_frames.shape)
                    
                    
                """#######################################################"""
                
                """#################ENDE DER VIDEOVERARBEITUNG###############"""
                
                """##########################################################"""
                
                
                
                
                len_pixel_vals_all_frames = pixel_vals_all_frames.shape[2]#-1
                
                x_values_norm    = np.linspace(0, len_pixel_vals_all_frames,len_pixel_vals_all_frames) #einfache Dichte der Werte/ Auflösung des Splines
                x_values10_norm  = np.linspace(0, len_pixel_vals_all_frames,len_pixel_vals_all_frames*10) #10fache Dichte
                x_values100_norm = np.linspace(0, len_pixel_vals_all_frames,len_pixel_vals_all_frames*100) #100 fache Dichte
                
                #Erstellung der mask max
                
                    
                curve_fit=curve_fit_spline
                
                
                
                
            
            
                           
                print('Evaluation')
                
                
                
                d=0
                r=0
                
                knots_norm=np.array(range((len_pixel_vals_all_frames-curve_fit),curve_fit,-curve_fit)) #erzeugt die knoten vom letzten punkt aus bis nach vorne
                knots_norm = knots_norm[::-1] #kehrt die reihenfolge der knoten um, da range nur so herum arbeiten kann (knoten von hinten erstellen)
                
                #mask_ERROR = np.zeros((row, column)).astype('float')
                #mask_SUCCESS = np.zeros((row, column)).astype('float')
                
                QA_Array_react = np.zeros((row,column,3), 'uint8')
                QA_Array_finish = np.zeros((row,column,3), 'uint8')
                #Erzeuge eine Textdatei in der für jeden fehlerhaft ausgewerteten Pixel der Grund steht          
                QA_Textfile_container = ''
                
                Point_of_Ignition_Array = np.zeros((row,column), 'uint16')
                
                
                
                
                
                
                
                
                array_container = np.array((pixel_vals_all_frames.shape[2],1),dtype= float) # temporärer speicher für pixel vals all frames für einen pixel mit den koordinaten der iteration
                result_number_of_reacted_pixels = np.zeros(pixel_vals_all_frames.shape[2])
                result_number_of_finished_pixels = np.zeros(pixel_vals_all_frames.shape[2])
                result_sum_of_reacted_pixels = np.zeros(pixel_vals_all_frames.shape[2])
                result_sum_of_finished_pixels = np.zeros(pixel_vals_all_frames.shape[2])
                result_sum_of_total = np.zeros(pixel_vals_all_frames.shape[2])
                number_of_frames_above_threshold_after_max_array = [0]*1 #initializing an arry with 4 time None (one for each iteration of threshold_list)
                
                while d < pixel_vals_all_frames.shape[0]:
                    r=0
                    if show_progress:
                        if (d % 100) == 0:
                            print(f'd:{d}/599')
                    while r < pixel_vals_all_frames.shape[1]:
                        #if max(pixel_vals_all_frames[d,r,:])>0:
                        if uniSmask[d,r]!=255:
                        #von ursprünglich 25 auf 1 gesetzt, wegen foseco speiser
                            r+=1
                        else:
                            """ Versuch Bugfix 18.06.2022"""
                            
                            #print(f'd:{d}/r:{r}')
                            
                            #versuche einen Überhang zu bilden, um Ausschläge des Splines am rechten Ende zu unterbinden
                            
                            values = pixel_vals_all_frames[d,r,:]
                            
                            len_values_init=values.shape[0]-1
                            
                            appendix = values[-1] #der letzte Wert der Messkurve wird konstant angehangen
                            appendix_list=[]
                            i=0
                            while i< curve_fit*5: 
                                appendix_list.append(appendix)
                                i+=1
                            values=np.append(values, appendix_list)
                            
                            
                            len_values=values.shape[0]-1
                            
                            
                            
                            knots=np.array(range((len_values-curve_fit),curve_fit,-curve_fit)) #-curve fit erzeugt die Orte der Knoten vom letzten Messwert aus nach vorne
                            knots = knots[::-1] #kehrt die Reihenfolge der Knoten um
                            
                            
                            
                            
                            x_values =np.linspace(0, len_values,len_values) #einfache Dichte der Werte/ Auflösung des Splines
                            #x_values10 = np.linspace(0, (len_values,len_values)*10) #10fache Dichte
                            #x_values100 =   np.linspace(0, (len_values,len_values)*100) #100 fache Dichte
                            
                            
                            spline = LSQUnivariateSpline(x_values, values[0:len_values], knots)
                            spline_deriv = spline.derivative()
                            
                            
                            y_spline = spline(x_values)
                            y_spline_deriv = spline_deriv(x_values)
                            
                            
                            #Punkt finden, an dem die Steigung des Splines maximal ist = Beginn der Exothermen Reaktion
                            
                            #frame_max_deriv = round((int((np.where(y_spline_deriv == y_spline_deriv.max()))[0])))
                            
                            ###### Aus MesskurveAusMehrerenValuesKurvenV4 04.07.2022 ######
                            
                            
                            count = -1 
                            maximum_value = y_spline.max()
                            sorted_deriv_values = np.sort(y_spline_deriv) #sortiere aufsteigend
                            Found = False # solange nicht bei der größten Steigung auch 70% des Maximalwertes errreicht wurde, suche die nächstkleinere Steigung
                            while not(Found):
                                deriv_value = sorted_deriv_values[count] # nimm die letzte Steigung = größte Steigung deriv_value[-1]
                                frame_max_deriv = round((int((np.where(y_spline_deriv == deriv_value))[0])))
                                if not(values[frame_max_deriv] >= 0.7*(maximum_value)): #überprüfe, ob auch 70% des Maximalwertes erreicht wurden
                                    count-=1 #ansonsten nimm die näcshtkleinere Steigung
                                else:
                                    Found=True
                                    
                            
                            
                            
                            
                            
                            
                            
                            try:
                                result_number_of_reacted_pixels[frame_max_deriv]+=1
                                
                                Point_of_Ignition_Array[d,r] = frame_max_deriv
                                
                                
                                
                                QA_Array_react[d,r,1] = 255*QA_intensity #GREEN
                            except:
                                QA_Array_react[d,r,2] = 255*QA_intensity #RED
                                QA_Textfile_container += (f'index is out of bounds. Reaction not found @ d:{d}/r:{r} RED\n')
                            
                            
                            
                            #Punkt nach dem Beginn der exothermen Reaktion finden, an dem der Wert 80% des Maximums erreicht = Ende der Exothermen Reaktion
                            #Punkt finden, an dem der Grauwert unter einen für die Messreihe des Speisertyps typischen Wert abfällt
                            frame_max = round((int(((np.where(y_spline == maximum_value)[0])[0]))))  

                            
                              
                            
                            
                            x_values_after_max_deriv = np.linspace(frame_max_deriv,len_values,(len_values-frame_max_deriv)) #schneide die Funktionswerte so zu, dass nur Werte nach frame_max_deriv ausgewertet werden
                            y_spline_after_max_deriv = spline(x_values_after_max_deriv)
                            
                            x_values_after_max = np.linspace(frame_max,len_values,(len_values-frame_max)) #schneide die Funktionswerte so zu, dass nur Werte nach frame_max_deriv ausgewertet werden
                            y_spline_after_max = spline(x_values_after_max)
                            
                            
                            ####------------------Speichere, wie lange (wieviel Frames) dieser Pixel über dem Threshold lag -------------------------#########
                            
                            #threshold_list = [0.95*maximum_value,0.90*maximum_value,0.85*maximum_value,0.80*maximum_value] #optional kann eine Reihe an verschiedener Thresholds gegeben werden
                            threshold_list = [Feeder_Cooldown_Criteria] #liste an Thresholds, für die die Dauer berechnet werden soll
                            #threshold = 100 # = 1145°C = 154

                            
                            
                            number_of_frames_above_threshold_after_max_list = []
                            for threshold in threshold_list:
                                number_of_frames_above_threshold_after_max = len(y_spline_after_max[y_spline_after_max >= threshold])
                                number_of_frames_above_threshold_after_max_list.append(number_of_frames_above_threshold_after_max) #in einer Zeile stehen für jeden threshold für diesen Pixel eine Zahl der Frames, die dieser Pixel über dem threshold bleibt. In der nächsten Zeile steht der nächste Pixel
                            number_of_frames_above_threshold_after_max_array = np.vstack((number_of_frames_above_threshold_after_max_array,number_of_frames_above_threshold_after_max_list)) #ein array mit einer spalte für jeden threshold wert in der threshold list und für jeden pixel eine zeile
                           
                            #######---------------------------------------------------------------------------------------------------------------------########
                            
                            
                            
                            
                            
                            
                            
                            
                            try:
                                #frame_80pOf_max_after_max = round(int(((np.where(y_spline_after_max < 0.8*max(y_spline_after_max)))[0])[0]))
                                frame_80pOf_max_after_max = round(int(((np.where(y_spline_after_max < threshold))[0])[0]))
                                result_number_of_finished_pixels[frame_80pOf_max_after_max+frame_max]-=1 #erniedrige den zähler der abgeklungenen pixel zum zeitpunkt/frame, der gefunden wurde
                                
                                QA_Array_finish[d,r,1] = 255*QA_intensity #green 255 -> success
                            
                            except IndexError: #IndexError tritt auf, wenn 80% des Maximalwertes nicht innerhalb der Laufzeit erreicht werden
                                result_number_of_finished_pixels[len_values_init-1]-=1 #liegt das Maximum genau am Ende der Messkurve nimm an, dass die Reaktion dort beendet ist
                                
                                QA_Textfile_container += (f'Indexerror End of reaction not found, assumed at the end @ d:{d}/r:{r} RED\n')
                                QA_Array_finish[d,r,2] = 255*QA_intensity#RED
                                if show_progress:
                                    print(f'Indexerror End of reaction not found, assumed at the end @ d:{d}/r:{r} RED\n')
                            except Exception as e: #Wenn ein anderer Error auftritt, stoppe das Programm und gib den Fehler aus (sinnvoll?)
                                print(e)
                                QA_Array_finish[d,r,0] = 255*QA_intensity#RED
                                QA_Textfile_container += (f'COMPLETE FAILURE @ d:{d}/r:{r} BLUE\n')
                                if show_progress:
                                    print(f'COMPLETE FAILURE @ d:{d}/r:{r}')
                                pass
                                
                                """Bugfix Ende"""    
                            
                            
                            
                            
                        r+=1
                    d+=1 
                
                print('number_of_frames_above_threshold_after_max_array.shape',number_of_frames_above_threshold_after_max_array[0,:])
                number_of_frames_above_threshold_after_max_array = np.delete(number_of_frames_above_threshold_after_max_array,0,0)
                print('number_of_frames_above_threshold_after_max_array.shape',number_of_frames_above_threshold_after_max_array[0,:])
                
                print('saving _Point_of_Ignition_Array ')
                np.save(f'F:/{resultdir}/{file_save_name}_Glättung{curve_fit_spline}Point_of_Ignition_Array',Point_of_Ignition_Array)
                
                
                if create_video:
                    cap = cv2.VideoCapture(file)
                    frameSize = (800,600)
                    out = cv2.VideoWriter(f'F:/{resultdir}/{file_save_name}Glättung{curve_fit_spline}_output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize) #Speichere ein Video der Verarbeitung
                    
                    Area_Frame  = np.zeros((row,column,3), 'uint8')
                    
                    fn=0
                    
                    
                    
                    while(cap.isOpened()):
                        #print('create ignition video')
                        # nächtes Videobild laden
                        ret, frame = cap.read()
                    
                        # Wenn das Video nicht zuende ist = (ret == true) (falls ret == false konnte kein (weiteres) Bild geladen werden)
                        if ret:
                            
                            #print(fn)
                            
                            uniframe=cv2.resize(frame, (800,600))
                            # Videobild in Grauwerte umwandeln
                            
                            roigray = cv2.cvtColor(uniframe, cv2.COLOR_BGR2GRAY)
                            
                            
                            #creating background image mask
                            roigray=cv2.bitwise_and(roigray, uniSmask)
                            
                            
                            
                            
                            
                            Ignition_Array  = np.zeros((row,column), 'uint8')
                            
                            
                            Ignition_Array[Point_of_Ignition_Array==fn] = 255 #Das Ignition_Array wird an den entsprechenden Pixeln rot (?)
                            
                            #print(np.where(Ignition_Array!=0))
                            
                            Ignition_Array = cv2.cvtColor(Ignition_Array, cv2.COLOR_GRAY2BGR) #RGBA = RGB and Alpha channel (for opacity)
                            roiframe_BGR       = cv2.cvtColor(roigray, cv2.COLOR_GRAY2BGR) #RGBA = RGB and Alpha channel (for opacity)
                            
                            
                            Ignition_Array[:,:,0] = 0
                            Ignition_Array[:,:,1] = 0
                            
                            #Ignition_Array=cv2.bitwise_and(Ignition_Array, uniSmask)
                            #print('type, shape Ignition Array',type(Ignition_Array), Ignition_Array.shape)
                            #print('type, shape  roiframe_BGR',type(roiframe_BGR), roiframe_BGR.shape)
                            
                            Area_Frame = Area_Frame+Ignition_Array
                            Area_Frame[uniSmask==0]=0
                            
                            
                            output_frame = cv2.addWeighted(Ignition_Array,10000,roiframe_BGR,1,0)  
                            
                            output_frame = cv2.addWeighted(output_frame,1,Area_Frame,0.1,0)                            
                            
                            if show_video:
                                
                                
                                
                                # das aktuell eingelesene Videobild wird für das OutputVideo kopiert
                                output = output_frame.copy()
                                
                                # Zeige Output-Videobild an
                                cv2.imshow('Process', output)
                                
                                #shows the image read from the video
                                cv2.imshow('input frame',frame)
                                
                                #shows the image read from the video
                                cv2.imshow('Ignition_Array frame',Ignition_Array)
                                
                                cv2.waitKey(1)
                            
                            
                            fn += 1
                            out.write(output_frame) #schreibe in das video
                        # Wenn das ret False (das Video ist zuende), beende die Schleife und löse die Videos aus dem Speicher
                        else:
                            #print('cap release')
                            cap.release()
                            #out.release()
                            cv2.destroyAllWindows()
                
                
                
                
                
                
                
                
                
                #erstelle einen Neuen Ordner für den entsprechenden Speisertypen oder ändere das Arbeitsverzeichnis zum entsprechenden Ordner
                #FeederType = file[:-7] #entferne vom Videodateinamen alles bis auf den Speisertypen (XYZ1.1.wmv -> XYZ) haha ja.... wäre schön
                try:
                    os.mkdir(f'F:/{resultdir}/{Feeder_Type}')
                    os.chdir(f'F:/{resultdir}/{Feeder_Type}')
                except:
                    os.chdir(f'F:/{resultdir}/{Feeder_Type}')
                
                
                
                
                
                
                
                #Überprüfe auf Duplikate und wenn es diese Datei schon gibt, füge eine neue Version davon hinzu
                n=2
                while os.path.exists(f'{file_save_name}Verarbeitet.txt'):
                    file_save_name = file_save_name + f'v{n}' 
                    n+=1


                
                #Speichere alle Ergebnisse ab
                
                #Schreibe eine Textdatei an den Ort des Resultate Speisertyp um kenntlich zu machen, dass dieses Video schon bearbeitet wurde          
                f = open(f"{file_save_name}Verarbeitet.txt","w+")
                f.write(f"file name: {file} file direction {file_dir} File_save_name:{file_save_name} Feeder Type:{Feeder_Type} Feeder_Name: {Feeder_Name} Cooldown_criteria: {threshold}") 
                f.close()
                print('Speichern')
                
                #Speichere die QA Textdatei ab
                QA_Textfile = open(f"{file_save_name}QA_Textfile.txt","w+")
                QA_Textfile.write(f'{QA_Textfile_container}')
                QA_Textfile.close()
                
               
                
                #Histogramm
                plt.savefig(f'{file_save_name}_histogram.png')
                
                #plt.imsave(f'{file_save_name}_mask_max.png',mask_max, cmap='gray')
                #plt.imsave(f'{file_save_name}_mask_spline_max.png',mask_spline_max, cmap='gray')
            
            
                img = Image.fromarray(QA_Array_react)
                if show_QA:    
                    img.show()
                img.save(f'{file_save_name}_QA_Array_react.png')
                
                img = Image.fromarray(QA_Array_finish)
                if show_QA:    
                    img.show()
                img.save(f'{file_save_name}_QA_Array_finish.png')
                
                #plt.imsave(f'{file_save_name}_mask_ERROR.png',mask_ERROR, cmap='gray')
                #plt.imsave(f'{file_save_name}_mask_SUCCESS.png',mask_SUCCESS, cmap='gray')
                
                
                
                
                #Speichere das Video ab
                if create_video:
                    out.release()
                
                
                    
                #Reaktionskurven
                np.save(f'{file_save_name}_result_number_of_reacted_pixels',result_number_of_reacted_pixels) 
                
                np.save(f'{file_save_name}_result_number_of_finished_pixels',result_number_of_finished_pixels) 
                
                #np.save(f'{file_save_name}_mask_ERROR',mask_ERROR) 
                
                #np.save(f'{file_save_name}_mask_SUCCESS',mask_SUCCESS)
                
                np.save(f'{file_save_name}_QA_Array_react',QA_Array_react)
                np.save(f'{file_save_name}_QA_Array_finish',QA_Array_finish)
            
            
            
            
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
                
                #print('saving result_sum_of_reacted_pixels')
                np.save(f'{file_save_name}_result_sum_of_reacted_pixels',result_sum_of_reacted_pixels) 
                #print('array was saved')
                
                #print('saving result_sum_of_finished_pixels')
                np.save(f'{file_save_name}_result_result_sum_of_finished_pixels',result_sum_of_finished_pixels) 
                #print('array was saved')
                
                #print('saving result_sum_of_total')
                np.save(f'{file_save_name}_result_sum_of_total',result_sum_of_total) 
                #print('array was saved')
                
                print('saving number_of_frames_above_threshold_after_max_array')
                np.save(f'{file_save_name}_number_of_frames_above_threshold_after_max_array',number_of_frames_above_threshold_after_max_array) 
                print('array was saved')
                
                
                #print('saving mask _max array')
                #np.save(f'{file_save_name}mask_max',mask_max) 
                #print('array was saved')
                
                #np.save(f'{file_save_name}mask_spline_max',mask_spline_max) 
                
                plt.close('all') #schließe alle pyplot fenster (musste ich einfügen, da sonst ein runtime error ausgegeben wird, der darauf hinweist, dass zu viele fenster geöffnet wurden plt.close(RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).))
            
                
                end = time.time()
                # total time taken
                print(f"Total runtime videoprocessing: {end - begin}s")
                print('\n')
            
            
            
            #continue   
        
        
        
        
        
        
        
        
                            
                            
                            
                            
                            
                            
                        
   