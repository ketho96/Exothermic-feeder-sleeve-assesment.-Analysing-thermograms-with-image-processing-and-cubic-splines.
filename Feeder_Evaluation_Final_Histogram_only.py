# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:14:52 2022

@author: Hiwi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 17:23:46 2022

Parent: Feeder_Evaluation_Final_v2.2.py
 
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
    
    
Change_Log:
   Keine Evaluation, nur Histogramme
    
    
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

Feeder_Type_List = ['ASK_KIM_10_13_Frisch','ASK_KIM_10_13_30_95_3G','ASK_KIM_10_13_R',
                    'Chemex_CB_40_GK70_100_R','Chemex_CB_40_GK70_100_30_95_7D','Chemex_CB_40_GK70_100_30_95_7G',
                    'Foseco_Kalminex_ZF_9_12_KLU_R','Foseco_Kalminex_ZF_9_12_KLU_Frisch','Foseco_Kalminex_ZF_9_12_KLU_30_95_7D',
                    'Foseco_Kalminex_ZF_7_9_Frisch','Foseco_Kalminex_ZF_7_9_30_95_7G','Foseco_Kalminex_ZF_7_9_R',
                    'Hofmann_L130_36','Hofmann_L130_37','Hofmann_L130_38','Hofmann_L130_39','Hofmann_L130_40',
                    'Foseco_Feedex_VSK_36_HD3','Foseco_Feedex_VSK_36_HDT','Foseco_Feedex_VSK_36_HD1',
                    'ASK_ADS_61ExF27_0','ASK_ADS_61ExF27_10','ASK_ADS_61ExF27_20','ASK_ADS_61ExF27_30',
                    'Chemex_EK_50_80W_CB22','Chemex_EK_50_80W_CB31','Chemex_EK_50_80W_CB35','Chemex_EK_50_80W_CB43', 
                    'Hofmann_LK95_R2909', 'Hofmann_LK95_R0601', 'Hofmann_LK95_R1102', 'Hofmann_LK95_R1301', 'Hofmann_LK95_R1310', 
                    'Hofmann_LK95_R1401', 'Hofmann_LK95_R1410', 'Hofmann_LK95_R1510', 'Hofmann_LK95_R1901','Chemex_Tele_80-17_R1201_C_oben',
                    'Chemex_Tele_80-17_R1201_C_unten', 'Chemex_Tele_80-17_R1201_E_oben', 'Chemex_Tele_80-17_R1201_E_unten', 'Chemex_Tele_80-17_R1201_G_oben',
                    'Chemex_Tele_80-17_R1201_G_unten', 'Chemex_Tele_80-17_R1201_X_oben', 'Chemex_Tele_80-17_R1201_X_unten', 'Chemex_Tele_80-17_R1201_Y_oben',
                    'Chemex_Tele_80-17_R1201_Y_unten', 'Chemex_Tele_80-17_R2311_A_oben', 'Chemex_Tele_80-17_R2311_A_unten', 'Chemex_Tele_80-17_R2311_B_oben',
                    'Chemex_Tele_80-17_R2311_B_oben', 'Foseco_V38_R0601_A', 'Foseco_V38_R0601_B', 'Foseco_V38_R1105', 'Foseco_V38_R1505', 'Foseco_V38_R1610',
                    'Foseco_V38_R1811']



dont_plot = True #zeige keine graphen von matplotlib an (nur speichern als png)
show_video = False #Zeigt die zwischenschritte der videoverarbeitung an
show_QA = True #Zeigt das RGB Image für die Quality Assurance an
 


resultdir = 'test5' #Unter diesm Pfad werden die Ergebnisse in den automatisch erstellten Ordnern abgelegt
rootdir = 'F:/02_Messreihen'
#rootdir = 'F:/02_Messreihen\Histogramme_erstellen_von' #Von diesem Verzeichnis ausgehend werden alle hierachisch darunterliegenden .wmv Dateien verarbeitet
#rootdir = 'F:/02_Messreihen/2021_11_09/Foseco_Kalminex_ZF_7_9_R (20.02.2020)'


curve_fit_spline = 100 #regler für den grad der glättung des splines, je größer desto gröber
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
            print('Feeder_Type:',Feeder_Type)
            try: # versuche die Textdatei zu lesen, die am Ende geschrieben wird, um die gleiche Datei nicht nochmal zu verarbeiten
                open(f'F:/{resultdir}/{Feeder_Type}/{file_save_name}Verarbeitet.txt')
                print('Resultat vorhanden')
                print('\n')
            except FileNotFoundError:
                begin = time.time()
                print('Resultat nicht vorhanden')
                
                #read mask
                Smask=cv2.imread(f'F:/01_HotAreaSegmentation/Masken/Masken_neu/Maske_{Feeder_Type}.png')
                print(f'F:/01_HotAreaSegmentation/Masken/Masken_neu/Maske_{Feeder_Type}.png')
                uniSmask=cv2.resize(Smask, (800,600))
                uniSmask = cv2.cvtColor(uniSmask, cv2.COLOR_BGR2GRAY)
                mask_size = np.count_nonzero(uniSmask) # gibt die fläche des speisers in der maske aus (anzahl pixel)
                print('mask size:',mask_size)
                
                
                
                
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
                            cv2.imshow('output', roiframe)
                            
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
                
                
                
                #Array mit der volatilität/standardabweichung erstellen   
                print('create mask_spline')
                #mask_max = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('ubyte')
                knots=np.array(range(curve_fit,pixel_vals_all_frames.shape[2]-curve_fit,curve_fit)) #hier werden die Positionen der Knoten definiert (evenly spaced mit der distanz von curve fit) changeLOG -curve_fit neu eingefügt
                d=0
                r=0
                mask_spline_max = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('float')
                mask_spline_integral = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('float')
                array_container = np.array((pixel_vals_all_frames.shape[2],1),dtype= float) # temporärer speicher für pixel vals all frames für einen pixel mit den koordinaten der iteration
                #spline_container = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('LSQUnivariateSpline')
                x0 = [pixel_vals_all_frames.shape[2]/2]
                bnds = [(0,pixel_vals_all_frames.shape[2])]
                #minimizer_kwargs = { "method": "Nelder-Mead","bounds":bnds }
                while d < pixel_vals_all_frames.shape[0]:
                    r=0
                    #if (d % 100) == 0:
                        #print(f'd:{d}/599')
                    while r < pixel_vals_all_frames.shape[1]:
                        #if max(pixel_vals_all_frames[d,r,:])>0:
                        if (uniSmask[d,r]<255):
                            #von ursprünglich 25 auf 1 gesetzt, wegen foseco speiser
                            r+=1
                        else:
                            #i=0
                            array_container = pixel_vals_all_frames[d,r,:]
                            spline = LSQUnivariateSpline(x_values_norm, array_container, knots)
                            mask_spline_max[d,r] = (spline(x_values10_norm)).max()
                            r+=1
                    d+=1

                print('saving mask_spline_max array')
               
                print('array was saved')
                
                #Histogramm erstellen => histogram_image_side_by_side.py
                
                if dont_plot:
                    # Turn interactive plotting off -> matplot show wird deaktiviert
                    plt.ioff()
                array_container = mask_spline_max.astype(int) #array kommt als float rein
                array_container = np.ndarray.flatten(array_container) #2D array in 1 D array weil histogram funktion nur ein 1D array verarbeiten kann
                #upper_limit = max(array_container)
                upper_limit = 255
                sns.set() #Stil des diagramms
                
                lower_limit = 5 #setzt die Untere Grenze für die Werte, die im Histogramm gezeigt werden, anfangs 25, für foseco auf  gesetzt
                step = 5 #setzt die Schrittweite der Bins des Histogramms
                
                bins = np.arange(lower_limit,upper_limit,step) #erstellt ein array mit bins vom lower limit bis zum upper limit mit schrittweite step
                plt.figure(figsize=(20,9.5)) #größe des figures in inch (oder so) (width,height)
                plt.subplot(2,3,(1,3))  # erstellt ein gitter innerhalb des figures mit 2 zeilen und 3 spalten -> 6 positionen. zähler beginnt oben links bei 1. die axis 1 (histogram) erstreckt sich von position 1 bis position 3 (1,3)
                plt.hist(array_container, bins = bins, align= 'left')  # 
                plt.title(f'Histogram of maximum smoothed gray values {file_save_name}')
                plt.style.use('ggplot')
                plt.xlabel('gray value [ \u2265 x ] ') #\u2265 ist unicode und wir als größer gleich zeichen ausgegeben
                plt.xticks(ticks=bins)
                #plt.xticklabels(tick_labels)
                #plt.xticks(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.
                plt.ylabel('occurrences [pixels]')
                plt.xlim((lower_limit-5), upper_limit-5)
                plt.ylim(0, mask_size*0.45)
                sns.reset_orig()
                
                #fig, ax = plt.subplot(2, 1, 2)
                plt.subplot(2,3,5)
                plt.title('Maximum smoothed gray values')
                plt.imshow(mask_spline_max, cmap='gray', vmin=lower_limit, vmax=upper_limit)
                plt.colorbar()
                plt.xticks([]) #keine achsenbeschriftung
                plt.yticks([])
                
                
                plt.tight_layout() # space between the plots
                #plt.savefig(f'{file_save_name}_histogram.png') #verschoben in den speicherbereich des codes
            
                
            
                           
                
                
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
                f.write(f"file name: {file} file direction {file_dir} File_save_name:{file_save_name} Feeder Type:{Feeder_Type}")
                f.close()
                
                #Speichere die QA Textdatei ab
                #QA_Textfile = open(f"{file_save_name}QA_Textfile.txt","w+")
                #QA_Textfile.write(f'{QA_Textfile_container}')
                #QA_Textfile.close()
                
                print('Speichern')
                
                #Histogramm
                plt.savefig(f'{file_save_name}_histogram.png')
                
                #plt.imsave(f'{file_save_name}_mask_max.png',mask_max, cmap='gray')
                plt.imsave(f'{file_save_name}_mask_spline_max.png',mask_spline_max, cmap='gray')
                
                
                #img = Image.fromarray(QA_Array)
                #if show_QA:    
                    #img.show()
                #img.save(f'{file_save_name}_QA_Array.png')
                
                #plt.imsave(f'{file_save_name}_mask_ERROR.png',mask_ERROR, cmap='gray')
                #plt.imsave(f'{file_save_name}_mask_SUCCESS.png',mask_SUCCESS, cmap='gray')
                
                
                
                #Reaktionskurven
                ###np.save(f'{file_save_name}_result_number_of_reacted_pixels',result_number_of_reacted_pixels) 
                
                #np.save(f'{file_save_name}_result_number_of_finished_pixels',result_number_of_finished_pixels) 
                
                #np.save(f'{file_save_name}_mask_ERROR',mask_ERROR) 
                
                #np.save(f'{file_save_name}_mask_SUCCESS',mask_SUCCESS)
                
               # np.save(f'{file_save_name}_QA_Array',QA_Array)
                
                
                
                  
                
                #
                
                #print('saving mask _max array')
                #np.save(f'{file_save_name}mask_max',mask_max) 
                #print('array was saved')
                
                np.save(f'{file_save_name}mask_spline_max',mask_spline_max) 
                
                plt.close('all') #schließe alle pyplot fenster (musste ich einfügen, da sonst ein runtime error ausgegeben wird, der darauf hinweist, dass zu viele fenster geöffnet wurden plt.close(RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).))
                
                end = time.time()
                # total time taken
                print(f"Total runtime videoprocessing: {end - begin}s")
                print('\n')
            
            
            
            #continue   
        
        
        
        
        
        
        
        
                            
                            
                            
                            
                            
                            
                        
   