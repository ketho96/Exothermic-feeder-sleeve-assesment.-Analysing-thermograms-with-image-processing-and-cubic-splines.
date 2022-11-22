# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 15:00:54 2022

Parent: ErzeugeGraphen_v2.7.py (Läuft)
 
Features:
    Alle Kurven starten zur gleichen zeit (nur die sum_of_reacted wird genutzt um alle kruven zu trimmen)
    es wird erst ab 10 pixeln geplottet
    Die werte auf der ordinate werden in prozent normiert 
    thresh_pixels ist eine Einstellung, um den Mindestwert an detektierten Pixeln zu definieren (manchmal werden zu Beginn schon einzelne Pixel fehlerhaft ausgewertet)
    Ich versuche für jede Messung nur genau einen Startpunkt zu definieren in der Form eines Dictionaries {'messung':startframe} -> success
    Liniendicke in der Legend angepasst
    legendenbezeichnung geänder case sesitive
    Abkühlkriterium in der Überschrift
    Alle Kurven sind jetzt synchronisiert (auch inder v2.5 übernommen)
    
    
Change_Log:
    Legende in eigenem Fenster anzeigen (funkioniert noch nicht)
    alle Kruven einer Messreihe zusammenfassen um innerhalb des Speisertyps zu vergleichen
    2.8 vereinheitlichung für die BA
    
    
    
    

@author: Hiwi
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import compress
import cv2
from fuzzywuzzy import fuzz

Feeder_Type_List = ['ASK','Feedex_Hoch','GK70','ZF9_12','KIM_10_13','Kalminex_7_9','L130','ASK_ADS_61','Feedex_Klein','KIM_30_95','Speisermessung_Kalminex_',
                    'ASK_KIM_10_13_Frisch','ASK_KIM_10_13_30_95_3G','ASK_KIM_10_13_R', #11-13
                    'Chemex_CB_40_GK70_100_R','Chemex_CB_40_GK70_100_30_95_7D','Chemex_CB_40_GK70_100_30_95_7G', #14-16
                    'Foseco_Kalminex_ZF_9_12_KLU_R','Foseco_Kalminex_ZF_9_12_KLU_Frisch','Foseco_Kalminex_ZF_9_12_KLU_30_95_7D', #17-19
                    'Foseco_Kalminex_ZF_7_9_Frisch','Foseco_Kalminex_ZF_7_9_30_95_7G','Foseco_Kalminex_ZF_7_9_R', #20-22
                    'Hofmann_L130_36','Hofmann_L130_37','Hofmann_L130_38','Hofmann_L130_39','Hofmann_L130_40', #23-27
                    'Foseco_Feedex_VSK_36_HD3','Foseco_Feedex_VSK_36_HDT','Foseco_Feedex_VSK_36_HD1', #28-30
                    'ASK_ADS_61ExF27_0','ASK_ADS_61ExF27_10','ASK_ADS_61ExF27_20','ASK_ADS_61ExF27_30', #31-34
                    'Chemex_EK_50_80W_CB22','Chemex_EK_50_80W_CB31','Chemex_EK_50_80W_CB35','Chemex_EK_50_80W_CB43', #35-38
                    'Hofmann_LK95_R2909', 'Hofmann_LK95_R0601', 'Hofmann_LK95_R1102', 'Hofmann_LK95_R1301', 'Hofmann_LK95_R1310', #39-43
                    'Hofmann_LK95_R1401', 'Hofmann_LK95_R1410', 'Hofmann_LK95_R1510', 'Hofmann_LK95_R1901','Chemex_Tele_80-17_R1201_C_oben', #44-48
                    'Chemex_Tele_80-17_R1201_C_unten', 'Chemex_Tele_80-17_R1201_E_oben', 'Chemex_Tele_80-17_R1201_E_unten', 'Chemex_Tele_80-17_R1201_G_oben', #49-52
                    'Chemex_Tele_80-17_R1201_G_unten', 'Chemex_Tele_80-17_R1201_X_oben', 'Chemex_Tele_80-17_R1201_X_unten', 'Chemex_Tele_80-17_R1201_Y_oben', #53-56
                    'Chemex_Tele_80-17_R1201_Y_unten', 'Chemex_Tele_80-17_R2311_A_oben', 'Chemex_Tele_80-17_R2311_A_unten', 'Chemex_Tele_80-17_R2311_B_oben', #57-60
                    'Chemex_Tele_80-17_R2311_B_unten', 'Foseco_V38_R0601_A', 'Foseco_V38_R0601_B', 'Foseco_V38_R1105', 'Foseco_V38_R1505', 'Foseco_V38_R1610', #61-66
                    'Foseco_V38_R1811' ]#67

Result_List = ['sum_of_reacted_pixels', 'sum_of_finished_pixels', 'sum_of_total']

Feeder_Name_List = ['ASK_KIM_10_13','Chemex_CB_40_GK70_100','Foseco_Kalminex_ZF_9_12_KLU','Foseco_Kalminex_ZF_7_9','Hofmann_L130', #0-4
                    'Foseco_Feedex_VSK_36','ASK_ADS_61ExF27','Chemex_EK_50_80W','Hofmann_LK95','Foseco_V38'] #5-9

Feeder_Name_cooldown_criteria = [65,100,80,70,55,30,90,55,60,25]

#Choose the results which should be displayed

select_Feeder_Type = 11 # 999 = Keine explizite Auswahl, sondern alle Messungen eines Typs
select_ALL_Feeder_Names = False # Schalter, ob alle Namen bearbeitet werden sollen oder nur ein bestimmter Typ
select_Feeder_Name = 0
sum_of_reacted_pixels = True #reagierte pixel
sum_of_finished_pixels = True #kurve die runter geht
sum_of_total = True #summe aus reagiert und beendet
show_all_lines = True #schalter, ob direkt an alle messkurven geplottet werden sollen

#thresh_pixels = 0 # Anzahl der Pixel die den Nullpunkt/Start des Plots markieren ACHTUNG hat Einfluss auf die Bestimmung des point_of_ignitions (point_of_ignition ist der Zeitpunkt, wenn thresh_pixel erreicht werden)
thresh_percentage = 0 # % Füllung der Maske ab der gezählt wird

ylim_upper = 105
ylim_lower = -105
loc='upper left' # Ausrichtung der Legende 'upper left', 'upper right', 'lower left', 'lower right'




"""EINSTELLUNGEN ENDE################################################################################################################################"""





Result_Select_List = [sum_of_reacted_pixels, sum_of_finished_pixels, sum_of_total] #Boolsche Werte
Result_Select_List = list(compress(Result_List, Result_Select_List)) #filter die result list mit der result select list









Feeder_Type = Feeder_Type_List[select_Feeder_Type]

rootdir = f'F:/test11/{Feeder_Type}' 
#rootdir = f'C:/Users/Steffen/Desktop/Plot Speiser/Finale Kurvenauswertung/test10/{Feeder_Type}' 
#rootdir = r'F:\Speiserauswahl Daniel\Hofmann_L130'






for Feeder_Name_Selected in Feeder_Name_List:



    if not select_ALL_Feeder_Names:
        Feeder_Name_Selected = Feeder_Name_List[select_Feeder_Name]
    Feeder_Type_List_Selection = [] #Suche alle Messreihen eines Speisertyps
    for Feeder_Type in Feeder_Type_List:
        if Feeder_Name_Selected in Feeder_Type:
            Feeder_Type_List_Selection.append(Feeder_Type) # Liste mit allen Messreihen eines Speiserstyps
            
    print('Feeder Type lIst Selection',Feeder_Type_List_Selection)

################---------------------Baustelle: Mittelwert eines Speisertyps ermitteln/ darstellen--------------------########################
for Feeder_Type in Feeder_Type_List_Selection: 
    reacted_list = []
    finished_list = []
    total_list = []
################---------------------Baustelle--------------------########################



#F:\test4\L130_36
plot_title = ''

lines = []
fig, ax = plt.subplots()
i=0
max_number_of_frames = 0
max_number_of_pixels = 0
max_point_of_ignition = 0
ignition_dict = {} #Dictionary mit key:value / messreihe:point_of_ignition

for result_selected in Result_Select_List:
    plot_title = plot_title + f'{result_selected}' + ' '
    substring = f'{result_selected}' 
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if substring in file: # Lade nur Dateien, die das entsprechend ausgewählte resultat enthalten
                os.chdir(subdir)
                result_container = np.load(file) #lade das entsprechende resultat der evaluation (sum of X /summe der pixel zu Zeitpunkt/Frame)
                #print(result_container[-1])
                print('result container shape',result_container.shape)
                print('Filename:',file)
                len_result_container = len(result_container)
                print('len_result_container:',len_result_container)
                
                if len_result_container > max_number_of_frames:
                    max_number_of_frames = len_result_container
                    
                if result_container[-1] > max_number_of_pixels:
                    max_number_of_pixels = result_container[-1]
                
                
                 #read mask
                Smask = cv2.imread(f'F:/01_HotAreaSegmentation/Masken/Masken_neu/Maske_{Feeder_Type}.png')
                #print(f'F:/01_HotAreaSegmentation/Masken/Masken_neu/Maske_{Feeder_Type}.png')
                Smask = cv2.cvtColor(Smask, cv2.COLOR_BGR2GRAY)
                uniSmask=cv2.resize(Smask, (800,600))
                mask_size = np.count_nonzero(uniSmask) # gibt die fläche des speisers in der maske aus (anzahl pixel)
                print('mask size:',mask_size)
                
                if 'sum_of_reacted_pixels' in file:
                    result_container[result_container<=((thresh_percentage/100)*mask_size)]=0 # alle Werte, die kleiner als der eingestellte Prozentasatz der Maske sind, werden rausgefiltert
                    #result_container[result_container<=thresh_pixels]=0 # alle werte kleiner als thresh_pixels werden rausgefiltert (damit ein zu beginn falsch ausgewerteter pixel nicht den point_of_ignition verfälscht)
                    point_of_ignition = result_container.size - np.count_nonzero(result_container) #bestimme den point_of_ignition
                    key = file.replace('__result_sum_of_reacted_pixels.npy','') #Filtere vom Dateinamen nur den Feeder_Type heraus und lösche die Benennung des Resultats
                    ignition_dict[f'{key}'] = point_of_ignition #weise jedem key/messreihe einen value/point_of_ignition zu

                if '__result_result_sum_of_finished_pixels.npy' in file:
                    key = file.replace('__result_result_sum_of_finished_pixels.npy','') #Filtere vom Dateinamen nur den Feeder_Type heraus und lösche die Benennung des Resultats
                    
               
                if '__result_sum_of_total.npy' in file:
                    key = file.replace('__result_sum_of_total.npy','') #Filtere vom Dateinamen nur den Feeder_Type heraus und lösche die Benennung des Resultats
                    
                try:    
                    point_of_ignition = ignition_dict[key]
                    result_container = result_container[point_of_ignition:]
                    print('len result container after cutting:',len(result_container))
                    print('point_of_ignition:',point_of_ignition)
                    #print('try')
                except KeyError:
                    point_of_ignition = 0
                    #print('except')
                    
                if point_of_ignition > max_point_of_ignition:
                    max_point_of_ignition = point_of_ignition
                
                print('Feeder_Type:',Feeder_Type)
                
                
                #index_list = []
                #for Type in Feeder_Type_List:
                #    index_list.append(fuzz.partial_ratio(Type, file))#berechne für jedes file einen Grad der Übereinstimmung mit den Speisertypen/Speisertypenliste
                #index = np.argmax(index_list) #fine den höchsten Grad der Übereinstimmung/dessen Index in der Feeder_type_List
                #Feeder_Type = Feeder_Type_List[index] #Setze den Speisertypen des Files fest, um die entsprechende Maske zu laden
                
                
               
                
                
                
                
                
                result_container = (np.divide(result_container,mask_size))*100# teile jeden wert durch die mask_size und multipliziere mit 100, um den verlauf in % zu bekommen
                
                file_name = f'{file}'
                
                if 'sum_of_reacted_pixels' in file:                 
                    file = file.replace(f'{Feeder_Type}', 'Aufheizkurve Probe') 
                    reacted_list.append(result_container)
    
                if 'sum_of_finished_pixels' in file:                 
                    file = file.replace(f'{Feeder_Type}', 'Abkühlkurve Probe') 
                    finished_list.append(result_container)
                    
                if 'sum_of_total' in file:                 
                    file = file.replace(f'{Feeder_Type}', 'Resultatkurve Probe') 
                    total_list.append(result_container)
                
                sub_list = ["Speisermessung_", f"result_{substring}", ".wmv_", "_curve_fit100.npy",'_.npy','_result','__result'] #liste der zu entfernenden substrings vom file namen
                for sub in sub_list:
                    file = file.replace(sub, '')
                    
                 
                #file = file.replace(f'{Feeder_Type}', 'Probe') 
                
                
                
                
                
                
                file = file.replace('_', ' ')
                
                linename = f'{file}' + str(i)
                if 'sum_of_reacted_pixels' in file_name:
                    linename, = ax.plot(np.arange(result_container.shape[0]),result_container, label=f'{file} ZzP = {point_of_ignition}')
                else:
                    linename, = ax.plot(np.arange(result_container.shape[0]),result_container, label=f'{file}')
                #linename, = ax.plot(np.arange(result_container.shape[0]),result_container, label='SAME')
                lines.append(linename)
                
                #print(f'{file_name}len res container',len(result_container))
                
                i+=1


row=len(reacted_list)
column=len(reacted_list[0])
print(f' reacted_list Rows:{row}, reacted_list Column:{column}')
reacted_array = np.array(reacted_list)




















if 'sum_of_reacted_pixels' in plot_title:                 
    plot_title = plot_title.replace('sum_of_reacted_pixels', 'Reaktionskurve')
    
if 'sum_of_total' in plot_title:                 
    plot_title = plot_title.replace('sum_of_total', ' - Aufheizrate')  

if 'sum_of_finished_pixels' in plot_title:                 
    plot_title = plot_title.replace('sum_of_finished_pixels', ' - Abkühlkurve')
    



######### -------------------------Bestimme das Abkühlkriterium (basiert auf Feeder Evaluation final 3.2) --------------------------#######################
index_list = []
for Name in Feeder_Name_List:
    index_list.append(fuzz.partial_ratio(Name, Feeder_Type))#berechne für jedes file einen Grad der Übereinstimmung mit den Speisertypen/Speisertypenliste
index = np.argmax(index_list) #fine den höchsten Grad der Übereinstimmung/dessen Index in der Feeder_type_List
Feeder_Name = Feeder_Name_List[index] #Setze den Speisertypen des Files fest, um die entsprechende Maske zu laden
Feeder_Cooldown_Criteria = Feeder_Name_cooldown_criteria[index]

Feeder_Cooldown_Criteria_temp_deg_C = 600 + (900/255)*Feeder_Cooldown_Criteria
Feeder_Cooldown_Criteria_temp_deg_C = "{:.2f}".format(Feeder_Cooldown_Criteria_temp_deg_C) #Zwei Nachkommastellen


############### ---------------------- Legende und Titel schreiben ------------------------###################################


    
#plot_title += f'{ Feeder_Type} Cooldown criteria corresponding temperature {Feeder_Cooldown_Criteria_temp_deg_C}°C'
plot_title += f'{ Feeder_Type}'

#print(lines)
ax.set_title(f"{plot_title}")
leg = ax.legend(fancybox=True, shadow=True)
leg = plt.legend(loc=loc) #location right(5)



for legobj in leg.legendHandles:
    legobj.set_linewidth(10.0)


if not show_all_lines:
    for lh in leg.legendHandles: 
        lh.set_alpha(0.2)


lined = {}  # Will map legend lines to original lines.
for legline, origline in zip(leg.get_lines(), lines):
    legline.set_picker(True)  # Enable picking on the legend line.
    lined[legline] = origline

# Blende alle Kurven aus, wenn der trigger so gesetzt wurde
if not show_all_lines:
    print(lined)
    for line in lines:
        visible = not line.get_visible()
        line.set_visible(visible)
        
        
        
        #legline = line.artist
        #legline[line].set_alpha(1.0 if visible else 0.2)
        
        




def on_pick(event):
    # On the pick event, find the original line corresponding to the legend
    # proxy line, and toggle its visibility.
    print('EVENT:')
    print(event)
    legline = event.artist
    origline = lined[legline]
    visible = not origline.get_visible()
    origline.set_visible(visible)
    # Change the alpha on the line in the legend so we can see what lines
    # have been toggled.
    legline.set_alpha(1.0 if visible else 0.2)
    fig.canvas.draw()


#Result_Select_List = [sum_of_reacted_pixels, sum_of_finished_pixels, sum_of_total]
print(Result_Select_List)







#wenn nur aufheiz und abkühl ausgewählt wird ylim auf -100 bis 0 skalieren
if 'sum_of_reacted_pixels' in Result_Select_List and 'sum_of_finished_pixels' in Result_Select_List and not('sum_of_total' in Result_Select_List) :
    plt.ylim(ylim_lower,0)
    print('1')


if 'sum_of_reacted_pixels' in Result_Select_List and 'sum_of_total' in Result_Select_List and not('sum_of_finished_pixels' in Result_Select_List) :
    plt.ylim(0,ylim_upper)
    print('2')
if 'sum_of_reacted_pixels' in Result_Select_List and not('sum_of_total' in Result_Select_List) and not ('sum_of_finished_pixels' in Result_Select_List) :
    plt.ylim(0,ylim_upper)
    print('3')
if 'sum_of_reacted_pixels' in Result_Select_List and ('sum_of_total' in Result_Select_List and ('sum_of_finished_pixels' in Result_Select_List)) :
    
    plt.ylim(ylim_lower,ylim_upper)
    print('4')
    

    
 







fig.canvas.mpl_connect('pick_event', on_pick)
print(max_number_of_frames)
plt.xlim(0, 7000)
plt.xlabel('[frames]')
plt.ylabel('[%]')
plt.show()

"""                    
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("frame")
plt.ylabel("pixel")
#plt.title(f"{plot_title}")

  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()
"""