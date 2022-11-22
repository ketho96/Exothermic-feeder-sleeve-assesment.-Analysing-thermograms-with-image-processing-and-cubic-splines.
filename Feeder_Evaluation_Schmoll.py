# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy.interpolate import  LSQUnivariateSpline
import time


if __name__ == '__main__':       
    """##########################################################    Einstellungen         #####################################"""
    
    curve_fit_spline = 100 #regler für den grad der glättung des splines, je größer desto gröber
    std_bin_size = 20 #regler für das intervall der volatilität/standardabweichung je kleiner desto sensibler
    
    
    # Beispiel für alle Bilder in der BA
    #VideoDateiName = 'Speisermessung_Foseco_Kalminex_ZF_7_9_Frisch_1.2.wmv'
    
    
    VideoDateiName = 'Speisermessung_Chemex_CB_40_GK70_100_R_1.2.wmv'
    
    Evaluation = False # Schalter, ob Auswertung durchgeführt werden soll
    show_video = False
    
    videoName = VideoDateiName
    
    #Schreibe immer ein paar von Koordinaten x:y in das Dictionary um mehrere Auswertungen für verschiedene Pixel zu bekommen, jeder x Wert darf in dem Dictionary nur einmal auftauchen (keine Duplikate)     
    coordinatesSpeiser = {369:392,383:372,283:362}
    coordinatesFlammen = {}
    
    """##########################################################################################################################"""
    
    ######################################
    #%% Programm
    ######################################
    
    dataFolder = 'Data'
    
    # Bereich des Videos mit der Aufnahme (ohne Temperautrskala, Rahmen, ...)
    regionOfInterest = [29, 656, 73, 542] # [xmin, xmax, ymin, ymax]
    #regionOfInterest = [0, 800, 0, 600] # [xmin, xmax, ymin, ymax] 
    
    # Framerate des Videos (kann theoretisch auch mit cap.get(cv2.CAP_PROP_FPS) abgerufen werden, ist hier nur leider nicht korrekt.)
    fps = 25
    
    # Dateiname für die csv-Datei (Inhalt frameNumber, area/px)
    dataOutputName = dataFolder+'/'+videoName[:-4]+'_areaValues.csv'
    # Dateiname für OutputVideo mit eingezeichneter Kontur
    videoOutputName = dataFolder+'/'+videoName[:-4]+'_outputVideo.avi'
    # Dateiname für Plot mit Fläche über Zeit
    plotAreaVsTimeName = dataFolder+'/'+videoName[:-4]+'_plotAreaVsTime.png'
    
        
    # Laden des Videos
    cap = cv2.VideoCapture(dataFolder+'/'+videoName)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    print('evaluating the total number of frames in the video')
    fn=0
    # solange das Video geöffnet ist = solange es dauert = solange noch neue Videobild geladen werden
    while(cap.isOpened()):
        # nächtes Videobild laden
        ret, frame = cap.read()

        # Wenn das Video nicht zuende ist = (ret == true) (falls ret == false konnte kein (weiteres) Bild geladen werden)
        if ret:

            fn += 1
            
        # Wenn das ret False (das Video ist zuende), beende die Schleife und löse die Videos aus dem Speicher
        else:
            cap.release()
            cv2.destroyAllWindows()
    
    totalframecount = fn
    
    print("The total number of frames in this video is ", totalframecount)
    
    curve_fit=std_bin_size
    
    #verringere die Anzahl der Frames, sodass sie durch std_bin_size ohne Rest teilbar sind
    Brenndauer = totalframecount
    divided = Brenndauer/curve_fit
    while int(divided)!=divided:
        Brenndauer-=1
        divided = Brenndauer/curve_fit
    print('Brenndauer/curve_fit:',Brenndauer/curve_fit, type(Brenndauer/curve_fit))

    startFrame = totalframecount-Brenndauer
    endFrame = totalframecount
    
    
    duration = endFrame-startFrame #duration, dauer des videos
    
    # Output Video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(videoOutputName, fourcc, fps, (regionOfInterest[1] - regionOfInterest[0], regionOfInterest[3] - regionOfInterest[2]))
    
    # Zählvariable (FrameNumber)
    fn = 0
    
    mask = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('ubyte')
    # Liste für Flächenwerte anlegen
    area = []
    x= 'p'
    z ='i'
    
    """#############################################################################################"""
    """Arrays INIT"""
    """#############################################################################################"""
    
    row =  regionOfInterest[3] - regionOfInterest[2]#dimensionen des bildes
    column = regionOfInterest[1] - regionOfInterest[0] #dimensionen des bildes
    
    
    
    pixel_vals = np.zeros((row,column,duration+1),dtype=np.ubyte) #array, das für jeden pixel alle werte über die zeit hält
    pixel_vals_showOneSlice = np.zeros((row,column),dtype=np.ubyte)
    mask = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('ubyte')
    list_all_frames = []
    
    """#############################################################################################"""
    """#############################################################################################"""
    """#############################################################################################"""
    """#############################################################################################"""
    """#############################################################################################"""

    try:
        open(f'{videoName}duration{duration}startframe{startFrame}.npy')
        pixel_vals_all_frames = np.load(f'{videoName}duration{duration}startframe{startFrame}.npy')
        print('pixel_vals_all_frames array was loaded')
    except FileNotFoundError:
        # Laden des Videos
        cap = cv2.VideoCapture(dataFolder+'/'+videoName)
        print('Beginn Videoverarbeitung')
        begin = time.time()
        # csv Datei zum Schreiben öffnen
        with open(dataOutputName, 'w', newline='') as csvFile:
            csvWriter = csv.writer(csvFile)
        
            # solange das Video geöffnet ist = solange es dauert = solange noch neue Videobild geladen werden
            while(cap.isOpened()):
                # nächtes Videobild laden
                ret, frame = cap.read()
        
                # Wenn das Video nicht zuende ist = (ret == true) (falls ret == false konnte kein (weiteres) Bild geladen werden)
                if ret:
                    
                    if show_video:
                        cv2.imshow('Raw camera image',frame)
                    
                    
                    # alles vor startFrame und nach endFrame überspringen
                    if fn<startFrame: #or fn>endFrame:
                        fn += 1
                        continue
                    # Region of Interest aus Videobild schneiden
                    roiframe = frame[regionOfInterest[2]:regionOfInterest[3],regionOfInterest[0]:regionOfInterest[1],:]
                    
        
                    # Videobild in Grauwerte umwandeln
                    roigray = cv2.cvtColor(roiframe, cv2.COLOR_BGR2GRAY)
                    
                    list_all_frames.append(roigray)
                    fn += 1
                    
                    if fn> endFrame:
                        break
                 
                
                # Wenn das ret False (das Video ist zuende), beende die Schleife und löse die Videos aus dem Speicher
                else:
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
            pixel_vals_all_frames = np.array(list_all_frames)
            pixel_vals_all_frames = pixel_vals_all_frames.transpose(1,2,0)
            print('pixel_vals_all_frames.shape',pixel_vals_all_frames.shape)
            print('saving array')
            #pixel_vals_all_frames = np.dstack(list_all_frames)
            #np.save(f'{videoName}duration{duration}startframe{startFrame}',pixel_vals_all_frames) #dauert unglaublich lange 
            print('array was saved')
            end = time.time()
            # total time taken
            print(f"Total runtime Videoprocessing: {end - begin}s")
            
            
            
        """#######################################################"""
        
        """#################ENDE DER VIDEOVERARBEITUNG###############"""
        
        """##########################################################"""
len_pixel_vals_all_frames = pixel_vals_all_frames.shape[2]-1


x_values    = np.linspace(0, len_pixel_vals_all_frames,len_pixel_vals_all_frames) #einfache Dichte der Werte/ Auflösung des Splines
x_values10  = np.linspace(0, len_pixel_vals_all_frames,len_pixel_vals_all_frames*10) #10fache Dichte
x_values100 = np.linspace(0, len_pixel_vals_all_frames,len_pixel_vals_all_frames*100) #100 fache Daichte


for x, y in coordinatesSpeiser.items():
        xS = x
        yS = y
        try:
            open(f'{videoName}duration{duration}startframe{startFrame}valuesxS{xS}yS{yS}.npy')
        except FileNotFoundError:
            values = pixel_vals_all_frames[yS,xS,:]
            print('saving values array')
            np.save(f'{videoName}duration{duration}startframe{startFrame}valuesxS{xS}yS{yS}',values) 
            print('array was saved') 
                
for x, y in coordinatesFlammen.items():
        xF = x
        yF = y
        try:
            open(f'{videoName}duration{duration}startframe{startFrame}valuesxF{xF}yF{yF}.npy')
        except FileNotFoundError:
            values = pixel_vals_all_frames[yF,xF,:]
            print('saving values array')
            np.save(f'{videoName}duration{duration}startframe{startFrame}valuesxF{xF}yF{yF}',values) 
            print('array was saved')  
 
    
try:
    open(f'mask{videoName}duration{duration}startframe{startFrame}_max.npy')
    mask_max = np.load(f'mask{videoName}duration{duration}startframe{startFrame}_max.npy')
    print('mask_max array was loaded')
except FileNotFoundError:
    #Array mit maximalen Werten erstellen        
    print('create mask_max')
    mask_max = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('ubyte')
    d=0
    r=0
      
    while d < pixel_vals_all_frames.shape[0]:
        r=0
        if d % 100 ==0 or d > 550:
            print(d,'/599 max')
        while r < pixel_vals_all_frames.shape[1]:
            
            mask_max[d,r] = pixel_vals_all_frames[d,r,:].max()
            r+=1
        d+=1
    print('saving mask _max array')
    np.save(f'mask{videoName}duration{duration}startframe{startFrame}_max',mask_max) 
    print('array was saved')


mask_max_copy = mask_max

# Kreuze der Pixel einzeichnen
for x, y in coordinatesSpeiser.items():
        
        if y > 463: #469 ist der Rand der Maske
            y = 463
        if x > 621: #627 ist der Rand der Maske
            x = 621
            
        xS = x
        yS = y
           
        i=0
        while i <= 10:
            mask_max_copy[yS-5+i,xS]=255
            i+=1
        i=0
        while i <= 10:
            mask_max_copy[yS,xS-5+i]=255
            i+=1
            
for x, y in coordinatesFlammen.items():
        
        if y > 463: #469 ist der Rand der Maske
            y = 463
        if x > 621: #627 ist der Rand der Maske
            x = 621
        
        xS = x
        yS = y
        
        i=0
        while i <= 10:
            mask_max_copy[yS-5+i,xS]=255
            i+=1
        i=0
        while i <= 10:
            mask_max_copy[yS,xS-5+i]=255
            i+=1
        i+=1

plt.figure()
plt.imshow(mask_max_copy, cmap='gray', vmin=0, vmax=255)
plt.show()

    
try:
    open(f'mask_avg{videoName}duration{duration}startframe{startFrame}.npy')
    mask_avg = np.load(f'mask_avg{videoName}duration{duration}startframe{startFrame}.npy')
    print('mask_avg array was loaded')
except FileNotFoundError:
    #Array mit Durchschnittswerten erstellen        
    print('create mask_avg')
    mask_avg = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('ubyte')
    d=0
    r=0
    while d < pixel_vals_all_frames.shape[0]:
        r=0
        if d % 100 ==0 or d > 550:
            print(d, '/599 mask_avg')
        while r < pixel_vals_all_frames.shape[1]:
            #if max(pixel_vals_all_frames[d,r,:])>0:
            if  (mask_max[d,r]<25): #(mask_vol_mean[d,r]>thresh_vol) or
                r+=1
            else:
                array_container=pixel_vals_all_frames[d,r,:]
                mask_avg[d,r] = np.true_divide(array_container.sum(0),(array_container!=0).sum(0)) #average ohne 0 
                r+=1
        d+=1
    print('saving mask_avg array')
    np.save(f'mask_avg{videoName}duration{duration}startframe{startFrame}',mask_avg) 
    print('array was saved')


curve_fit=curve_fit_spline


try:
    open(f'{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}mask_spline_max.npy')
    mask_spline_max = np.load(f'{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}mask_spline_max.npy')
    print('mask_spline_max array was loaded')
except FileNotFoundError:
    #Array mit dem maximum des Splines erstellen   
    print('create mask_spline')
    knots=np.array(range(curve_fit,pixel_vals_all_frames.shape[2]-curve_fit,curve_fit)) #hier werden die Positionen der Knoten definiert (evenly spaced mit der distanz von curve fit) changeLOG -curve_fit neu eingefügt
    d=0
    r=0
    mask_spline_max = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('float')
    mask_spline_integral = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('float')
    array_container = np.array((pixel_vals_all_frames.shape[2],1),dtype= float) # temporärer speicher für pixel vals all frames für einen pixel mit den koordinaten der iteration
    x0 = [pixel_vals_all_frames.shape[2]/2]
    bnds = [(0,pixel_vals_all_frames.shape[2])]
    minimizer_kwargs = { "method": "Nelder-Mead","bounds":bnds }
    while d < pixel_vals_all_frames.shape[0]:
        r=0
        if d % 100 ==0 or d > 550:
            print(d,'/599 max')
        while r < pixel_vals_all_frames.shape[1]:
            #if max(pixel_vals_all_frames[d,r,:])>0:
            if  (mask_max[d,r]<70): #(mask_vol_mean[d,r]>thresh_vol) or
                r+=1
            else:
                i=0
                array_container = pixel_vals_all_frames[d,r,:]
                spline = LSQUnivariateSpline(x_values, array_container[0:len_pixel_vals_all_frames], knots)
                #mask_spline_max[d,r] = max(spline(x_values))
                mask_spline_max[d,r] = (max(spline(x_values)))
                r+=1
            #r+=1
        d+=1
    
    #mask_spline_max=mask_spline_max*2 #Kontrast erhöhen
    
    print('saving mask_spline_max array')
    np.save(f'{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}mask_spline_max',mask_spline_max) 
    print('array was saved')

curve_fit=std_bin_size

try:
    open(f'mask_vol_mean{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}.npy')
    mask_vol_mean = np.load(f'mask_vol_mean{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}.npy')
    print('mask_vol_mean array was loaded')
except FileNotFoundError:
    #Array mit der volatilität/standardabweichung erstellen   
    print('create mask_vol_mean')
    #mask_max = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('ubyte')
    d=0
    r=0
    mask_vol_mean = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('float')
    array_container = np.array((pixel_vals_all_frames.shape[2],1),dtype= float)
    while d < pixel_vals_all_frames.shape[0]:
        r=0
        if d % 100 ==0 or d > 550:
            print(d,'/599 vol')
        while r < pixel_vals_all_frames.shape[1]:
            #if max(pixel_vals_all_frames[d,r,:])>0:
            if mask_max[d,r]<70:
                r+=1
            else:
                array_container = pixel_vals_all_frames[d,r,:]
                
                

                values = array_container
                values_conv = np.reshape(values,(round(len(values)/curve_fit),curve_fit)) #schreibe immer curve_fit anzahl an werten in eine spalte 
                df_values_conv = pd.DataFrame(values_conv)
                df_values_conv[df_values_conv==0]=np.NaN #ändere alle nullen in NaN um diese bei der standardabweichungsberechnung zu ignorieren (wenn der Messwer 0 ist)
                df_std = df_values_conv.std(axis=1, skipna=True) #berechne die standardabweichung jeder spalte (also curve_fit anzahl an werten)
                mask_vol_mean[d,r] = df_std.mean() #gib von allen standardabweichungen aller bins den durchschnitt aus
# =============================================================================
#                
#                 array_container = array_container.astype(float) # NaN elemente sind float elemente, desewegen muss das array in float datentyp gewandelt werden
#                 array_container = np.reshape(array_container, (int(array_container.shape[0]/std_bin_size), std_bin_size)) # array, anzahl zeilen, anzahl spalten
#                 array_container[array_container==0] = np.NaN
#                 array_container = np.nanstd(array_container, axis = 0) #berechne die standardabweichung ohne nan für jede spalte entlang der achse zeile
#                 mask_vol_mean[d,r] = array_container.mean()
# =============================================================================
               
                
               
                r+=1
            #r+=1
        d+=1
    
    print('saving mask_vol_mean array')
    np.save(f'mask_vol_mean{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}',mask_vol_mean) 
    print('array was saved')


mask_vol_mean_max    = np.amax(mask_vol_mean) 
mask_vol_mean_avg = np.average(mask_vol_mean,weights=mask_vol_mean.astype(bool)) # durchschnitt ohne 0 ?? was macht diese zeile?
#thresh_vol = mask_vol_mean_max-mask_vol_mean_avg-1
thresh_vol = mask_vol_mean_avg
thresh_vol = thresh_manually = 4


        
mask_spline_max_max = np.amax(mask_spline_max)  
mask_spline_max_avg = np.average(mask_spline_max,weights=mask_spline_max.astype(bool))


print('mask_spline_max_max',mask_spline_max_max)
print('mask_vol_mean_max',mask_vol_mean_max)
print('mask_spline_max_avg',mask_spline_max_avg)
print('mask_vol_mean_avg',mask_vol_mean_avg)

#thresh_max = mask_spline_max_max - mask_spline_max_avg
#thresh_max = mask_spline_max_avg
thresh_max = int(mask_spline_max_avg*0.95)
thresh_max = thresh_manually = 150

print('thresh_max',thresh_max)
print('thresh_vol',thresh_vol)


try:
    open(f'mask{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}.npy')
    mask = np.load(f'mask{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}.npy')
    print('mask array was loaded')
except FileNotFoundError:
    #Array mit der volatilität/standardabweichung erstellen   
    print('create mask')
    d=0
    r=0
    while d < pixel_vals_all_frames.shape[0]:
        r=0
        if d % 100 ==0 or d > 550:
            print(f'd:{d}')
        while r < pixel_vals_all_frames.shape[1]:
            #if max(pixel_vals_all_frames[d,r,:])>0:
            if (mask_vol_mean[d,r] == 0 or
                mask_spline_max[d,r] < 80 and
                mask_vol_mean[d,r] > 2 or
                mask_vol_mean[d,r] > 4 or
                mask_spline_max[d,r] < 70): 
                #Kriterien, die einen NICHT SPEISER PIXEL identifizieren
                r+=1
            else:
                mask[d,r]=255
                r+=1
        d+=1
    print('saving mask array')
    np.save(f'mask{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}',mask) 
    print('array was saved')


# Kreuze der Pixel einzeichnen
for x, y in coordinatesSpeiser.items():
        if y > 463: #469 ist der Rand der Maske
            y = 463
        if x > 621: #627 ist der Rand der Maske
            x = 621
        xS = x
        yS = y
        
        i=0
        while i <= 10:
            mask[yS-5+i,xS]=100
            i+=1
        i=0
        while i <= 10:
            mask[yS,xS-5+i]=100
            i+=1
            
for x, y in coordinatesFlammen.items():
        
        if y > 463: #469 ist der Rand der Maske
            y = 463
        if x > 621: #627 ist der Rand der Maske
            x = 621
        xS = x
        yS = y
        i=0
        while i <= 10:
            mask[yS-5+i,xS]=100
            i+=1
        i=0
        while i <= 10:
            mask[yS,xS-5+i]=100
            i+=1
        i+=1
        

# (mask_max zum ablesen von schwellwerten 
plt.figure(3)
plt.title('mask_max')
plt.imshow(mask_max, cmap='gray', vmin=0, vmax=255)
plt.show()
#'mask_spline_maxn zum ablesen von schwellwerten 
plt.figure(4)
plt.title('mask_spline_max')
plt.imshow(mask_spline_max, cmap='gray', vmin=0, vmax=150)
plt.show()
#mask_vol_mean zum ablesen von schwellwerten 
plt.figure(5)
plt.title('mask_vol_mean')
plt.imshow(mask_vol_mean, cmap='gray', vmin=0, vmax=4)
plt.show()
#mask
plt.figure(8)
plt.title('Segmentation mask')
plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
plt.show 



knots=np.array(range(curve_fit,pixel_vals_all_frames.shape[2]-curve_fit,curve_fit))


array_container = pixel_vals_all_frames[yS,xS,:]

spline_Speiser = LSQUnivariateSpline(x_values, array_container[0:len_pixel_vals_all_frames], knots)
y_spline_Speiser = spline_Speiser(x_values100)
spline_Speiser_deriv = spline_Speiser.derivative()
df = pd.DataFrame(array_container)

array_container = pixel_vals_all_frames[yF,xF,:]
spline_flamme = LSQUnivariateSpline(x_values, array_container[0:len_pixel_vals_all_frames], knots)
y_spline_flamme = spline_flamme(x_values100)
spline_flamme_deriv = spline_flamme.derivative()
df = pd.DataFrame(array_container)



if Evaluation:
                
    try:
        open(f'result_number_of_reacted_pixels{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}.npy')
        result_number_of_reacted_pixels = np.load(f'result_number_of_reacted_pixels{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}.npy')
        print('result_number_of_reacted_pixels array was loaded')
        
        open(f'result_sum_of_reacted_pixels{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}.npy')
        result_sum_of_reacted_pixels = np.load(f'result_sum_of_reacted_pixels{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}.npy')
        print('result_sum_of_reacted_pixels array was loaded')
        
        open(f'{videoName}duration{duration}startframe{startFrame}result_number_of_finished_pixels.npy')
        result_number_of_finished_pixels = np.load(f'{videoName}duration{duration}startframe{startFrame}result_number_of_finished_pixels.npy')
        print('result_number_of_finished_pixels array was loaded')
        
        open(f'result_sum_of_finished_pixels{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}.npy')
        result_sum_of_finished_pixels = np.load(f'result_sum_of_finished_pixels{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}.npy')
        print('result_sum_of_finished_pixels array was loaded')
        
        open(f'{videoName}duration{duration}startframe{startFrame}result_sum_of_total.npy')
        result_sum_of_total = np.load(f'{videoName}duration{duration}startframe{startFrame}result_sum_of_total.npy')
        print('result_sum_of_total array was loaded')
    except FileNotFoundError:
        #Array mit der volatilität/standardabweichung erstellen   
        print('Evaluation')
        #mask_max = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('ubyte')
        d=0
        r=0
        
        #curve fit für die auswertung gröber gewählt, um eine stärkere glättung zu erreichen und damit flammen über speiser pixeln zu ignorieren
        curve_fit = 100
        #knots=np.array(range(curve_fit,pixel_vals_all_frames.shape[2]-5*curve_fit,curve_fit)) #hier werden die Positionen der Knoten definiert (evenly spaced mit der distanz von curve fit) der erste knoten bei curve_fit und der letzte knoten bei end frame-2*curve_fit
        #knots=np.array(range(curve_fit,pixel_vals_all_frames.shape[2]-curve_fit,curve_fit))
        knots=np.array(range((len_pixel_vals_all_frames-curve_fit),curve_fit,-curve_fit)) #erzeugt die knoten vom letzten punkt aus bis nach vorne
        knots = knots[::-1] #kehrt die reihenfolge der knoten um, da range nur so herum arbeiten kann (knoten von hinten erstellen)
        x_values=np.linspace(0, pixel_vals_all_frames.shape[2],pixel_vals_all_frames.shape[2])
        
        mask_ERROR = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('float')
        #mask_spline_max = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('float')
        #mask_spline_integral = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('float')
        array_container = np.array((pixel_vals_all_frames.shape[2],1),dtype= float) # temporärer speicher für pixel vals all frames für einen pixel mit den koordinaten der iteration
        result_number_of_reacted_pixels = np.zeros(pixel_vals_all_frames.shape[2])
        result_number_of_finished_pixels = np.zeros(pixel_vals_all_frames.shape[2])
        result_sum_of_reacted_pixels = np.zeros(pixel_vals_all_frames.shape[2])
        result_sum_of_finished_pixels = np.zeros(pixel_vals_all_frames.shape[2])
        result_sum_of_total = np.zeros(pixel_vals_all_frames.shape[2])
        #spline_container = np.zeros((regionOfInterest[3] - regionOfInterest[2], regionOfInterest[1] - regionOfInterest[0])).astype('LSQUnivariateSpline')
        x_values100 = np.linspace(0, pixel_vals_all_frames.shape[2]-1,pixel_vals_all_frames.shape[2]*100)
        while d < pixel_vals_all_frames.shape[0]:
            r=0
            print(f'd:{d}/469')
            while r < pixel_vals_all_frames.shape[1]:
                #if max(pixel_vals_all_frames[d,r,:])>0:
                if mask[d,r]==0:
                    r+=1
                else:
                    
                    i=0
                    values = pixel_vals_all_frames[d,r,:]
                    
                    spline = LSQUnivariateSpline(x_values, values, knots) #Spline Funktion initiieren
                    spline_deriv = spline.derivative() #Ableitung initiieren
    
                    y_spline = spline(x_values) #Plotte die Funktionswerte über x (definiert ist die einfache, zehnfache und 100 fache Auflösung der x Werte)
                    y_spline_deriv = spline_deriv(x_values)
                    
                    try:
                        #Punkt finden, an dem die Steigung des Splines maximal ist = Beginn der Exothermen Reaktion
                        frame_max_deriv = round((int((np.where(y_spline_deriv == max(y_spline_deriv)))[0])))
                        result_number_of_reacted_pixels[frame_max_deriv]+=1 
                        
                        #Punkt nach dem Beginn der exothermen Reaktion finden, an dem der Wert maximal wird = Ende der Exothermen Reaktion
                        x_values_after_max_deriv = np.linspace(frame_max_deriv,len_pixel_vals_all_frames,(len_pixel_vals_all_frames-frame_max_deriv)) #schneide die Funktionswerte so zu, dass nur Werte nach frame_max_deriv ausgewertet werden
                        y_spline_after_max_deriv = spline(x_values_after_max_deriv)
                        frame_max_after_max_deriv = round(int((np.where(y_spline_after_max_deriv == max(y_spline_after_max_deriv)))[0]))
                        result_number_of_finished_pixels[frame_max_after_max_deriv+frame_max_deriv]-=1
                    except:
                        try:
                            #versuche einen Überhang zu bilden, um Ausschläge des Splines am rechten Ende zu unterbinden
                            len_values_init=values.shape[0]-1
                            appendix = values[-1] #der letzte Wert der Messkurve wird konstant angehangen
                            appendix_list=[]
                            while i< curve_fit*5: 
                                appendix_list.append(appendix)
                                i+=1
                            values=np.append(values, appendix_list)
                            #Punkt finden, an dem die Steigung des Splines maximal ist = Beginn der Exothermen Reaktion
                            frame_max_deriv = round((int((np.where(y_spline_deriv == max(y_spline_deriv)))[0])))
                            result_number_of_reacted_pixels[frame_max_deriv]+=1 
                            
                            #Punkt nach dem Beginn der exothermen Reaktion finden, an dem der Wert maximal wird = Ende der Exothermen Reaktion
                            x_values_after_max_deriv = np.linspace(frame_max_deriv,len_pixel_vals_all_frames,(len_pixel_vals_all_frames-frame_max_deriv)) #schneide die Funktionswerte so zu, dass nur Werte nach frame_max_deriv ausgewertet werden
                            y_spline_after_max_deriv = spline(x_values_after_max_deriv)
                            frame_max_after_max_deriv = round(int((np.where(y_spline_after_max_deriv == max(y_spline_after_max_deriv)))[0]))
                            try:
                                result_number_of_finished_pixels[frame_max_after_max_deriv+frame_max_deriv]-=1 #erniedrige den zähler der abgeklungenen pixel zum zeitpunkt/frame, der gefunden wurde
                            except:
                                result_number_of_finished_pixels[len_values_init-1]-=1 #liegt das Maximum genau am Ende der Messkurve nimm an, dass die Reaktion dort beendet ist
                            print(f'appendix created @ d:{d}/r:{r}')
                            mask_ERROR[d,r]=200
                        except:
                            try:
                                #versuche zumindest den Ort der maximalen Steigung = Zündzeitpunkt zu ermitteln = unendlich lange Brenndauer aber besser als nichts
                                #versuche einen Überhang zu bilden, um Ausschläge des Splines am rechten Ende zu unterbinden
                                appendix = values[-1] #der letzte Wert der Messkurve wird konstant angehangen
                                appendix_list=[]
                                while i< curve_fit*5: 
                                    appendix_list.append(appendix)
                                    i+=1
                                values=np.append(values, appendix_list)
                                #Punkt finden, an dem die Steigung des Splines maximal ist = Beginn der Exothermen Reaktion
                                frame_max_deriv = round((int((np.where(y_spline_deriv == max(y_spline_deriv)))[0])))
                                result_number_of_reacted_pixels[frame_max_deriv]+=1 
                                print(f'Reaction only @ d:{d}/r:{r}')
                                mask_ERROR[d,r]=200
                            except:
                                #Wenn nichts davon funktioniert hat, vernachlässige den pixel
                                print(f' Fatal Error @ d:{d}/r:{r}')
                                mask_ERROR[d,r]=200
                                pass
                            
                                
                                
                    
                        
                        
                        
                        
                    
                    r+=1
            d+=1
                  
        
        
        print('saving result_number_of_reacted_pixels')
        np.save(f'result_number_of_reacted_pixels{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}',result_number_of_reacted_pixels) 
        print('array was saved')
        
        print('saving result_number_of_finished_pixels')
        np.save(f'{videoName}duration{duration}startframe{startFrame}result_number_of_finished_pixels',result_number_of_finished_pixels) 
        print('array was saved')
        
        print('saving mask_ERROR')
        np.save(f'{videoName}duration{duration}startframe{startFrame}mask_ERROR',mask_ERROR) 
        print('array was saved')
        
        
        
        i=0
        while i < len(result_number_of_reacted_pixels):
            result_sum_of_reacted_pixels[i] = sum(result_number_of_reacted_pixels[0:i+1])
            i+=1
    
        i=0
        while i < len(result_number_of_reacted_pixels):
            result_sum_of_finished_pixels[i] = sum(result_number_of_finished_pixels[0:i])
            i+=1
    
        i=0
        while i < len(result_number_of_reacted_pixels):
            result_sum_of_total[i]=result_sum_of_reacted_pixels[i]+result_sum_of_finished_pixels[i]
            i+=1    
        
        print('saving result_sum_of_reacted_pixels')
        np.save(f'result_sum_of_reacted_pixels{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}',result_sum_of_reacted_pixels) 
        print('array was saved')
        
        print('saving result_sum_of_finished_pixels')
        np.save(f'result_sum_of_finished_pixels{videoName}duration{duration}startframe{startFrame}curve_fit{curve_fit}',result_sum_of_finished_pixels) 
        print('array was saved')
        
        print('saving result_sum_of_total')
        np.save(f'{videoName}duration{duration}startframe{startFrame}result_sum_of_total',result_sum_of_total) 
        print('array was saved')
        
            
                
    fig = plt.figure()
    ax = fig.subplots()
    #ax.plot(np.arange(pixel_vals_all_frames.shape[2]),pixel_vals_all_frames[yS,xS,:], color = 'r', alpha=0.5)
    #ax.plot(np.arange(pixel_vals_all_frames.shape[2]),pixel_vals_all_frames[yF,xF,:], color = 'b', alpha=0.5)
    #ax.plot(x_values100,y_spline_flamme, color = 'g') 
    #ax.plot(x_values100,y_spline_Speiser, color = 'g') 
    ax.plot(np.arange(result_number_of_reacted_pixels.shape[0]),result_sum_of_total, color = 'g') 
    ax.plot(np.arange(result_number_of_reacted_pixels.shape[0]),result_number_of_finished_pixels, color = 'b') 
    ax.plot(np.arange(result_sum_of_reacted_pixels.shape[0]),result_number_of_reacted_pixels, color = 'r')
    
    
    #plt.title(f'X-Coordinate{c}')
    plt.ylim(-10, 10)
    plt.xlabel('frame')
    plt.ylabel('value')
    plt.grid()
    plt.show()
    









