# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:20:20 2019

@author: saura
"""

import os
import pandas as pd
import datetime
from datetime import timedelta
import numpy as np
import math
from sklearn.cluster import KMeans
import scipy.signal
import pylab as pl




def haversine(coord1, coord2):
    """
    Haversine distance in meters(but we converted in to Km by dividing 1000) for two (lat, lon) coordinates
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    radius = 6371000 # mean earth radius in meters (GRS 80-Ellipsoid)
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d/1000

def convolution(x,weight,sigma,convpoints):
    """
    For convolution of points i.e converting discrete data into continous time data using convolution algorithm
    
    """
    
    conv = [i for i in range(len(convpoints))]
    dmin = np.min(convpoints)
    dmax = np.max(convpoints)
  
    for i in range(len(convpoints)):
      dNu = 0
      for j in range(len(x)):
         dNu = dNu + weight[j]*np.exp(-0.5 * np.power((x[j] - convpoints[i])/sigma,2))
     
      dDen = scipy.stats.norm.cdf(dmax,convpoints[i],sigma) - scipy.stats.norm.cdf(dmin,convpoints[i],sigma)
    
      conv[i] = (dNu /(dDen * np.sqrt(2*np.pi) * sigma))
 
    return conv


def convolution_peaks(convolve_list):
    """
    For finding convolution peaks in data using scipy.signal package in python
    
    """
    
    convolve_list_peak = scipy.signal.find_peaks(convolve_list)[0].tolist()
                
    if len(convolve_list_peak) == 0:
        convolve_list_peak.append(convolve_list.index(np.max(convolve_list)))
    
    elif len(convolve_list_peak) < 2:
        rightbound = scipy.signal.peak_prominences(convolve_list,[convolve_list_peak[0]])[2][0]
        leftbound =  scipy.signal.peak_prominences(convolve_list,[convolve_list_peak[0]])[1][0]
        
        if leftbound == 0:
          convolve_list_peak.append(0)

        else:
          convolve_list_peak.append(convolve_list[:leftbound].index(np.max(convolve_list[:leftbound])) + leftbound)
         
        if rightbound == len(convolve_list) - 1:
          convolve_list_peak.append(len(convolve_list) - 1)

        else:
          convolve_list_peak.append(convolve_list[rightbound+1:].index(np.max(convolve_list[rightbound+1:])) + rightbound+1)
        
    else:
        rightbound = scipy.signal.peak_prominences(convolve_list,[convolve_list_peak[len(convolve_list_peak) - 1]])[2][0]
        leftbound =  scipy.signal.peak_prominences(convolve_list,[convolve_list_peak[0]],wlen = convolve_list_peak[1])[1][0]
         
        if leftbound == 0:
          convolve_list_peak.append(0)
    
        else:
          convolve_list_peak.append(convolve_list[:leftbound].index(np.max(convolve_list[:leftbound])) + leftbound)
         
        if rightbound == len(convolve_list) - 1:
          convolve_list_peak.append(len(convolve_list) - 1)
    
        else:
          convolve_list_peak.append(convolve_list[rightbound+1:].index(np.max(convolve_list[rightbound+1:])) + rightbound+1)
        
        
    return convolve_list_peak

def peak_std_high(peaks,convpoints,conv):
     """
     For finding range of high peak using scipy.signal package in python
    
     """
     m = np.max([convpoints[i] for i in peaks])
     w = np.max([conv[i] for i in peaks])
     index = peaks.index([i for i in peaks if conv[i] == w][0])
     
     if w  == 0:
         return 0
     
     else:
         try :
             r = scipy.signal.peak_prominences(conv,[peaks[index]],wlen = peaks[index] + 1)[1:3]
             low = r[0][0]
             up =  r[1][0]
             
             
         
         except ValueError:
             if index == 0:
                low = 0
                up = conv[0:].index(np.max(conv[1:]))
            
             elif index == len(convpoints) - 1:
                low = conv[:100].index(np.max(conv[:100]))
                up = len(convpoints) - 1
                             
             elif index == len(peaks) - 2:
                low = 0
                up = scipy.signal.peak_prominences(conv,[peaks[0]],wlen = peaks[0] + 1)[1][0]
                
             elif index == len(peaks) - 1:
                r = scipy.signal.peak_prominences(conv,[peaks[len(peaks) - 3]],wlen = peaks[len(peaks) - 3])[1:3]
                low = r[1][0]
                if low == len(convpoints) - 1:
                    up = low
                else:
                 up =  convpoints[r[1][0]:len(convpoints)-1].index(np.max(convpoints[r[1][0]:len(convpoints)-1])) + r[1][0]
        
         variance = 0
         weight = 0
         for i in list(range(low,up+1)):
             variance = variance + np.power((m*w) - (convpoints[i] * conv[i]),2)
             weight = weight + conv[i]
         if weight == 0:
             return 0
         else:
             return np.sqrt(variance / weight)
        
def peak_std_low(peaks,convpoints,conv):
    
     """
     For finding range of low peak using scipy.signal package in python
    
     """
     m = np.min([convpoints[i] for i in peaks])
     w = np.min([conv[i] for i in peaks])
     index = peaks.index([i for i in peaks if conv[i] == w][0])
     
     if  w == 0:
         return 0
     
     else:
         
         try :
             r = scipy.signal.peak_prominences(conv,[peaks[index]],wlen = peaks[index] + 1)[1:3]
             low = r[0][0]
             up =  r[1][0]
             
             
         
         except ValueError:
             if index == 0:
                low = 0
                up = conv[0:].index(np.min(conv[1:]))
            
             elif index == len(convpoints) - 1:
                low = conv[:100].index(np.min(conv[:100]))
                up = len(convpoints) - 1
                             
             elif index == len(peaks) - 2:
                low = 0
                up = scipy.signal.peak_prominences(conv,[peaks[0]],wlen = peaks[0] + 1)[1][0]
                
             elif index == len(peaks) - 1:
                r = scipy.signal.peak_prominences(conv,[peaks[len(peaks) - 3]],wlen = peaks[len(peaks) - 3])[1:3]
                low = r[1][0]
                if low == len(convpoints) - 1:
                    up = low
                else:
                    up = convpoints[r[1][0]:len(convpoints)-1].index(np.min(convpoints[r[1][0]:len(convpoints)-1])) + r[1][0]
        
         variance = 0
         weight = 0
         for i in list(range(low,up+1)):
             variance = variance + np.power((m*w) - (convpoints[i] * conv[i]),2)
             weight = weight + conv[i]
         
         if weight == 0:
             return 0
         else:
             return np.sqrt(variance / weight)
        
 
def geometry(points):
      """
      For finding how many segemnts lie in same coordinate range
    
      """

      points = points.reset_index(drop=True)
      X = pd.DataFrame()
      X = X.append(points['Centroid Point'].tolist())
      X.columns = ['Latitude','Longitude']
      n = len(points)
      
      
      cluster_count = 2
      
      if n > 5:
         uniquec = []
         while(True):
              cluster_count = 2
              kmeans = KMeans(n_clusters=cluster_count) 
              estimator = kmeans.fit(X[['Latitude','Longitude']])
              X['Label'] = estimator.labels_
              for i in range(0,cluster_count):
                  for j in range(i+1,cluster_count):
                          dist = haversine((estimator.cluster_centers_[i]),(estimator.cluster_centers_[j]))
                          if dist > 100:
                            uniquec.append(i)
              if len(np.unique(uniquec)) != n:
                 break
              else:
                 cluster_count = cluster_count + 1
                 
      else:
            uniquec = []
            for i in range(0,n):
              for j in range(i+1,n):
                      dist = haversine((X['Latitude'][i],X['Longitude'][i]),(X['Latitude'][j],X['Longitude'][j]))
                      if dist > 100:
                        uniquec.append(i)
                    
      return len(points['Stretch']) - len(np.unique(uniquec))         
   
    
    
    
def feature(f,dir):        
      dfi = pd.DataFrame(columns = ['IMEI',
                                      
                                      'Total Stretch',               # Total Stretch number
                                          
                                      'Total Segment',               # Total Segments
                                          
                                      'Segments in same Geography',  # Segments coming in same geographical area
                                       
                                      'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday', # Total Stretches in this Weekday
                                      #Total Stretches in this hour
                                      '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23'
                                          
                                       ])      
        
        
      dfs = pd.DataFrame(columns = ['IMEI',
                                      
                                      'Stretch',                     # Stretch number
                                          
                                      'Total Segment',               # Total Segments in each stretch
                                          
                                      'Segments in same Geography',  # Segments coming in same geographical area
                                          
                                      'Start Day of Week',           # Start day of week of stretch    
                                          
                                      'Start hour of day',           # Start hour of day of stretch 
                                          
                                      'Total time of stretch'])      # Total time of each stretch
    
        
      dft =  pd.DataFrame(columns = ['IMEI','Stretch','Segment', # For each segment of data in a stretch
                                       
                                       'Total Points',                  # Total data points
                                       
                                       'Start Hour',                   # Start hour of data
                                       
                                       'Start Day',                    # Start day of data
                                       
                                       'Total Distance',               # Distance Travelled in each segment of a particular stretch
                                       
                                       'Total Time',                   # Time between in each segment of data
                                       
                                       'Displacement',                 # Distance Travelled between particular end points of a particular segment
                                       
                                       'Speed',                        # Speed of the vehicle in this segment
                                       
                                       'Speed_SD',                     # S.D of speed of the vehicle in this segment where speed > 0
                                       
                                       'Speed_Max',                    # Maximum Speed of the vehicle in this segment
                                                                         
                                       'Speed_Peak_Total',             # Total number of  Speed peaks of the vehicle in this segment
                                       
                                       'Speed_Peak_Highest',           # Weight of highest peak in speed calculation
                                                                          
                                       'Speed_Max_Peak_Mean',          # Mean of max peaks in speed
                                       
                                       'Speed_Max_Peak_SD',            # SD of max peaks in speed
                                       
                                       'Total Driving Time',           # Total driving time where distance > 0
                                       
                                       'Continous Driving Distance',   # At a segment how much distance driver is driving without breaks i.e total effective distance in total segment
                                       
                                       'Continous Driving Time',       # At a segment how much time driver is driving without breaks i.e total effective distance in total segment                                       
                                       
                                       'Acceleration_Max',             # Maximum acceleration during the segment
                                       
                                       'Retardation_Max',              # Maximum Retardation during the segment
                                       
                                       'Acceleration_Mean',            # Mean acceleration during the segment where acceleration != 0
                                       
                                       'Retardation_Mean',             # Mean Retardation during the segment where acceleration != 0
                                       
                                       'Acceleration_SD',              # SD acceleration during the segment where acceleration != 0
                                       
                                       'Retardation_SD',               # SD Retardation during the segment where acceleration != 0
                                       
                                       'Acceleration Peaks',           # Total peaks of acceleration and deaccelertion
                                       
                                       'Total Acceleration Peaks',     # Total peaks in acceleration
                                       
                                       'Acceleration_Peak_Highest',    # Weight of highest peak in speed calculation
                                       
                                       'Acceleration_Max_Peak_Mean',   # Mean of max peaks in acceleration
                                       
                                       'Acceleration_Max_Peak_SD',     # SD of max peaks in acceleration
                                       
                                       'Acceleration_Peak_Minimum',    # Weight of lowest peak in speed calculation
                                       
                                       'Acceleration_Min_Peak_Mean',   # Mean of min peaks in acceleration
                                       
                                       'Acceleration_Min_Peak_SD',     # SD of min peaks in acceleration
                                       
                                       'Total Retardation Peaks',     # Total peaks in Retardation
                                       
                                       'Retardation_Peak_Highest',    # Weight of highest peak in speed calculation
                                       
                                       'Retardation_Max_Peak_Mean',   # Mean of max peaks in Retardation
                                       
                                       'Retardation_Max_Peak_SD',     # SD of max peaks in Retardation
                                       
                                       'Retardation_Peak_Minimum',    # Weight of lowest peak in speed calculation
                                       
                                       'Retardation_Min_Peak_Mean',   # Mean of min peaks in Retardation
                                       
                                       'Retardation_Min_Peak_SD',     # SD of min peaks in Retardation
                                       
                                       'Centroid Point',               # Centroid point within a segment
                                       
                                       'Centroid Max Distance'         # Max distance from centroid in segment
                                                                       
                                        ])
     #For segment   
      d= pd.read_csv(dir + 'IMEI_Files' +'\\' + f)
      d = d[d['Remove']=="0"].reset_index(drop=True)
      trip = d['TripID'].unique().tolist()
      imei = d['IMEI'][0]
      if len(d) > 1:
          for t in trip:
                     segm = d[d.TripID == t].reset_index(drop=True)
                     seg = d['SegmentID'].unique().tolist()
                     for s in seg:
                       df = segm[segm.SegmentID == s].reset_index(drop=True)
                       if len(df) > 1:
                        Mark = df['TripID'][0]
                        Segment = df['SegmentID'][0]
                        df = df[["IMEI","Date Time","Latitude","Latitude_Direction","Longitude","Longitude_Direction","rdp"]]
                        df['Acceleration'] = 0
                        df['Date Time'] = df['Date Time'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))                       
                        ind = []
                        s =  [i for i in range(1,len(df)-1) if df['rdp'][i] == 1]
                        ind = [s[i] for i in range(0,len(s)) if df['rdp'][s[i]] != df['rdp'][s[i] + 1] or df['rdp'][s[i]] != df['rdp'][s[i] - 1] ]
                        ind.append(len(df))
                        ind.insert(0,0)
                        speed_first = 0 #For speed calculation at first
                        time_first = df['Date Time'][0] - timedelta(0,30) #For time calculation at first(30 second difference at first then last second difference)
                        for index in range(0,len(ind)-1):                
                            low = ind[index] + 1
                            if index != len(ind)-2:
                               high = ind[index + 1] + 1
                            else:
                               high = ind[index + 1]
                            # Calculating params speed and time for acceleration  
                            dists =[haversine((df['Latitude'][i],df['Longitude'][i]),(df['Latitude'][i-1],df['Longitude'][i-1])) for i in range(low,high)]         
                            time =  [((df['Date Time'][i] - df['Date Time'][i-1]).total_seconds() / 3600) for i in range(low,high)]
                            time.insert(0,(df['Date Time'][ind[index]] - time_first).total_seconds() / 3600)
                            speed = [(dists[i] / time[i]) if time[i] > 0  else 0 for i in range(0,len(dists)) ]                                
                            speed.insert(speed_first,0)
                            speed_max  = np.max(speed)
                            speed_min  = np.min(speed)
                            speed_max_index = speed.index(speed_max) + ind[index]
                            speed_min_index = speed.index(speed_min) + ind[index]
                            if(speed_max_index != speed_min_index):
                               df['Acceleration'][ind[index]:ind[index+1]+1] = (speed_max - speed_min) / ((df['Date Time'][speed_max_index] - df['Date Time'][speed_min_index]).total_seconds() / 3600)                       
                            else:
                                df['Acceleration'][ind[index]:ind[index+1]+1] = 0
                            #Taking both the last index of date time and speed for acceleratipn calc
                            speed_first = speed[-1]
                            time_first = df['Date Time'][ind[index]] 
                        df = df[df['rdp'] == 1].reset_index(drop=True)
                       
                        #df = df.drop_duplicates(subset=['Date Time'], keep='first').reset_index(drop=True)           
                        #df['Date Time'] = sorted(df['Date Time']) 
                        #df['Week_Number'] = df['Date Time'].apply(lambda x : x.week)
                        #df["Weekday"] = df["Date Time"].apply(lambda x  : x.weekday_name)
                        #df['Time'] = df['Date Time'].apply(lambda x : datetime.datetime.strftime(x,'%H:%M:%S'))
                        
                        distance = [haversine((df['Latitude'][i-1],df['Longitude'][i-1]),(df['Latitude'][i],df['Longitude'][i])) for i in range(1,len(df)) ] 
                        time = [((df['Date Time'][i] - df['Date Time'][i-1]).total_seconds() / 3600) for i in range(1,len(df))]
                        speed = [(distance[i] / time[i]) if time[i] > 0  else 0 for i in range(0,len(distance))] 
                        speed.insert(0,0)               
                        acceleration = df['Acceleration'].tolist()
                        acceleration.append((0 - speed[len(speed) - 1]) / ((df['Date Time'][len(df) - 1] + datetime.timedelta(0,30)) - df['Date Time'][len(df) - 2]).total_seconds() / 3600)
                        convpoints = list(pl.frange(0,202,0.1)) if len(acceleration) > 50 else list(range(0,202,2))
                        
                        if any(a>0 for a in speed):  
                            conv_speed = convolution(speed,[x+1 for x in np.ones(len(speed))],1,convpoints)
                            
                            speed_peak = convolution_peaks(conv_speed)
                            
                        if any(a > 0 for a in acceleration):  
                            conv_acc = convolution([a for a in acceleration if a > 0],np.ones(len([a for a in acceleration if a > 0])),1,convpoints)
                            
                            acc_peak = convolution_peaks(conv_acc)
                            
                        if any(a < 0 for a in acceleration):  
                            conv_dcc = convolution([a for a in acceleration if a < 0],np.ones(len([a for a in acceleration if a < 0])),1,convpoints)
            
                            dcc_peak = convolution_peaks(conv_dcc)
                        
                     
                            
                        
                        if all(d == 0.0 for d in distance):
                            cd = 0
                            ct = 0
                        
                        elif 0.0 not in distance:
                            cd = np.sum(distance)
                            ct = np.sum(time)
                        
                        
                        else:
                            #For finding maximum stretch time and distance of vehicle
                            x = [i+1 for i in range(0,len(distance)) if (distance[i] > 0.025 and time[i] > 0.0333)]
                            x.append(0)
                            x.append(len(distance))
                            x.sort(reverse=False)
                            z = ([(x[i] - x[i-1]) for i in range(1,len(x))])
                            y = [z.index(np.max(z)),z.index(np.max(z))+1]
                            cd = haversine((df['Latitude'][x[y[0]]],df['Longitude'][x[y[0]]]),(df['Latitude'][x[y[1]]],df['Longitude'][x[y[1]]]))
                            
                            ct = (df['Date Time'][x[y[1]]] - df['Date Time'][x[y[0]]]).total_seconds() / 3600
                        
                        dft = dft.append({'IMEI': imei ,'Stretch': Mark , 'Segment': Segment ,
                                                   
                                                   'Total Points': len(df) ,            
                                                   
                                                   'Start Hour': df['Date Time'][0].strftime('%H') ,                   
                                                   
                                                   'Start Day': df["Date Time"][0].weekday_name ,                   
                                                   
                                                   'Total Distance': np.sum(distance) ,               
                                                   
                                                   'Total Time': np.sum(time) ,          
                                                   
                                                   'Displacement': haversine((df['Latitude'][len(df) - 1],df['Longitude'][len(df) - 1]),(df['Latitude'][0],df['Longitude'][0])) ,                 
                                                   
                                                   'Speed': np.sum(distance)/ np.sum(time),          
                                                   
                                                   'Speed_SD': np.std([s for s in speed if s!= 0]) if any(v!=0 for v in speed) else 0  ,  # S.D of speed of the vehicle in this stretch where speed > 0
                                                   
                                                   'Speed_Max': np.max(speed) ,        # Maximum Speed of the vehicle in this stretch
                                                                                     
                                                   'Speed_Peak_Total': len(speed_peak) if any(a>0 for a in speed) else 0 ,             # Total number of  Speed peaks of the vehicle in this stretch
                                                   
                                                   'Speed_Peak_Highest': np.max([conv_speed[i] for i in speed_peak]) if any(a>0 for a in speed) else 0,           # Weight of highest peak in speed calculation
                                                                                      
                                                   'Speed_Max_Peak_Mean': np.max([convpoints[i] for i in speed_peak]) if any(a>0 for a in speed) else 0 ,          # Mean of max peaks in speed
                                                   
                                                   'Speed_Max_Peak_SD': peak_std_high(speed_peak,convpoints,conv_speed) if any(a>0 for a in speed) else 0,            # SD of max peaks in speed
                                                   
                                                   'Total Driving Time': np.sum([(df['Date Time'][i] - df['Date Time'][i-1]).total_seconds() / 3600 for i in range(1,len(distance)) if distance[i]!=0]) if ct!= np.sum(time) else ct ,           # Total driving time where distance > 0
                                                   
                                                   'Continous Driving Distance': cd ,       # At a stretch how much driver is driving without breaks
                                                   
                                                   'Continous Driving Time':ct ,
                                                   
                                                   'Acceleration_Max': np.max([a for a in acceleration if a >= 0] if any(a > 0 for a in acceleration) else 0) ,             # Maximum acceleration during the stretch
                                                   
                                                   'Retardation_Max': np.max([a  for a in acceleration if a <= 0] if any(a < 0 for a in acceleration) else 0) ,              # Maximum Retardation during the stretch
                                                   
                                                   'Acceleration_Mean': np.mean([a for a in acceleration if a > 0] if any(a > 0 for a in acceleration) else 0)  ,            # Mean acceleration during the stretch where acceleration != 0
                                                   
                                                   'Retardation_Mean':  np.mean([a for a in acceleration if a < 0] if any(a < 0 for a in acceleration) else 0)  ,             # Mean Retardation during the stretch where acceleration != 0
                                                   
                                                   'Acceleration_SD': np.std([a for a in acceleration if a > 0] if any(a > 0 for a in acceleration) else 0) ,              # SD acceleration during the stretch where acceleration != 0
                                                   
                                                   'Retardation_SD': np.std([a  for a in acceleration if a < 0] if any(a < 0 for a in acceleration) else 0) ,               # SD Retardation during the stretch where acceleration != 0
                                                   
                                                   'Acceleration Peaks': len(acc_peak) + len(dcc_peak) if any(a>0 for a in acceleration) else 0,           # Total peaks of acceleration and deaccelertion
                                                   
                                                   'Total Acceleration Peaks': len(acc_peak) if any(a>0 for a in acceleration) else 0 ,     # Total peaks in accleration and retardation
                                                   
                                                   'Acceleration_Peak_Highest': np.max([conv_acc[i] for i in acc_peak]) if any(a>0 for a in acceleration) else 0,    # Weight of highest peak in acceleration calculation
                                                   
                                                   'Acceleration_Max_Peak_Mean': np.max([convpoints[i] for i in acc_peak]) if any(a>0 for a in acceleration) else 0,   # Mean of max peaks in acceleration
                                                   
                                                   'Acceleration_Max_Peak_SD': peak_std_high(acc_peak,convpoints,conv_acc)if any(a>0 for a in acceleration) else 0 ,     # SD of max peaks in acceleration
                                                   
                                                   'Acceleration_Peak_Minimum': np.min([conv_acc[i] for i in acc_peak])if any(a>0 for a in acceleration) else 0 ,    # Weight of lowest peak in deacceleration calculation
                                                   
                                                   'Acceleration_Min_Peak_Mean': np.min([convpoints[i] for i in acc_peak]) if any(a>0 for a in acceleration) else 0,   # Mean of min peaks in deacceleration
                                                   
                                                   'Acceleration_Min_Peak_SD': peak_std_low(acc_peak,convpoints,conv_acc) if any(a > 0 for a in acceleration) else 0,     # SD of min peaks in deacceleration
                                                   
                                                   'Total Retardation Peaks':len(dcc_peak) if any(a<0 for a in acceleration) else 0,     # Total peaks in Retardation
                                                   
                                                   'Retardation_Peak_Highest':np.max([conv_dcc[i] for i in dcc_peak]) if any(a<0 for a in acceleration) else 0,    # Weight of highest peak in acceleration calculation
                                                   
                                                   'Retardation_Max_Peak_Mean':np.max([convpoints[i] for i in dcc_peak]) if any(a<0 for a in acceleration) else 0,   # Mean of max peaks in Retardation
                                                   
                                                   'Retardation_Max_Peak_SD':peak_std_high(dcc_peak,convpoints,conv_dcc) if any(a<0 for a in acceleration) else 0,     # SD of max peaks in Retardation
                                                   
                                                   'Retardation_Peak_Minimum':np.min([conv_dcc[i] for i in dcc_peak]) if any(a<0 for a in acceleration) else 0,    # Weight of lowest peak in acceleration calculation
                                                   
                                                   'Retardation_Min_Peak_Mean':np.min([convpoints[i] for i in dcc_peak]) if any(a<0 for a in acceleration) else 0,   # Mean of min peaks in Retardation
                                                   
                                                   'Retardation_Min_Peak_SD':peak_std_low(dcc_peak,convpoints,conv_dcc) if any(a<0 for a in acceleration) else 0,     # SD of min peaks in Retardation
                                                   
                                                   'Centroid Point':  KMeans(n_clusters=1).fit(df[['Latitude','Longitude']]).cluster_centers_[0],
                                                   
                                                   'Centroid Max Distance':  haversine((df['Latitude'][np.max(KMeans(n_clusters=1).fit(df[['Latitude','Longitude']]).transform(df[['Latitude','Longitude']])).argmax()],df['Longitude'][np.max(KMeans(n_clusters=1).fit(df[['Latitude','Longitude']]).transform(df[['Latitude','Longitude']])).argmax()]) ,(KMeans(n_clusters=1).fit(df[['Latitude','Longitude']]).cluster_centers_[0][0],KMeans(n_clusters=1).fit(df[['Latitude','Longitude']]).cluster_centers_[0][1]))
                                                   
                                                   },ignore_index = True)
                       #For stretch
          for s in dft.Stretch.unique().tolist():
                                     dfs = dfs.append({'IMEI' : imei ,
                                                              'Stretch' : s,
                                                                  
                                                              'Total Segment': len(dft[dft.Stretch == s]['Segment'].unique()),              
                                                                  
                                                              'Segments in same Geography':geometry(dft[dft.Stretch == s][['Centroid Point','Stretch']].reset_index(drop=True)), 
                                                                  
                                                              'Start Day of Week':dft[dft.Stretch == s].reset_index()['Start Day'][0] ,            
                                                                  
                                                              'Start hour of day':dft[dft.Stretch == s].reset_index()['Start Hour'][0], 
                                                                  
                                                              'Total time of stretch':np.sum(dft[dft.Stretch == s]['Total Time'])},ignore_index = True) 
                        
                #For IMEI
          dfi = dfi.append({'IMEI':imei,
                                              'Total Stretch':len(dfs['Stretch'].unique().tolist()),             
                                              'Total Segment' : len(dft),               
                                              'Segments in same Geography':geometry(dft[dft.IMEI == imei][['Centroid Point','Stretch']].reset_index(drop=True)), 
                                               
                                              'Monday':len(dfs[dfs['Start Day of Week'] == 'Monday']),'Tuesday':len(dfs[dfs['Start Day of Week'] == 'Tuesday']),
                                              'Wednesday':len(dfs[dfs['Start Day of Week'] == 'Wednesday']),'Thursday':len(dfs[dfs['Start Day of Week'] == 'Thursday']),
                                              'Friday':len(dfs[dfs['Start Day of Week'] == 'Monday']),'Saturday':len(dfs[dfs['Start Day of Week'] == 'Saturday']),                 
                                              'Sunday':len(dfs[dfs['Start Day of Week'] == 'Sunday']), 
                                              '0':len(dfs[dfs['Start hour of day'] == '00']),'1':len(dfs[dfs['Start hour of day'] == '01']),'2':len(dfs[dfs['Start hour of day'] == '02']),'3':len(dfs[dfs['Start hour of day'] == '03']),'4':len(dfs[dfs['Start hour of day'] == '04']),
                                              '5':len(dfs[dfs['Start hour of day'] == '05']),'6':len(dfs[dfs['Start hour of day'] == '06']),'7':len(dfs[dfs['Start hour of day'] == '07']),'8':len(dfs[dfs['Start hour of day'] == '08']),
                                              '9':len(dfs[dfs['Start hour of day'] == '09']),'10':len(dfs[dfs['Start hour of day'] == '10']),'11':len(dfs[dfs['Start hour of day'] == '11']),'12':len(dfs[dfs['Start hour of day'] == '12']),
                                              '13':len(dfs[dfs['Start hour of day'] == '13']),'14':len(dfs[dfs['Start hour of day'] == '14']),'15':len(dfs[dfs['Start hour of day'] == '15']),'16':len(dfs[dfs['Start hour of day'] == '16']),
                                              '17':len(dfs[dfs['Start hour of day'] == '17']),'18':len(dfs[dfs['Start hour of day'] == '18']),'19':len(dfs[dfs['Start hour of day'] == '19']),'20':len(dfs[dfs['Start hour of day'] == '20']),
                                              '21':len(dfs[dfs['Start hour of day'] == '21']),'22':len(dfs[dfs['Start hour of day'] == '22']),'23':len(dfs[dfs['Start hour of day'] == '23'])
                                                  
                                               },ignore_index = True) 
                    
          dft = dft.sort_values(['Stretch', 'Segment'], ascending=[True, True]).reset_index(drop=True)
                
            #Appending for each IMEI features
      with open(dir + "Features\\" + "Feature_IMEI" +".csv", 'a') as imeifile:
                              dfi.to_csv(imeifile,index = False,header=False)
                              
                              
                              
      with open(dir +"Features\\" + "Feature_Trip" +".csv", 'a') as imeifile:
                              dfs.to_csv(imeifile,index = False,header=False)
                              
                
                          
      with open(dir +"Features\\" + "Feature_Segment" +".csv", 'a') as imeifile:
                              dft.to_csv(imeifile,index = False,header=False)
                              
      print("Done for file :" + str(imei))                 
                          
'''                     
def peak_finding(x,y):
    
  ZERO = math.pow(10,-10)
  cp = 1
  n = len(x)
  peaks = np.zeros(n)

  if (n < 2 ): 
    return 1,1,x[0],[0],0,x[0],x[0],[]
 
  # Initialize the diff and slope
  dir = 0
  if ((y[1] - y[0]) > ZERO):  ## Going Up
    dir = 1
  elif ((y[0] - y[1]) > ZERO): ## Going Down
    dir = -1

  
  ## Irrespective of the direction it is part of first peak
  
  peaks[1] = 1  
  Mu = []
  xmax = 1
  cp = 1 ## Next peak value 
  for i in range(1,n):
    prevdir = dir 
    # Find Current Direction
    dir = 0
    if (y[i] - y[i-1]) > ZERO:   ## Going Up
      dir = 1
      
    elif (y[i-1] - y[i]) > ZERO:   ## Going Down
      dir = -1 
    
    # Set the Peak number 
    if (prevdir == dir):  ## No Change
      peaks[i] = peaks[i-1]
      if y[xmax] < y[i] : xmax = i
      
    elif ((prevdir == -1) or (dir == 1)): ## New Peak Start
          Mu.append(xmax)
          xmax = i
          cp = cp + 1
          peaks[i] = cp
          
    else:  ## Continue with previous peak
       peaks[i] = peaks[i-1]
       if (y[xmax] < y[i]): xmax = i
    
  
  Mu.append(xmax)
  Pks = Mu
  
  ## Find Parameters
  N = [x + y for x, y in zip(y, peaks)]
  Pks = [x[i] for i in Pks]
  Mu = [k + m for k,m in zip([i*j for i,j in zip(x,y)], peaks)]
  Mu = [x / y for x,y in zip(Mu,N)]
  Counts = tapply(peaks, peaks,length,simplify = T)
  Mus = rep(Mu,Counts)
  Sigma = tapply(y * (x - Mus)^2, peaks,sum,simplify = T)
  Sigma = Sigma / N
  Sigma = Sigma ^ 0.5
  LBound = tapply(x, peaks,min,simplify = T)
  RBound = tapply(x, peaks,max,simplify = T)
  Inflexion = c()
  
  ## Combine Peaks with N < 5%
  #  Peaks = sort(unique(peaks))
  #  NP = N / sum(N)
  #  for (i in 2:length(NP)) {
  #    if ((NP[i-1] < 0.05) && (NP[i]< 0.05)) {
  #      peaks[peaks==Peaks[i]] = Peaks[i-1]
  #      Peaks[i] = Peaks[i-1]
  #    }
  #  }
  #  N = tapply(y , peaks,sum,simplify = T)
  #  Mu = tapply(y * x , peaks,sum,simplify = T)
  #  Mu = Mu / N
  #  Counts = tapply(peaks, peaks,length,simplify = T)
  #  Mus = rep(Mu,Counts)
  #  Sigma = tapply(y * (x - Mus)^2, peaks,sum,simplify = T)
  #  Sigma = Sigma / N
  #  Sigma = Sigma ^ 0.5
  #  LBound = tapply(x, peaks,min,simplify = T)
  #  RBound = tapply(x, peaks,max,simplify = T)
  
  if (srt) {
    ## Now sort it
    pos = which(N == max(N))[1]
    i = 1
    while ((length(pos) > 0) & (!is.na(pos))) {
      if (pos != i) {
        N[pos] = N[pos] + N[i]
        N[i] = N[pos] - N[i]
        N[pos] = N[pos] - N[i]
        Pks[pos] = Pks[pos] + Pks[i]
        Pks[i] = Pks[pos] - Pks[i]
        Pks[pos] = Pks[pos] - Pks[i]
        Mu[pos] = Mu[pos] + Mu[i]
        Mu[i] = Mu[pos] - Mu[i]
        Mu[pos] = Mu[pos] - Mu[i]
        Sigma[pos] = Sigma[pos] + Sigma[i]
        Sigma[i] = Sigma[pos] - Sigma[i]
        Sigma[pos] = Sigma[pos] - Sigma[i]
        LBound[pos] = LBound[pos] + LBound[i]
        LBound[i] = LBound[pos] - LBound[i]
        LBound[pos] = LBound[pos] - LBound[i]
        RBound[pos] = RBound[pos] + RBound[i]
        RBound[i] = RBound[pos] - RBound[i]
        RBound[pos] = RBound[pos] - RBound[i]
      }
      i = i + 1
      pos = which(N[i:length(N)] == max(N[i:length(N)]))[1] + i -1
    }
  }
  
  return(list(PeakCount = length(N),N=N,Peaks=Pks,Mu=Mu,Sigma=Sigma,
              LBound=LBound,RBound=RBound,Inflexion = Inflexion))
}
                      
                      
                      
                      
                  
                  

def coordinate(x):
    
    central_coordinate = (21.1458, 79.0882)
    if x[0] < central_coordinate[0]:
         X = -haversine((central_coordinate[0],0),(x[0],0))
    else:
         X = haversine((central_coordinate[0],0),(x[0],0))
         
    if x[1] < central_coordinate[1]:
         Y = -haversine((0,central_coordinate[1]),(0,x[1]))
    else:
         Y = haversine((0,central_coordinate[1]),(0,x[1]))
    
    return (X/100,Y/100)
'''        