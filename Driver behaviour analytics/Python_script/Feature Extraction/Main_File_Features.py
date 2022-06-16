# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:58:53 2019

@author: saura
"""

import os
import datetime
import warnings
import pandas as pd
warnings.simplefilter('ignore', UserWarning)
import numpy as np
import multiprocessing as mp
import csv

from Data_Features import haversine
from Data_Features import convolution
from Data_Features import convolution_peaks
from Data_Features import peak_std_high
from Data_Features import peak_std_low
from Data_Features import geometry
from Data_Features import feature

if __name__ == '__main__':   
        
    dir = "C:\\Users\\saura\\Documents\\Clean Records\\"
    os.chdir(dir + "IMEI_Files\\")  #Changing current working directory
    directory = os.getcwd()  #Getting current working directory of the CLI
    
    
    with open(dir + "Features\\" + "Feature_IMEI" + ".csv", 'a') as f:
      writer = csv.writer(f)
      writer.writerow(['IMEI','Total Stretch','Total Segment','Segments in same Geography',              # Total
                                       
                                      'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday', # Total Stretches in this Weekday
                                      #Total Stretches in this hour
                                      '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23'
                                          
                                       ])
    with open(dir + "Features\\" +"Feature_Trip" + ".csv", 'a') as f:
      writer = csv.writer(f)
      writer.writerow(['IMEI',
                                      
                                      'Stretch',                     # Stretch number
                                          
                                      'Total Segment',               # Total Segments in each stretch
                                          
                                      'Segments in same Geography',  # Segments coming in same geographical area
                                          
                                      'Start Day of Week',           # Start day of week of stretch    
                                          
                                      'Start hour of day',           # Start hour of day of stretch 
                                          
                                      'Total time of stretch'])
    
        
    with open(dir +"Features\\" + "Feature_Segment" + ".csv", 'a') as f:
      writer = csv.writer(f)
      writer.writerow(['IMEI','Stretch','Segment', # For each segment of data in a stretch
                                       
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
                                    
    pool = mp.Pool(processes=4)                            
    print("Operation Feature Extraction started  %s"%datetime.datetime.now())
    files = next(os.walk(directory))[2]
    var = [pool.apply_async(feature,args=[f,dir]) for f in files]
    print(var)
    pool.close()
    pool.join()

