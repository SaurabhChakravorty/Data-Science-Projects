# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:10:36 2019

@author: saura
"""

import pandas as pd
import math
import numpy as np


def haversine(coord1, coord2):
    """
    Haversine distance in meters for two (lat, lon) coordinates
    
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


def check_segments(curindex,previndex,dft,speed_ref1,speed_ref2,dist_ref,static_dist):
    
    """
    Checking The criterion for seperating segments returns values"

    """
    
    
    if dft['Date Time'][curindex] == dft['Date Time'][previndex]:
        dist = haversine((dft['Latitude'][curindex],dft['Longitude'][curindex]),(dft['Latitude'][previndex],dft['Longitude'][previndex]))    
        if dist < static_dist:
           return "S"
        else:
           return "D"
           
    
    else:
       dist = haversine((dft['Latitude'][curindex],dft['Longitude'][curindex]),(dft['Latitude'][previndex],dft['Longitude'][previndex]))    
       time = (dft['Date Time'].loc[curindex] - dft['Date Time'].loc[previndex]).total_seconds() / 3600
       speed = dist / time
       if speed > speed_ref1:
           return "D"
       
       elif speed > speed_ref2 and speed < speed_ref1:
           if dist < dist_ref:
               return "S"
           else: 
               return "D"
       else:
           return "S"



      
def segments(dfi,speed_ref1,speed_ref2,dist_ref,static_dist):
    
   """
   For finding different paths for vehicle movement
     
   """
   dff = pd.DataFrame()
   #For getting trip ID's of vehicle
    
   df = dfi[dfi.Remove == "0"].reset_index(drop=True)
   
   if len(df) == 0:
       return dfi
   
       
   list = df['TripID'].unique()

   # For each segemt of data
   for t in list:
        
        #print("Doing segement %d"%t)
        # For each trip of data
        dft = df[df['TripID'] == t].reset_index(drop=True)
        dft['Label'] = "S"
        dft['Label'][0] = "D"
        dft['Label'][1:len(dft)] = pd.Series(range(1,len(dft))).apply(lambda x :check_segments(x,x-1,dft,speed_ref1,speed_ref2,dist_ref,static_dist) )
        # Where there are different points mismatch       
        index = [i for i in range(0,len(dft)) if dft['Label'][i] == "D"]

        if len(index) > 1:
           for i in index:
               if i == 0:
                   dft['SegmentID'][i:] = 0 #if index is first
                   continue
               
               if i == 1:
                   dft['SegmentID'][i:] = np.max(dft['SegmentID']) + 1 #If index is second we cannot compare with last as there is no value left new segment
                   continue
               
               else:
                   # list for new segment
                   list = dft['SegmentID'][:i-1].unique().tolist()
                   # indices for new segment 
                   indices = [dft[:i-1][dft[:i-1]['SegmentID'] == l][-1:].index[0] for l in list] #Getting index of last mak point previous to the current point only
                   indices = sorted(indices,reverse=True)
                   for j in indices:
                       check = check_segments(i,j,dft,speed_ref1,speed_ref2,dist_ref,static_dist)
                       # Check for values and accordingly update SegmentID
                       if check == "S":
                          dft['SegmentID'][i:] = dft['SegmentID'][j]
                          break
                       else:
                          if j == indices[-1]:
                            dft['SegmentID'][i:] = np.max(dft['SegmentID']) + 1
                
            
        else:
              #If only new segment (most of the data)
              dft['SegmentID'] = 0
            
           
        dft['Dist'] = 1  
        dft['Timediff'] = 1
        dft['Dist'][1:len(dft)] = [haversine((dft['Latitude'][x-1],dft['Longitude'][x-1]),(dft['Latitude'][x],dft['Longitude'][x])) for x in dft['Dist'][1:].index.tolist()]
        dft['Timediff'][1:len(dft)] = [(dft['Date Time'][i] - dft['Date Time'][i-1]).total_seconds() for i in range(1,len(dft))]    
        dft['Remove'] = np.where((dft.Dist < static_dist) & (dft.Timediff != 0),"Static Points",dft['Remove'])
        # Final output data
        dff = dff.append(dft).reset_index(drop=True)
   dff = dff.sort_values(['TripID','SegmentID','Date Time'], ascending=[True,True,True]).reset_index(drop=True)
   df = dff[["VendorID","IMEI","VehicleNumber","Date Time","Latitude","Latitude_Direction","Longitude","Longitude_Direction","Speed","TripID","SegmentID","Remove","PacketType","PacketStatus","GPSFix","Heading","SatelliteNumber","Altitude","PDOP","Operator","Ignition","MainPowerStatus","EmergencyStatus","TamperAlert","GSMSignal","MCC","MNC","LAC","CellID","DigitalInput","DigitalOutput","FrameNumber"]]       
   df = df.append(dfi[dfi.Remove!="0"]).reset_index(drop=True)
    
   return df                  
  
