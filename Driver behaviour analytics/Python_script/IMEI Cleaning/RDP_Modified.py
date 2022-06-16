# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:05:35 2019

@author: saura
"""
import pandas as pd
import numpy as np
import math
import csv

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

def checksegmentpoints(lat,long,time,cindex,epsilon,zeta):
    """
    for segmenting points
    """
    startIndex = cindex-2
    lastIndex = cindex
    index = cindex - 1
    #For calculating distance of all points from start and last index
    y = haversine((lat[startIndex],long[startIndex]),(lat[lastIndex],long[lastIndex]))
    h1 = haversine((lat[startIndex],long[startIndex]),(lat[index],long[index]))
    h2 = haversine((lat[index],long[index]),(lat[lastIndex],long[lastIndex]))
    if (y == 0):
        return 1
    y1 = ((np.power(h1,2)) - (np.power(h2,2)) + (np.power(y,2))) / (2 * y)
    distance = np.sqrt(abs(np.power(haversine((lat[startIndex],long[startIndex]),(lat[index],long[index])),2) - np.power(y1,2)))
    
    if (distance > epsilon): return 1
        
    distance_speed = haversine((lat[startIndex],long[startIndex]),(lat[index],long[index]))
    if(time[index] > time[startIndex]):
        distance_speed1 = distance_speed / ((time[index] - time[startIndex]).total_seconds() / float(3600))
    else:
        if (distance_speed > 0):
            distance_speed1 = 1000
        else:
            distance_speed1 = 0
            
    distance_speed = haversine((lat[index],long[index]),(lat[lastIndex],long[lastIndex]))
    if(time[index] < time[lastIndex]):
        distance_speed2 = distance_speed / ((time[lastIndex] - time[index]).total_seconds() / float(3600))
    else:
        if (distance_speed > 0):
            distance_speed2 = 1000
        else:
            distance_speed2 = 0
       
    if (abs(distance_speed1 - distance_speed2) < zeta): return 0   
    return 1


def distance_speed(lat,long,time,index):
    distance_speed = haversine((lat[index],long[index]),(lat[index - 1],long[index - 1]))
    if(time[index] > time[index-1]):
        distance_speed = distance_speed / ((time[index] - time[index-1]).total_seconds() / float(3600))
    else:
        if (distance_speed > 0):
            distance_speed = 1000
        else:
            distance_speed = 0
    #print(distance,distance_speed)
    return distance_speed

    

    
def call_rdp(dfi,dir,imeifile,epsilon,zeta):
    """
    For calling Ramer-Douglas-Peucker algorithm

    """
   
    dfi['rdp'] = -1
    dfc = dfi[dfi.Remove == "0"].reset_index(drop=True)
    if len(dfc) == 0:
       dfi['TripID'][0] = 0
       dfc = dfi[dfi.Remove == "0"].reset_index(drop=True)
    dfc['Remove'] = 0
    dfc['rdp'] = 1
    trip = dfc['TripID'].unique()
    df_final = pd.DataFrame()
    s = 0
    maxsegment = 0
    maxtime = 0
    totaltime = 0
    smallsegment = 0
    for i in trip:
        dft = dfc[dfc.TripID == i].reset_index(drop=True)
        segment = dft['SegmentID'].unique()
        s = s + len(segment)
        for j in segment:
                df = dft[dft['SegmentID'] == j].reset_index(drop=True)
                maxindex = len(df)
                if (maxindex > 2):
                    time = df['Date Time']
                    lat = df['Latitude']
                    long = df['Longitude'] #lat long and time defined for all points
                    df.rdp[1:(maxindex-1)] = pd.Series(range(2,maxindex)).apply(lambda x: checksegmentpoints(lat,long,time,x,epsilon,zeta))
                    df.Remove = np.where(df.rdp==0,"RDP Remainder",df.Remove)
                t = (df['Date Time'][len(df)-1] -  df['Date Time'][0]).total_seconds() / 3600
                if t >= maxtime:
                    maxtime = t
                    Tripmax = i
                    Segmentmax = j
                totaltime = totaltime + t
                if(len(df) > maxsegment):
                    maxsegment = len(df[df['rdp'] == 1])
                if(len(df[df.rdp==1])) <= 3:
                    smallsegment = smallsegment +  1
                df_final = df_final.append(df).reset_index(drop=True)  


    df_final = df_final[["IMEI","Date Time","Latitude","Latitude_Direction","Longitude","Longitude_Direction","Speed","TripID","SegmentID","Remove","VendorID","VehicleNumber","rdp","PacketType","PacketStatus","GPSFix","Heading","SatelliteNumber","Altitude","PDOP","Operator","Ignition","MainPowerStatus","EmergencyStatus","TamperAlert","GSMSignal","MCC","MNC","LAC","CellID","DigitalInput","DigitalOutput","FrameNumber"]]
    #df_final.to_csv(dir + "IMEI_Files\\" + imeifile[0:15]  + ".csv")
    
    #df = df_final[df_final['rdp'] != 0].reset_index(drop=True)
    df = df_final.append(dfi[(dfi.Remove!="0")]).reset_index(drop=True)
    df.to_csv(dir + "IMEI_Files\\" + imeifile[0:15]  + ".csv")
    
    with open(dir + 'Clean_Record_Summary.csv', mode='a',newline='') as recordfile:
        imei_writer = csv.writer(recordfile, delimiter=',')
        imei_writer.writerow(["'" + imeifile[0:15],len(df),len(df[df['Remove'] == 'Duplicate Points']),len(df[df.Remove == "Static Points"]) + len(df[df.TripID == -1]),len(trip),s,len(df[df.rdp == 0]),maxsegment,maxtime,totaltime,Tripmax,Segmentmax,smallsegment])


