# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:32:31 2019

@author: saura
"""


import datetime
import pandas as pd
import math
import numpy as np
import os
#import multiprocessing as mp

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

r = lambda f,p: f - f % p # For rounding decimals

dir = "C:\\Users\\saura\\Documents\\Clean Records\\"
os.chdir(dir + "IMEI_Files")  #Changing current working directory
directory = os.getcwd()  #Getting current working directory of the CLI
files = next(os.walk(directory))[2] #Creates a list of all files

pvtpoints = 0
usedpvtpoints = 0
imei = 0
tripcount = []
tripsegment = []
tripdistance = []
tripday = []
triphour = []
triptime = []


for f in files:
    
    df = pd.read_csv(dir + "\\" + "IMEI_Files\\" + f , converters={'Latitude': str,'Longitude':str})
    df = df[['IMEI','Date Time','Latitude','Longitude','TripID','SegmentID','Remove']]
    
    pvtpoints = pvtpoints + len(df)
    imei = imei + 1
    
    df['Remove'] = df['Remove'].astype(str)
    df = df[df['Remove'] == "0"].reset_index(drop=True)
    
    df['Latitude'] = [r(pd.to_numeric(df['Latitude'][i]),0.0000001) for i in range(0,len(df))]
    df['Longitude'] = [r(pd.to_numeric(df['Longitude'][i]),0.0000001) for i in range(0,len(df))]
    df['Date Time'] = df['Date Time'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    
    tripcount.append(len(df['TripID'].unique().tolist()))

    #k = 0
    for t in df['TripID'].unique().tolist():
        dft = df[df['TripID'] == t].reset_index(drop=True)
        if len(dft) > 1:
                            
            usedpvtpoints = usedpvtpoints + len(dft)

            #k = k + 1
            tripdistance.append(haversine((dft['Latitude'][0],dft['Longitude'][0]),(dft['Latitude'][len(dft) - 1],dft['Longitude'][len(dft) - 1])))
            #dist = tuple(tripdistance)
            
            tripday.append(dft['Date Time'][0].strftime("%A"))
            
            triphour.append(dft['Date Time'][0].strftime("%H"))
            
            triptime.append(((dft['Date Time'].loc[len(dft)- 1] - dft['Date Time'].loc[0]).total_seconds()) / 60)
            
            tripsegment.append(len(dft['SegmentID'].unique().tolist()))
            
            
        else:
            usedpvtpoints = usedpvtpoints + 1

            tripdistance.append(0)
            
            tripday.append(dft['Date Time'][0].strftime("%A"))
            
            triphour.append(dft['Date Time'][0].strftime("%H"))
            
            triptime.append(0)
            
            tripsegment.append(len(dft['SegmentID'].unique().tolist()))
            
            
    
    if imei % 1000 == 0 :       
      print("Done for 1000 IMEI files")
            
# Data frame for all summary screen     

print("Done for all files now generating IMEI summary")


tripd = pd.DataFrame(data = {'Days'  :["Monday","Tuesday","Wednesday","Thursay","Friday","Saturday","Sunday"] , 
        'Count' :    [len([i for i in tripday if i == "Monday"])
                    ,len([i for i in tripday if i == "Tuesday"])
                    ,len([i for i in tripday if i == "Wednesday"])
                    ,len([i for i in tripday if i == "Thursday"])
                    ,len([i for i in tripday if i == "Friday"])
                    ,len([i for i in tripday if i == "Saturday"])
                    ,len([i for i in tripday if i == "Sunday"])]})


triph = pd.DataFrame(data = {'Hour' : ['12A.M to 4A.M','4A.M to 8A.M','8A.M to 10A.M','10A.M to 12P.M','12P.M to 2P.M','2P.M to 6P.M','6P.M to 9P.M','9P.M to 12P.M'],
        'Count' :   [len([i for i in triphour if int(i) in range(0,4)])
                   ,len([i for i in triphour if int(i) in range(4,8)])
                   ,len([i for i in triphour if int(i) in range(8,10)])
                   ,len([i for i in triphour if int(i) in range(10,12)])
                   ,len([i for i in triphour if int(i) in range(12,14)])
                   ,len([i for i in triphour if int(i) in range(14,18)])
                   ,len([i for i in triphour if int(i) in range(18,21)])
                   ,len([i for i in triphour if int(i) in range(21,25)])]})



tript = pd.DataFrame(data =  {'Time' : ['Less Than 15 min','15 min - 30 min','30 min to 1 hour','1 hour to 1.5 hour','1.5 hour to 2 hour','2 hour to 3 hour','3 hour to 6 hour','6 hour to 10 hour','More than 10 hour'],
        'Count' : [len([i for i in triptime if np.round(i) in range(0,16)])
                  ,len([i for i in triptime if np.round(i) in range(16,31)])
                  ,len([i for i in triptime if np.round(i) in range(31,61)])
                  ,len([i for i in triptime if np.round(i) in range(61,91)])
                  ,len([i for i in triptime if np.round(i) in range(91,121)])
                  ,len([i for i in triptime if np.round(i) in range(121,181)])
                  ,len([i for i in triptime if np.round(i) in range(181,241)])
                  ,len([i for i in triptime if np.round(i) in range(241,601)])
                  ,len([i for i in triptime if np.round(i) > 600])]})


   

tripc = pd.DataFrame(data =  {'Trip': ['1','2','3','4','5','6','7 to 11','11 to 16','16 to 21','21 to 26','26 to 31','31 to 50','More than 50'],
        'Count' : [len([i for i in tripcount if i == 1])
                   ,len([i for i in tripcount if i == 2])
                   ,len([i for i in tripcount if i == 3])
                   ,len([i for i in tripcount if i == 4])
                   ,len([i for i in tripcount if i == 5])
                   ,len([i for i in tripcount if i == 6])
                   ,len([i for i in tripcount if i in range(7,11)])
                   ,len([i for i in tripcount if i in range(11,16)])
                   ,len([i for i in tripcount if i in range(16,21)])
                   ,len([i for i in tripcount if i in range(21,26)])
                   ,len([i for i in tripcount if i in range(26,31)])
                   ,len([i for i in tripcount if i in range(31,50)])
                   ,len([i for i in tripcount if i > 50])]})


tripdis = pd.DataFrame(data =  {'Distance': ['Less than 0.5','0.5 to 1','1 to 5','6 to 10','11 to 20','20 to 40','40 to 60','More than 60'],
        'Count' : [len([i for i in tripdistance if np.round(i) < 0.5])
                   ,len([i for i in tripdistance if np.round(i) in np.arange(0.5,1,0.1)])
                   ,len([i for i in tripdistance if np.round(i) in range(1,6)])
                   ,len([i for i in tripdistance if np.round(i) in range(6,11)])
                   ,len([i for i in tripdistance if np.round(i) in range(11,21)])
                   ,len([i for i in tripdistance if np.round(i) in range(21,41)])
                   ,len([i for i in tripdistance if np.round(i) in range(41,61)])
                   ,len([i for i in tripdistance if np.round(i) > 60])]})


segment =   pd.DataFrame(data =  {'Segment': ['1','2','3','4','5','6','7','Equal to more than 8'],
        'Count Trips' : [len([i for i in tripsegment if i  == 1])
                   ,len([i for i in tripsegment if i == 2])
                   ,len([i for i in tripsegment if i == 3])
                   ,len([i for i in tripsegment if i == 4])
                   ,len([i for i in tripsegment if i == 5])
                   ,len([i for i in tripsegment if i == 6])
                   ,len([i for i in tripsegment if i == 7])
                   ,len([i for i in tripsegment if i >= 8])]})

data_description =   pd.DataFrame(data =  {'Fields': ['Start Date','End Date','Total IMEI','Total PVT Points','Used PVT Points','Rejected PVT Points'],
        'Description' : ['30 - June - 2018'
                   ,'31 - July - 2018'
                   ,imei
                   ,pvtpoints
                   ,usedpvtpoints
                   ,pvtpoints - usedpvtpoints]})
   

# Writing to csv files
tripd.to_csv(dir + "IMEI_Description\\" + "Days_Description" + ".csv")
triph.to_csv(dir + "IMEI_Description\\" +"Hour_Description" + ".csv")
tript.to_csv(dir + "IMEI_Description\\" +"Time_Description" + ".csv")
tripdis.to_csv(dir + "IMEI_Description\\" +"Distance_Description" + ".csv")
segment.to_csv(dir + "IMEI_Description\\" +"Segment_Description" + ".csv")
tripc.to_csv(dir + "IMEI_Description\\" +"Trip_Description" + ".csv")
data_description.to_csv(dir + "IMEI_Description\\" +"IMEI_Description" + ".csv")