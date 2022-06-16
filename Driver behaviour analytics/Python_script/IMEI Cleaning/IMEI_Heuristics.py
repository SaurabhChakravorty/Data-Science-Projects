# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:54:43 2019

@author: saura
"""
import datetime
import pandas as pd
import math
import numpy as np
import os
import multiprocessing as mp

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
    return d 

r = lambda f,p: f - f % p # For rounding decimals
 


def Heuristics(f,dir):
        dfd = pd.DataFrame()
        dft = pd.DataFrame()
        dfsp = pd.DataFrame()
        
        df_dist = pd.DataFrame(columns = ['IMEI','Distance','Dist_freq'])
        df_time = pd.DataFrame(columns = ['IMEI','Time','Time_freq'])
        df_speed = pd.DataFrame(columns = ['IMEI','Speed','Speed_freq'])
            
        #for f in files:
        df = pd.read_csv(dir + "\\" + "Data\\" + f , converters={'Latitude': str,'Longitude':str})
        if len(df) > 1:
           df = df[['IMEI','Date Time','Latitude','Longitude']]
           #df = df[df['Remove'] == "0"].reset_index(drop=True)
           df['Latitude'] = [r(pd.to_numeric(df['Latitude'][i]),0.0000001) for i in range(0,len(df))]
           df['Longitude'] = [r(pd.to_numeric(df['Longitude'][i]),0.0000001) for i in range(0,len(df))]
           df['Date Time'] = df['Date Time'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
           df = df.sort_values(by = ['Date Time','Latitude','Longitude']).reset_index(drop=True)
    
         
           dist = [math.ceil(haversine((df['Latitude'][x-1],df['Longitude'][x-1]),(df['Latitude'][x],df['Longitude'][x]))) for x in range(1,len(df))]
           time = [math.ceil((df['Date Time'].loc[i] - df['Date Time'].loc[i - 1]).total_seconds()) for i in range(1,len(df))]
           speed = [math.ceil(dist[i] / time[i]) if time[i] > 0 else 100000 for i in range(0,len(time)) ]
        
           d = np.unique(dist, return_counts=True)
           t = np.unique(time, return_counts=True)
           s = np.unique(speed, return_counts=True)
                            
           df_dist['Distance'] = d[0]
           df_dist['Dist_freq'] = d[1]
           df_dist['IMEI'][0:len(d[0])] = df['IMEI'][0]
          
           df_time['Time'] = t[0]
           df_time['Time_freq'] = t[1]
           df_time['IMEI'][0:len(t[0])] = df['IMEI'][0]
           
            
           df_speed['Speed'] = s[0]
           df_speed['Speed_freq'] = s[1]
           df_speed['IMEI'][0:len(s[0])] = df['IMEI'][0]
          
        
           dfd = dfd.append(df_dist).reset_index(drop=True)
           dft = dft.append(df_time).reset_index(drop=True)
           dfsp = dfsp.append(df_speed).reset_index(drop=True)
                          
                            
           if len(dfd) > 10000: 
                with open(dir +"IMEI_Heuristics\\" + "Distance_Heuristics" +".csv", 'a',encoding='utf-8') as imeifile:
                     dfd.to_csv(imeifile,index = False,header=False)
                with open(dir +"IMEI_Heuristics\\" + "Time_Heuristics" +".csv", 'a',encoding='utf-8') as imeifile:
                     dft.to_csv(imeifile,index = False,header=False)  
                with open(dir +"IMEI_Heuristics\\" + "Speed_Heuristics" +".csv", 'a',encoding='utf-8') as imeifile:
                     dfsp.to_csv(imeifile,index = False,header=False)  
                
                     dfd = pd.DataFrame()
                     dft = pd.DataFrame()
                     dfsp = pd.DataFrame()
                     df_dist = pd.DataFrame(columns = ['IMEI','Distance','Dist_freq'])
                     df_time = pd.DataFrame(columns = ['IMEI','Time','Time_freq'])
                     df_speed = pd.DataFrame(columns = ['IMEI','Speed','Speed_freq'])
                            
                     print("Done many for lines 10000 check")
       #print("Done for file %s"%f)
       #files.remove(f)
    
    
        
        with open(dir + "IMEI_Heuristics\\" +  "Distance_Heuristics" +".csv", 'a',encoding='utf-8') as imeifile:
                                     dfd.to_csv(imeifile,index = False,header=False)
                                     
        with open(dir + "IMEI_Heuristics\\"  + "Time_Heuristics" +".csv", 'a',encoding='utf-8') as imeifile:
                                     dft.to_csv(imeifile,index = False,header=False)  
                                     
        with open(dir +"IMEI_Heuristics\\" + "Speed_Heuristics" +".csv", 'a',encoding='utf-8') as imeifile:
                                     dfsp.to_csv(imeifile,index = False,header=False)
                

if __name__ == '__main__':   
 dir = "C:\\Users\\saura\\Documents\\Clean Records\\"
 os.chdir(dir + "Data")  #Changing current working directory
 directory = os.getcwd()  #Getting current working directory of the CLI
 files = next(os.walk(directory))[2] #Creates a list of all files
 pool = mp.Pool(processes=4)
 #Values to be paased for processing epsilon,Zeta,Tau,Zoom_ref
 print("Operation IMEI Clean up started %s"%datetime.datetime.now())
 var = [pool.apply_async(Heuristics,args=[x,dir]) for x in files]
 print(var)
 pool.close()
 pool.join()
 
 
 