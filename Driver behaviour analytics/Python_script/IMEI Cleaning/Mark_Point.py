import datetime
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

r = lambda f,p: f - f % p # For rounding decimals
 
def mark_point(dir,imeifile,tau,static_dist,bigtau,speed_trip):
      
    """
    Marks Start and Stop point of the points 
    
    """
    
    df = pd.read_csv(dir + "Data\\" + imeifile, converters={'Latitude': str,'Longitude':str})
    df['Latitude'] = [r(pd.to_numeric(df['Latitude'][i]),0.0000001) for i in range(0,len(df))]
    df['Longitude'] = [r(pd.to_numeric(df['Longitude'][i]),0.0000001) for i in range(0,len(df))]
    df['Date Time'] = df['Date Time'].apply(lambda x : datetime.datetime.strptime(x,'%Y%m%d %H:%M:%S'))
    df = df.sort_values(by = ['Date Time']).reset_index(drop=True)
    df['TripID'] = 0
    df['SegmentID'] = 0
    df['Remove'] = "Static Points" #Removing pints which are stationary i.e due to GPS inaccuracy
    df['Dist'] = 1  
    df['Timediff'] = 1
    dfi = pd.DataFrame()
    #Calculate distnace of those points
    df['Dist'][1:len(df)] = [haversine((df['Latitude'][x-1],df['Longitude'][x-1]),(df['Latitude'][x],df['Longitude'][x])) for x in df['Dist'][1:].index.tolist()]
    df['Remove']= np.where(df['Dist'] > static_dist,"0","Static Points")
    # Calculate Static points until they are moving i.e distnace > static_dist   
    while any(df['Remove'][i] == "Static Points" for i in range(1,len(df))):
            dfi = dfi.append(df[df['Remove'] != "0"]).reset_index(drop=True)
            df = df[df['Remove'] == "0"].reset_index(drop=True)
            df['Dist'] = 1
            df['Dist'][1:len(df)] = [haversine((df['Latitude'][x-1],df['Longitude'][x-1]),(df['Latitude'][x],df['Longitude'][x])) for x in df['Dist'][1:].index.tolist()]
            df['Remove']= np.where(df['Dist'] < static_dist,"Static Points","0")
      
    
    if len(dfi) > 1:
        dfi = dfi.sort_values(by = ['Date Time']).reset_index(drop=True)
        dfi['Dist'] = 1  
        dfi['Timediff'] = 0
        dfi['Dist'][1:len(dfi)] = [haversine((dfi['Latitude'][x-1],dfi['Longitude'][x-1]),(dfi['Latitude'][x],dfi['Longitude'][x])) for x in dfi['Dist'][1:].index.tolist()]
        dfi['Timediff'][1:len(dfi)] = [(dfi['Date Time'][i] - dfi['Date Time'][i-1]).total_seconds()/3600 for i in range(1,len(dfi))]    
        dfi['Remove'] = np.where((dfi.Dist == 0) & (dfi.Timediff == 0),"Duplicate Points","Static Points")
        
    #Mark them with differet trip label of data      
    if len(df) > 1:
      df['Timediff'] = 0
      df['Timediff'][1:len(df)] = [(df['Date Time'][i] - df['Date Time'][i-1]).total_seconds()/3600 for i in range(1,len(df))]  
      for i in range(0,len(df)):
          if df.Timediff[i] > bigtau:
              df.TripID[i] = 1
              continue
          else:
              if df.Timediff[i] > tau:
                  if haversine((df['Latitude'][i],df['Longitude'][i]),(df['Latitude'][i-1],df['Longitude'][i-1])) / df.Timediff[i] < speed_trip:
                     df.TripID[i] = 1
                  
                  
      
      
      df['TripID'][0:(len(df)-1)] = [np.sum(df['TripID'][0:(i+1)]) for i in range(0,len(df)-1)] 
      df['TripID'][len(df)-1] = df['TripID'][len(df)-1] + df['TripID'][len(df)-2]
   
  
    dfi = dfi.append(df).reset_index(drop=True)
    df = dfi[["VendorID","IMEI","VehicleNumber","Date Time","Latitude","Latitude_Direction","Longitude","Longitude_Direction","Speed","TripID","SegmentID","Remove","PacketType","PacketStatus","GPSFix","Heading","SatelliteNumber","Altitude","PDOP","Operator","Ignition","MainPowerStatus","EmergencyStatus","TamperAlert","GSMSignal","MCC","MNC","LAC","CellID","DigitalInput","DigitalOutput","FrameNumber"]]
    
    return df
    #df.to_csv(dir + "\" +"Mark_Point\" + "Label_" + imeifile[0:15]+ ".csv",date_format='%Y-%m-%d %H:%M:%S')




















    
''' 
      mark = 0
      i = 1
      j = 0
      while i < len(df):
          if i == len(df) - 1:
              if ((df['Date Time'].loc[i] - df['Date Time'].loc[i - 1]).total_seconds()) / 3600 > tau:
                   df['TripID'].loc[j:i] = mark
                   df['TripID'].loc[i] = -1
              else:
                  df['TripID'].loc[j:len(df)] = mark
              break
             
          if ((df['Date Time'].loc[i] - df['Date Time'].loc[i - 1]).total_seconds()) / 3600 > tau:
                if i == j + 1:
                    df['TripID'].loc[j] = -1
                    j = i
                    i = i + 1
                else:
                    df['TripID'].loc[j:i] = mark
                    df['TripID'].loc[i] = -1
                    j = i
                    i = i + 1
                    mark = mark + 1
          else:
                i = i + 1
'''
      