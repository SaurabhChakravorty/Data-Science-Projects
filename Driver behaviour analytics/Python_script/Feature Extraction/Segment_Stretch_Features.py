# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:59:55 2019

@author: saura
"""

"""
Finding stretches in segement of a trip which share or follow same path


1. Convert latitude longitude to x and y coordinate

2. Cluster segments which share same coordinate location of start and stop points

3. Divide each segement point to displacements of equal parameters

4. Find angles between each segment point and find average angle to find similar stretches

5. Find speed within stretches to get proper driving behaviour in a particular driving location


"""
#Calling libraries


import os
import pandas as pd
import datetime
from datetime import timedelta
import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotnine as p9
import csv
# Defining functions





def haversine(coord1, coord2):
    """
    Haversine distance in meters
    
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



def latlong_to_coordinate(x,y,ref_lat,ref_long):
    
    """
    Converting latitude and longitude to x-y coordinate system using a reference lat and long
    
    In this case to project every distance as postitve we use reference latitude and longitude of Lakshwadeep
    
    """
    
    xdist = haversine((0,ref_lat),(x,y))
    
    ydist = haversine((ref_long,0),(x,y))
    
    return xdist,ydist



def Check_Points(df,i,dist_cutoff):
     """
     For checking start ans stop points didtance and determining which start and stop point to take
     
     """
     if(i % 1000) == 0:
         print("Done for %d Points"%i)
     df['SDist'][i] =  haversine((df['StartLat'][i-1],df['StartLong'][i-1]),(df['EndLat'][i],df['EndLong'][i]))
     if df['SDist'][i] > dist_cutoff:
         df['ProblemPoints'][i]= "S"
         if df['ProblemPoints'][i-1] == "S":
             df['ProblemPoints'][i-1] = "SE"
         else:
              df['ProblemPoints'][i-1] = "E"


r = lambda f,p: f - f % p # For rounding decimals



def cartesian_distance(x0,y0,x1,y1):
    
    return  np.power(np.sum(np.power(x1 - x0,2) + np.power(y1 - y0,2)),0.5)


def line_plot_cluster(x, y):
    
    """
     Takes a list of xy coordinates plots a straight line and then calculates distance of point from line to 
    
     find maximum distance and return x point cluster
        
    """
    
    a = y[-1] - y[0] 
    b = x[-1] - x[0]  
    m = a / b
    
    c = y[0] - m * x[0]
   
    print("Line Solution is y = {m}x + {c}".format(m=m,c=c))
      
   
    
    coordinates = [(_x,_y) for _x,_y in zip(x[1:len(x)-1],y[1:len(x)-1])]
    
    # Finding maximum distance
    point_dist = [m*i[0] - i[1] + c / np.sqrt(np.power(m,2) + np.power(1,2)) for i in coordinates]
    
    # returning number of clusters
    return x[point_dist.index(np.max(point_dist)) + 1]
    
    
    #m = y[-1] - y[0] / x[-1] - x[0]
    
    #c = y[-1] - m * x[-1] 
    
def same_geo_points(X,ref_dist1,dir):
    
    """
    
    To find centroid to be given to kmeans algorithm for finding geographical clusters spaced together
    
    """
    t = datetime.datetime.now()
    print(t)
    i = 0
    Z = pd.DataFrame()

    centroid = pd.DataFrame(columns = ['X','Y','Points','Type'])
 
    while len(X) != 0:
        

        i = i + 1
        
        X['Dist'] = 1
            
        X['Dist'] = [cartesian_distance(X['Latitude'][0],X['Longitude'][0],X['Latitude'][i],X['Longitude'][i]) for i in range(len(X))]
            
        X['Cluster'] = i
        
        Y = pd.DataFrame()
        
        ref_dist2 =ref_dist1
    
        while True:
            
            Y = Y.append(X[X.Dist <= ref_dist2].reset_index(drop = True)).reset_index(drop = True)
            
            X = X[X.Dist > ref_dist2].reset_index(drop = True)
                   
            X['Dist'] = [cartesian_distance(np.mean(Y['Latitude']),np.mean(Y['Longitude']),X['Latitude'][i],X['Longitude'][i]) for i in range(len(X))]
            
            point_dist = [cartesian_distance(np.mean(Y['Latitude']),np.mean(Y['Longitude']),Y['Latitude'][i],Y['Longitude'][i]) for i in range(len(Y))]
            
            ref_dist2 = max(ref_dist1,2 * np.mean(point_dist))
            
            if len(X[X.Dist < ref_dist2]) == 0 :
                break
        
        Z = Z.append(Y).reset_index(drop = True)
        print("Done for %d points in %s time and %d is left"%(len(Y),datetime.datetime.now() - t,len(X)))
        centroid = centroid.append({'X':np.mean(Y['Latitude']) ,'Y':np.mean(Y['Longitude']),'Points':len(Y),'Type':Y['ProblemPoints'][0]},ignore_index=True)
        
    with open(dir + "Features\\" + "Centroid" + ".csv", 'a') as metricfile:
       centroid.to_csv(metricfile,index = False,header=False)
        #kmeans = KMeans(init=np.array(list(zip(centroid.X,centroid.Y))),n_clusters=len(centroid)).fit(Z[['Latitude','Longitude']])
        
    elbow = pd.DataFrame(columns=['K-Value','Distances'])
    Sum_of_squared_distances = []

    K = range(1,101)
    X = list(zip(centroid.X,centroid.Y))
    for k in K:
        #print("Clustering for %d in "%k,str(datetime.datetime.now() - t))
        km = KMeans(n_clusters=k)
        km = km.fit(np.array(X))
        Sum_of_squared_distances.append(km.inertia_)
        elbow = elbow.append({'K-Value':k,'Distances':km.inertia_},ignore_index = True)
    n = line_plot_cluster([k for k in range(1,101)],Sum_of_squared_distances) #Specifying number of clusters
    C = KMeans(n_clusters=n)
    C = C.fit(X)
    centroid['Cluster'] = C.labels_
    
    i = 0
    j = 0
    while j < len(centroid):
        b = int(centroid['Points'][j])
        Z['Cluster'][i:b+i] =  centroid['Cluster'][j]
        i = b + i
        j = j + 1
    return Z
    
 
def proper_coordinates(df,dir):
    '''
    To do proper scaling and arranging df columns in proper order with problem points
    
    
    '''
    
    if 'StartLat' in df.columns:
        
        X = df
        
        b =  [latlong_to_coordinate(X.StartLat[i],X.StartLong[i],np.min(df.StartLat),np.min(df.StartLong)) for i in range(len(X))]
        
        X['Latitude'] = [i[0] for i in b] #For getting latitude in x coordinate
        
        X['Longitude'] = [i[1] for i in b] #For getting latitude in y coordinate
 
        Z = same_geo_points(X,0.5,dir)
    
        Z.StartCluster = Z.Cluster
        
    
    else:
           X = df
            
           b =  [latlong_to_coordinate(X.EndLat[i],X.EndLong[i],np.min(df.EndLat),np.min(df.EndLong)) for i in range(len(X))]
            
           X['Latitude'] = [i[0] for i in b] #For getting latitude in x coordinate
            
           X['Longitude'] = [i[1] for i in b] #For getting latitude in y coordinate
     
           Z = same_geo_points(X,0.5,dir)
        
           Z.EndCluster = Z.Cluster
                       
    
    return Z
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__': 
    
 
 dir = "C:\\Users\\saura\\Documents\\Clean Records\\"
 os.chdir(dir + "Data")  #Changing current working directory
 directory = os.getcwd()  #Getting current working directory of the CLI
 files = next(os.walk(directory))[2] #Creates a list of all files
   
 summary = pd.DataFrame(columns = ['IMEI','TripID','SegmentID','TotalDistanceTravelled','TotalDisplacement','StartLat','StartLong','EndLat','EndLong','StartDateTime','EndDateTime'])
 with open(dir + "Features\\" + "Summary_Cleaned_IMEI" +".csv", 'a') as imeifile:
                      summary.to_csv(imeifile,index = False,header=True)
 for f in files:
#for f in files:
    df = pd.read_csv(dir +"Data\\" +   f , converters={'Latitude': str,'Longitude':str})
   
    if len(df) > 1:
        df['Remove'] = df['Remove'].astype(str)
        df = df[df['Remove'] == "0"].reset_index(drop=True)
        df = df[['IMEI','Date Time','Latitude','Longitude','TripID','SegmentID']]
        df['Latitude'] = [r(pd.to_numeric(df['Latitude'][i]),0.0000001) for i in range(0,len(df))]
        df['Longitude'] = [r(pd.to_numeric(df['Longitude'][i]),0.0000001) for i in range(0,len(df))]
        df['Date Time'] = df['Date Time'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        trip = df['TripID'].unique().tolist()
        for i in trip:
            dft = df[df['TripID'] == i].reset_index(drop=True)
            segment = dft['SegmentID'].unique().tolist()
            for s in segment:
                dfs = dft[dft['SegmentID'] == s].reset_index(drop=True)
                b = (dfs['Latitude'][0],dfs['Longitude'][0])
                l =  (dfs['Latitude'][len(dfs)-1],dfs['Longitude'][len(dfs)-1])
                if haversine(b,l) > 1:
                   summary = summary.append({'IMEI':dfs['IMEI'][0],'TripID':dfs['TripID'][0],'SegmentID':dfs['SegmentID'][0],'TotalDistanceTravelled':np.sum([haversine((dfs['Latitude'][i],dfs['Longitude'][i]),(dfs['Latitude'][i - 1],dfs['Longitude'][i - 1])) for i in range(1,len(dfs))]),'TotalDisplacement':haversine(b,l),'StartLat':b[0],'StartLong':b[1],'EndLat':l[0],'EndLong':l[1],'StartDateTime':dfs['Date Time'][0],'EndDateTime':dfs['Date Time'][len(dfs)-1]}, ignore_index=True)
                
        if len(summary) % 1000 ==0:
           print("Done for 1000 files")
           with open(dir + "Features\\" + "Summary_Cleaned_IMEI" +".csv", 'a') as imeifile:
                      summary.to_csv(imeifile,index = False,header=False)
           summary = pd.DataFrame(columns = ['IMEI','TripID','SegmentID','TotalDistanceTravelled','TotalDisplacement','StartLat','StartLong','EndLat','EndLong','StartDateTime','EndDateTime'])
         
               
 cluster_df = pd.read_csv(dir +"Features\\" + "Summary_Cleaned_IMEI" +".csv")
 
 cluster_df = cluster_df[cluster_df.StartLat.between(left=10, right=36)].reset_index(drop=True)
 cluster_df = cluster_df[cluster_df.EndLat.between(left=10, right=36)].reset_index(drop=True)
 cluster_df = cluster_df[cluster_df.StartLong.between(left=70, right=94)].reset_index(drop=True)
 cluster_df = cluster_df[cluster_df.EndLong.between(left=70, right=94)].reset_index(drop=True)
 
 cluster_df['SDist'] = 0
 
 cluster_df['ProblemPoints'] = 0
  
 #For marking and labelling start and end problem points              
 g = [Check_Points(cluster_df,i,0.5) for i in range(1,len(cluster_df))] # for calling function and getting valid points marked
    
 cluster_df.to_csv(dir + "Features\\" + "Geo_IMEI" + ".csv")
 #start  = [(df['StartLat'][i],df['StartLong'][i]) for i in range(0,len(df)) if df['ProblemPoints'][i] != "E"]
 #stop =  [(df['EndLat'][i],df['EndLong'][i]) for i in range(0,len(df)) if df['ProblemPoints'][i] != "S"]
 
 cluster_df = pd.read_csv(dir +"Features\\" + "Geo_IMEI" + ".csv" )
 
 X = cluster_df[cluster_df['ProblemPoints'] != "SE"].reset_index(drop=True)
 
 #X = X[X['TotalDisplacement'] > 1].reset_index(drop=True)
 
 X = X[['IMEI','TripID','SegmentID','StartDateTime','EndDateTime','StartLat','StartLong','EndLat','EndLong','TotalDistanceTravelled','TotalDisplacement','ProblemPoints']]
 
 X['StartCluster'] = -1
 
 X['EndCluster'] = -1
 
 final_df = pd.DataFrame()
 
 start = X[X['ProblemPoints'] != "S"].reset_index(drop=True)
 
 end =  X[X['ProblemPoints'] != "E"].reset_index(drop=True)
 
 
 # For clutering start and end points for points having both start and end points
 temp_start = proper_coordinates(start[['IMEI','TripID','SegmentID','StartDateTime','EndDateTime',"StartLat","StartLong","StartCluster",'TotalDistanceTravelled','TotalDisplacement','ProblemPoints']],dir)
 
 temp_end = proper_coordinates(end[['IMEI','TripID','SegmentID','StartDateTime','EndDateTime',"EndLat","EndLong","EndCluster",'TotalDistanceTravelled','TotalDisplacement','ProblemPoints']],dir)
 
 
 # For joining columns of both df in proper format
 temp_start = temp_start.rename(columns={"Latitude": "StartLatitude", "Longitude": "StartLongitude"})
 
 temp_end = temp_end.rename(columns={"Latitude": "EndLatitude", "Longitude": "EndLongitude"})
 
 temp_start.to_csv(dir + "Features\\" + "Start_Cluster" + ".csv")
 
 temp_end.to_csv(dir + "Features\\" + "End_Cluster" + ".csv")
 
 temp = pd.concat([temp_start[temp_start['ProblemPoints'] == "E"].reset_index(drop=True),temp_end[temp_end['ProblemPoints'] == "S"].reset_index(drop=True)],axis = 0,join = 'outer').reset_index(drop=True)
 
 temp = temp[['IMEI','TripID','SegmentID','StartDateTime','EndDateTime','StartLat','StartLong','EndLat','EndLong','StartLatitude','StartLongitude','EndLatitude','EndLongitude','TotalDistanceTravelled','TotalDisplacement','ProblemPoints','StartCluster','EndCluster']]
 
 temp1 =  pd.concat([temp_start[temp_start['ProblemPoints'] == "0"].reset_index(drop=True),temp_end[temp_end['ProblemPoints'] == "0"].reset_index(drop=True)[['EndLat','EndLong','EndLatitude','EndLongitude','EndCluster']]], axis=1).reset_index(drop=True)
 
 temp =  pd.concat([temp,temp1], axis=0, join='outer').reset_index(drop=True)

 #temp = temp.append(temp_start[temp_start['ProblemPoints'] == "0"].reset_index(drop=True)).reset_index(drop=True)

 #temp = temp.append(temp_end[temp_end['ProblemPoints'] == "0"].reset_index(drop=True)).reset_index(drop=True)
 
 temp = temp.replace(np.nan,-1, regex=True)

 
 # For cmarking routes by start and end points 
 temp['Route'] = [(str(int(temp['StartCluster'][i])) + "_" + str(int(temp['EndCluster'][i]))) for i in range(len(temp))]
 
 final_df = temp[['IMEI','TripID','SegmentID','StartDateTime','EndDateTime','StartLat','StartLong','EndLat','EndLong','StartLatitude','StartLongitude','EndLatitude','EndLongitude','TotalDistanceTravelled','TotalDisplacement','ProblemPoints','StartCluster','EndCluster','Route']]
 
 final_df.to_csv(dir + "Features\\" + "Cluster_Features" + ".csv")
 
 #df['StartDateTime'] = df['StartDateTime'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
 #df['EndDateTime'] = df['EndDateTime'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))


 #c = cluster_data(Start,2,25,153,0.1) #Calling cluster data to see convergence and check

 



#df.to_csv(dir + "Segment_Stretch" + ".csv")


'''
# For plotting
 i = 0
 while i < np.max(X.Cluster):
   i = i + 1000
   p9.ggplot(data=X[X.Cluster.between(left=0, right=i)], mapping=p9.aes(x='StartX', y='StartY',color='Cluster')) + p9.geom_point(alpha=0.8)
   
tripseg = pd.DataFrame(columns= ['Cluster','Points'])
tripseg.Cluster = [i for i in range(np.max(Z.Cluster))]
tripseg.Points = [len(Z[Z.Cluster==i]) for i in range(np.max(Z.Cluster))]





def cluster_data(df,n,i,ref_cluster,ref_dist):
    
   
    t = datetime.datetime.now()
    K = range(0,i)
    X = df[['StartX','StartY']]
    X['Cluster'] = np.zeros(len(X))
    for k in K:
        print("Clustering for %d in "%k,str(datetime.datetime.now() - t))
        kmeans = KMeans(n_clusters=n).fit(X[['StartX','StartY']])
        X['Label'] = kmeans.labels_
        
        #Unique cluster by finding results in each iteration
        X['Cluster'] = [str(X.Cluster[i]) + "-" + str(X['Label'][i]) for i in range(0,len(X))]
        
        #Finding centroid of latitude and longitude for each unique cluster ID
    u = X['Cluster'].unique().tolist()
    
    
    #For assigning cluster id's to the clustered points
    dict = {u[i] : i for i in range(len(u))}
    X = X.replace({"Cluster": dict})
    
    u = X['Cluster'].unique().tolist()
    
    cent_lat = []
    cent_long = []
    
    
    for j in u:
        
        cent_lat.append(np.mean(X[X['Cluster'] == j]['StartX']))
        cent_long.append(np.mean(X[X['Cluster'] == j]['StartY']))

    centroid = list(zip(cent_lat,cent_long))
    

    dist = []
    for j in u:
        segment = X[X['Cluster'] == j].reset_index(drop=True)
        mean_lat = np.mean(segment.StartX)
        mean_long = np.mean(segment.StartY)
        dist.append(np.mean([np.power(np.sum(np.power(segment.StartX[i] - mean_lat,2) + np.power(segment.StartY[i] - mean_long,2)),0.5) for i in range(0,len(segment))]))
    
    #metric = pd.DataFrame(columns=['N','Type','Mean Distances','Centroid','Distances','Minimum cluster'])
    #metric = metric.append({'N':n,'Type':"N-Cluster",'Mean Distances':np.mean(dist),'Centroid':centroid,'Distances':dist,'Minimum cluster':len([x > ref_dist for x in dist])},ignore_index = True)
    #with open(dir + "Segment_Geo" + ".csv", 'a') as metricfile:
        #metric.to_csv(metricfile,index = False,header=False)

   
    if  len(centroid) < ref_cluster:
       print(" Cluster centers have not converged for n = %d"%n)
       c = cluster_data(X,len(centroid),i,ref_cluster,ref_dist)
       #if c[0] ==  True:
           #print(" Cluster centers have converged for n = %d"%n)
           #return c[1],c[2]
        
    else:
        print(" Cluster centers have converged for n = %d"%len(centroid))
        return X,len([x > ref_dist for x in dist])



 start = list(zip(df.StartLat,df.StartLong))
 
 stop = list(zip(df.EndLat,df.EndLong))
 
 b =  [latlong_to_coordinate(x[0],x[1]) for x in start]
 
 l =  [latlong_to_coordinate(x[0],x[1]) for x in stop]
                         
 start_angle = [angle(x) for x in b]
 
 stop_angle =  [angle(x) for x in l]
 
 start_distance = [np.sqrt(np.power(x[0],2) + np.power(x[1],2)) for x in b]
 
 stop_distance = [np.sqrt(np.power(x[0],2) + np.power(x[1],2)) for x in l]
 
 sigma_start =  np.std(start_angle)
 
 sigma_stop = np.std(stop_angle)
 
 normangle_start = [x / sigma_start for x in start_angle]
 
 normangle_stop = [x / sigma_stop for x in stop_angle]
 
 sigma_start =  np.std(start_distance)
 
 sigma_stop = np.std(stop_distance)
 
 norm_start_distance =  [x / sigma_start for x in start_distance]
 
 norm_stop_distance =  [x / sigma_stop for x in stop_distance]
 




 #Stretches having start and stop points are clusered to find regions sharing same geography
 t = datetime.datetime.now()
 segments = pd.DataFrame({"Start Latitude":df['StartLat'],"Start Longitude":df['StartLong'],"Stop Latitude":df['EndLat'],"Stop Longitude":df['EndLong'],"Label":0,"Start Distance":norm_start_distance,"Stop Distance":norm_stop_distance,'Start_angle':start_angle,'Stop_angle':stop_angle})                        
 X=segments.loc[:,['Start Distance','Stop Distance','Start_angle','Stop_angle']]
 
 #For finding optimum clusters
    with open(dir + "elbow" + ".csv", 'a') as elbowfile:
        elbow.to_csv(elbowfile,index = False,header=False)
        
      
 # Checking for elbow point descriptively
 plt.figure(figsize=(12,12))
 plt.ylabel('Inertia', fontsize=12)
 plt.xlabel('Clusters', fontsize=12)  
 plt.plot(K,Sum_of_squared_distances)
 
 df = pd.read_csv(dir + "elbow.csv")
 Sum_of_squared_distances = df.Inertia.tolist()
 
 
 



k = same_geo_points(X,0.1)
cent = list(zip(k['X'],k['Y']))
X = cluster_datacentroid(X,cent,10)












#Clustering again to find points of cluster
 n = line_plot_cluster([k for k in range(1,101)],Sum_of_squared_distances) #Specifying number of clusters
 C = KMeans(n_clusters=n)
 C = C.fit(X)
 id_label = C.labels_
 segments['Label'] = C.labels_
 

#For visualiziing the marked clusters
 
 plot = pd.DataFrame()
    
 start = segments[['Start Latitude','Start Longitude','Label']]
 start['Mark'] = 'S'
 stop = segments[['Stop Latitude','Stop Longitude','Label']]
 stop['Mark'] = 'E'
 stop.rename(columns = {'Stop Latitude':'Start Latitude','Stop Longitude':'Start Longitude','Label':'Label','Mark':'Mark'},inplace = True)
 plot = start.append(stop).reset_index(drop=True)
 

 # Selecting where distance is less than 1 k.m
 #df = df[df['TotalDistanceTravelled'] > 1].reset_index(drop= True)
 
 df['StartLat'] = [r(pd.to_numeric(df['StartLat'][i]),0.0000001) for i in range(0,len(df))]
 df['StartLong'] = [r(pd.to_numeric(df['StartLong'][i]),0.0000001) for i in range(0,len(df))]
 df['EndLat'] = [r(pd.to_numeric(df['EndLat'][i]),0.0000001) for i in range(0,len(df))]
 df['EndLong'] = [r(pd.to_numeric(df['EndLong'][i]),0.0000001) for i in range(0,len(df))]
 #Removing those which are out of India
 df = df[df.StartLat.between(left=10, right=36)].reset_index(drop=True)
 df = df[df.EndLat.between(left=10, right=36)].reset_index(drop=True)
 df = df[df.StartLong.between(left=70, right=94)].reset_index(drop=True)
 df = df[df.EndLong.between(left=70, right=94)].reset_index(drop=True)
   
    
    
#For Map plotting
 import gmplot
 gmap3 = gmplot.GoogleMapPlotter(df['StartLat'][len(df)/2], 
                                df['StartLong'][len(df)/2], 13) 
 gmap3.scatter( df['StartLat'][0:10000], df['StartLong'][0:10000], '# FF0000', 
                              size = 40, marker = True )
 gmap3.plot(df['StartLat'][0:10000], df['StartLong'][0:10000], 'yellow', edge_width = 2.5)
 
 gmap3.draw( "C:\\Users\\saura\\Desktop\\map13.html" ) 
 
 gmap1 = gmplot.GoogleMapPlotter(segments['Start Latitude'].iloc[round(len(segments['Start Latitude'])/2)],segments['Start Longitude'].iloc[round(len(segments['Start Longitude'])/2)],13)
 gmap1.scatter(segments['Start Latitude'][:100], segments['Start Longitude'][:100], '# FF0000', size = 40, marker = False )
 
 gmap1.plot(segments['Start Latitude'][:100], segments['Start Longitude'][:100], 'yellow', edge_width = 2.5)
 gmap1.scatter(segments['Stop Latitude'][:100], segments['Stop Longitude'][:100], '# FF0000', size = 40, marker = False )
 gmap1.plot(segments['Stop Latitude'][:100], segments['Stop Longitude'][:100], 'green', edge_width = 2.5)  
    
 gmap1.draw(f"C:\\Users\\saura\\Desktop\\cluteredmap.html")
    











    
   



def angle(coords):
    
    """
    
    To find angle of start and stop points with coordinate axis
    
    """
    
    return np.arctan2(coords[1],coords[0])* 180 / np.pi
    
  

def cluster_datacentroid(X,centroid,i):
    
   
    #To cluster out data with centroids with i iterations
    
    
    
    
    t = datetime.datetime.now()
    K = range(0,i)
    X = X[['StartX','Longitude']]
    X['Cluster'] = np.zeros(len(X))
    for k in K:
        print("Clustering for %d in "%k,str(datetime.datetime.now() - t))
        kmeans = KMeans(init=np.array(centroid),n_clusters=len(centroid)).fit(X[['StartX','StartY']])
        X['Label'] = kmeans.labels_
        
        #Unique cluster by finding results in each iteration
        X['Cluster'] = [str(X.Cluster[i]) + "-" + str(X['Label'][i]) for i in range(0,len(X))]
        
    #Finding centroid of latitude and longitude for each unique cluster ID
    u = X['Cluster'].unique().tolist()

   #For assigning cluster id's to the clustered points
    dict = {u[i] : i for i in range(len(u))}
    X = X.replace({"Cluster": dict})
    
    u = X['Cluster'].unique().tolist()
    
    
    dist = []
    cent_lat = []
    cent_long = []
    
    for j in u:
        cent_lat.append(np.mean(X[X['Cluster'] == j]['StartX']))
        cent_long.append(np.mean(X[X['Cluster'] == j]['StartY']))

    centroid = list(zip(cent_lat,cent_long))
    
    for j in u:
        segment = X[X['Cluster'] == j].reset_index(drop=True)
        mean_lat = np.mean(segment.StartX)
        mean_long = np.mean(segment.StartY)
        dist.append(np.mean([np.power(np.sum(np.power(segment.StartX[i] - mean_lat,2) + np.power(segment.StartY[i] - mean_long,2)),0.5) for i in range(0,len(segment))]))

    
    metric = pd.DataFrame(columns=['N','Type','Mean Distances','Centroid','Distances','Minimum cluster'])
    metric = metric.append({'N':len(np.array(centroid)),'Type':"Centroid Cluster",'Mean Distances':np.mean(dist),'Centroid':centroid,'Distances':dist,'Minimum cluster':len([x > ref_dist for x in dist])},ignore_index = True)
 
    with open(dir + "Segment_Geo" + ".csv", 'a') as metricfile:
        metric.to_csv(metricfile,index = False,header=False)
    
   
    print(" Cluster centers have converged for n = %d with clustercenters"%len(u))
    return X


 
   
 '''   
    
    






