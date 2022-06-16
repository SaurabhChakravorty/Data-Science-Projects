# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:27:08 2018

@author: saurabh_Sensorise
"""
import os
import datetime
import warnings
import pandas as pd
warnings.simplefilter('ignore', UserWarning)
import numpy as np
import multiprocessing as mp
import csv

from Seperate_Segments import check_segments
from Seperate_Segments import haversine
from Seperate_Segments import segments

from RDP_Modified import haversine
from RDP_Modified import checksegmentpoints
from RDP_Modified import call_rdp
from RDP_Modified import distance_speed

from Mark_Point import mark_point



def call_functions(file,dir):
    epsilon = 0.001  #For rdp distance tolerance
    zeta = 10        #For rdp speed tolerance
    tau = 0.5        #For time cutt off in mark point to mark trips
    bigtau = 6        #For time cutt off in mark point to mark trips
    speed_ref1 = 120  #For speed cutt off in mark point to mark trips
    speed_ref2 = 119  #For speed cutt off in mark point to mark trips
    dist_ref = 100    #For speed cutt off in segement point to mark segments
    static_dist = 0.035 #To remove duplicte error points 
    
    speed_trip = 5
    t = datetime.datetime.now()
    print("Final cleaned file of %s to be Generated"%file)  
    df = mark_point(dir,file,tau,static_dist,bigtau,speed_trip)   # to mark trips
    print("Finished Mark Point of file %s in %s"%(file,str(datetime.datetime.now() - t))) 
    df = segments(df,speed_ref1,speed_ref2,dist_ref,static_dist)  # to mark segments
    print("Finished segmenting of file %s in %s"%(file,str(datetime.datetime.now() - t)))
    call_rdp(df,dir,file,epsilon,zeta)  # to remove rdp points
    print("Finished RDP and Cleaned IMEI File Generated ,Time taken to handle file %s of size %f MB is %s" %(file,os.path.getsize(dir + "Data\\" + file) / (1024*1024),str(datetime.datetime.now() - t)))      
    
    #os.remove(dir + "IMEI_Data\\" + file)
 

if __name__ == '__main__':  
 dir = "C:\\Users\\saura\\Documents\\Clean Records\\"
 os.chdir(dir  + "Data")  #Changing current working directory
 directory = os.getcwd()  #Getting current working directory of the CLI
 files = next(os.walk(directory))[2] #Creates a list of all files
 pool = mp.Pool(processes=4)
 #Values to be paased for processing epsilon,Zeta,Tau,Zoom_ref
 print("Operation IMEI Clean up started %s"%datetime.datetime.now())
 with open(dir + 'Clean_Record_Summary.csv', mode='a',newline='') as recordfile:
        imei_writer = csv.writer(recordfile, delimiter=',')
        imei_writer.writerow(["IMEI","Points","Duplicate Points","Static Points","Total Trip","Total Segements","RDP Points Removed","Max Segment","Maximum Time","Total Time","Segment max Number","Trip max number","Small Segments"])

 var = [pool.apply_async(call_functions,args=[x,dir]) for x in files]
 print(var)
 pool.close()
 pool.join()

