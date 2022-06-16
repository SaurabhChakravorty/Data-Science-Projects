# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:27:08 2018

@author: saurabh_Sensorise
"""
#import Record_parse
import os
import csv
from Record_parse_modified import read_and_write_single_file
from Record_parse_modified import parse_date
from Record_parse_modified import isTimeFormat
from Record_parse_modified import is_float
from Record_parse_modified import process
import multiprocessing as mp
import datetime


if __name__ == '__main__':
    
 directory = "C:\\Users\\saura\\Documents\\Clean Records"   
 os.chdir(directory + "\\Devicedata")  #Changing current working directory
 dir = os.getcwd()  #Getting current working directory of the CLI
 files = next(os.walk(dir))[2] #Creates a list of all files
 pool = mp.Pool(processes=4)
 print(datetime.datetime.now())
 var = [pool.apply_async(read_and_write_single_file,args=[x,directory]) for x in files]
 print(var)
 pool.close()
 pool.join()
