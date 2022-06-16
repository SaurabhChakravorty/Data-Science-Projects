# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 15:51:10 2018

@author: saura
"""


#Importing all libraries
import os
import pandas as pd
import time
import numpy as np
import datetime
from datetime import datetime
import multiprocessing as mp
import sys
import csv
import re


#Function for getting tokens out of packet
def read_token(filename,delimiter,buffer):
    if buffer == "":
        buffer = filename.readline()
        if (not buffer):
            return list(["",""])

    if delimiter == "":
        token = buffer.strip('[\"\n]')
        return list([token,""])

    token = buffer.split(delimiter,1)
    tokens = [x.strip('[\"\n]') for x in token]
    while(len(tokens)==1):
        buffer1 = filename.readline()
        buffer = buffer + buffer1.strip('[\"\n]')
        tokens = buffer.split(delimiter,1)
        tokens = [x.strip('[\"\n]') for x in tokens]
    return tokens


def parse_date(text):
    if (len(text) == 6):
        if (text[0:2] == "18"):
            try:
             recdate = datetime.strptime(text,'%y%m%d')
             date_format = recdate.strftime('%Y%m%d')
             return True,date_format
            except ValueError:
             return False,"Date Format Not Found"

        elif (text[-2:] == "18"):
            try:
             recdate = datetime.strptime(text,'%d%m%y')
             date_format = recdate.strftime('%Y%m%d')
             return True,date_format
            except ValueError:
             return False,"Date Format Not Found"

        else:
            return False,"Date Format Not Found"

    elif (len(text)==8):
        if (text[0:4] == "2018"):
            try:
             recdate = datetime.strptime(text,'%Y%m%d')
             date_format = recdate.strftime('%Y%m%d')
             return True,date_format
            except ValueError:
             return False,"Date Format Not Found"

        elif (text[-4:] == "2018"):
            try:
             recdate = datetime.strptime(text,'%d%m%Y')
             date_format = recdate.strftime('%Y%m%d')
             return True,date_format
            except ValueError:
             return False,"Date Format Not Found"
        else:
            return False,"Date Format Not Found"
    else:
        return False,"Date Format Not Found"



def isTimeFormat(text1):
    try:
        t = time.strptime(text1,'%H%M%S')
        return True,text1[:2] + ":" + text1[2:4] + ":" + text1[4:6]
    except ValueError:
        return False,"Time Format Not Found"

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def process(record):

    newrecord = ""
    modelrecord = "" # Creating newrecord and model record string for appending values
    records = record.replace(';','').split(",")
    noToks = len(records)
    if(noToks < 30):
        return False,"Minimum Number of Records not found", record

    if not (str(records[0]).startswith('$')):
        if not ('$' in str(records[0])):
          return False,"Start character not found",record
        records[0] = "$" + records[0].split('$',1)[1]


    curTok = 0
    prevTok = 0
    header = records[curTok]
    curTok = curTok + 1
    if(len(header) == 1):
        header = header + records[curTok]
        prevTok = curTok
        curTok = curTok + 1


    newrecord = header #Added header


    ## Find Packet Type
    while (curTok < noToks):
        if records[curTok] in ["NR","EA","HP","IN","IF","TA","BD","BR","BL","TS","TE","HB","HA","RT"]:
            break
        curTok=curTok+1
    
    if (curTok >= noToks):
        return False,"No Packet type found",record
    
    if (curTok - 2) > prevTok and is_float(records[curTok - 2]) == True:
        newrecord = newrecord + "," + records[curTok - 2]  # added Vendor id
        modelrecord = modelrecord +  records[curTok - 2]
        
    else:
        newrecord = newrecord + ",VENDOR NOT FOUND"
        modelrecord = modelrecord + "00"
        
    if ((curTok - 1) > prevTok):
        newrecord = newrecord + "," + records[curTok-1]  # added Firmware Version
    else:
        newrecord = newrecord + ",FW Missing"

    newrecord = newrecord + "," + records[curTok] ## Added PacketType
    modelrecord = modelrecord + "," + records[curTok]
    prevTok = curTok
    curTok= curTok+1


    ## Validate PaketStatus
    if (records[curTok] not in ("L","H")):
        return False,"Packet Status not found",record

    newrecord = newrecord + "," + records[curTok] ## Added PacketStatus
    modelrecord = modelrecord + "," + records[curTok]

    curTok= curTok+1

    ## Validate IMEI
    if (len(records[curTok]) != 15):
        return False,"IMEI length not proper",record

    if is_float(records[curTok])  == False:
        return False,"IMEI length not proper",record
    
    records[curTok] = records[curTok].lstrip('0')
    if (len(records[curTok]) != 15):
        return False,"IMEI length not proper",record
    IMEI = records[curTok] #IMEI for file

    newrecord = newrecord + "," + records[curTok] ## Added IMEI
    modelrecord = modelrecord + "," + records[curTok]
    prevTok = curTok
    curTok= curTok+1


    #No validation add vehicle Number as it is
    newrecord = newrecord + "," + records[curTok] ## Added Vehicle Reg No
    modelrecord = modelrecord + "," + records[curTok]
       
    curTok= curTok+1


    ## Find Date
    while (curTok < noToks):
        if (((len(records[curTok]) == 6) or (len(records[curTok]) == 8)) and records[curTok].isdigit()):
             break
        curTok = curTok + 1

    if (curTok >= noToks):
        return False,"Date Not found",record


    ##Validate Date with function
    dt = parse_date(records[curTok])
    if dt[0] == False:
       return False,"Date Not found",record

    #Validate GPS Fix
    if records[curTok - 1] not in ("0","1"):
     newrecord = newrecord + "," + "00"
     modelrecord = modelrecord + "," + "00"

    else:
     newrecord = newrecord + "," + records[curTok - 1] ## Added GPS Fix
     modelrecord = modelrecord + "," + records[curTok - 1]

    newrecord = newrecord + "," + dt[1] ## Added date
    modelrecord = modelrecord + "," + dt[1]
    curTok = curTok + 1

    #Validate time format
    time =  isTimeFormat(records[curTok])

    if time[0] == True:
         newrecord = newrecord + "," + time[1] ## Added time
         modelrecord = modelrecord + " " + time[1]
    else:
     return False,"Time Not found",record
    curTok = curTok + 1


    #Validate Latitude
   
    Lat = is_float(records[curTok])
    if Lat == True and bool(re.match('^[0.]+$', records[curTok])) == False:
         newrecord = newrecord + "," + records[curTok] ## Added Latitude
         modelrecord = modelrecord + "," + records[curTok]
    else:
      return False,"No Latitude",record
    curTok = curTok + 1

    #Validate Latitude direction
    if records[curTok] not in ("N","S"):
         records[curTok] = "N"
    newrecord = newrecord + "," + records[curTok] ## Added Latitude Direction
    modelrecord = modelrecord + "," + records[curTok]
    curTok = curTok+1


    #Validate Longitude
   
    Long = is_float(records[curTok])
    if Long == True and bool(re.match('^[0.]+$', records[curTok])) == False :
         newrecord = newrecord + "," + records[curTok] ## Added Longitude
         modelrecord = modelrecord + "," + records[curTok]
    else:
      return False,"No Longitude",record
    curTok = curTok + 1

    #Validate Longitude direction
    if records[curTok] not in ("E","W"):
          records[curTok] = "E"
    newrecord = newrecord + "," + records[curTok] ## Added Longitude Direction
    modelrecord = modelrecord + "," + records[curTok]
    curTok= curTok+1

     #Validate Speed
    Speed = is_float(records[curTok])
    if Speed == True:
      newrecord = newrecord + "," + records[curTok] ## Added Speed
      modelrecord = modelrecord + "," + records[curTok]
    else:
      newrecord = newrecord + "," + "00"
      modelrecord = modelrecord + "," + "00"
    prevTok = curTok
    curTok = curTok + 1

    if (is_float(records[curTok]) == True):
        newrecord = newrecord + "," + records[curTok]  # added Heading in degrees
        modelrecord = modelrecord + "," + records[curTok]
    else:
        newrecord = newrecord + "," + "00"
        modelrecord = modelrecord + "," + "00"
    curTok = curTok + 1

     #Validate number of satellites
    if not records[curTok].isdigit():
        newrecord = newrecord + "," + "00"
        modelrecord = modelrecord + "," + "00"
    else:
        newrecord = newrecord + "," + records[curTok] ## Added satellites
        modelrecord = modelrecord + "," + records[curTok]
    curTok= curTok + 1



    #Validate Alitiude of device
    Alt_dev = is_float(records[curTok])
    if Alt_dev == True:
         newrecord = newrecord + "," + records[curTok] ## Added Altitude of device
         modelrecord = modelrecord + "," + records[curTok]
    else:
         newrecord = newrecord + "," + "00"
         modelrecord = modelrecord + "," + "00"
    prevTok = curTok
    curTok = curTok + 1

    # Validate PDOP
    if is_float(records[curTok]) == True: 
         if float(records[curTok]) < 10:
             newrecord = newrecord + "," + records[curTok] ## Added positional dilution
             modelrecord = modelrecord + "," + records[curTok]
         else:
              return False,"No PDOP accuracy value more than 9",record
    else:
      return False,"Wrong PDOP value",record
    curTok = curTok + 1


    #Validate distance travelled
    ODO = is_float(records[curTok])
    if ODO == True:
         newrecord = newrecord + "," + records[curTok] ## Added distance travelled
    else:
         newrecord = newrecord + "," + "00"
    prevTok = curTok
    curTok = curTok + 1


    if (isinstance(records[curTok],str) == True):
        newrecord = newrecord + "," + records[curTok]  # added Network operator name
        modelrecord = modelrecord + "," + records[curTok]
    else:
        newrecord = newrecord + ",Network Operator not found"
        modelrecord = modelrecord + ",Network Operator not found"
    curTok = curTok + 1


    #Validate Ignition
    if records[curTok] not in ("1","0","O"):
         newrecord = newrecord + "," +  "00"
         modelrecord = modelrecord + "," + "00"
    else:     
        newrecord = newrecord + "," + records[curTok] ## Added Ignition
        modelrecord = modelrecord + "," + records[curTok]
    curTok = curTok+1


      #Validate Main Power Status
    if records[curTok] not in ("1","0","O"):
         newrecord = newrecord + ","  + "00"
         modelrecord = modelrecord + "," + "00"
    else:
        newrecord = newrecord + "," + records[curTok] ## Added Main Power status
        modelrecord = modelrecord + "," + records[curTok]
    prevTok = curTok
    curTok = curTok + 2




    ## Find Tamper Alert
    if records[curTok] not in ("C","O"):
        return True,"No Tamper Alert",newrecord,IMEI,modelrecord + "," + "00" + ',' +  "N" + "," + "00"+ "," + "00"+ "," + "00" + "," + "00"+ "," + "00"+ "," + "00" + "," + "00" + "," + "00"


    if is_float(records[curTok - 1]) == True:
        newrecord = newrecord + "," + records[curTok-1]  # added Emergency status
        modelrecord = modelrecord + "," + records[curTok - 1]
        
    else:
        newrecord = newrecord + ",No Emergency status"
        modelrecord = modelrecord + "," + "00"

    newrecord = newrecord + "," + records[curTok] ## Added Tamper alert
    modelrecord = modelrecord + "," + records[curTok]

    prevTok = curTok
    curTok = curTok + 1


    #Validate GSM Handling signal strength
    if (curTok >= noToks):
       return True,"Records over",newrecord,IMEI,modelrecord + "," + "00"+ "," + "00"+ "," + "00" + "," + "00" + "," + "00"+ "," + "00"+ "," + "00"+ "," + "00"
    if not records[curTok].isdigit() in range(0,31):
        newrecord = newrecord + "," + "00"
        modelrecord = modelrecord + "," + "00"
    else:
        newrecord = newrecord + "," + records[curTok] ## Added  GSM Handling signal strength
        modelrecord = modelrecord + "," + records[curTok]
    prevTok = curTok
    curTok= curTok+1


     #Validate MCC
    if (curTok >= noToks):
        return True,"Records over",newrecord,IMEI,modelrecord + "," + "00"+ "," + "00"+ "," + "00" + "," + "00" + "," + "00"+ "," + "00"+ "," + "00"
    if (is_float(records[curTok]) == True):
        newrecord = newrecord + "," + records[curTok]
        modelrecord = modelrecord + "," + records[curTok]
    else:
        newrecord = newrecord + ",MCC"
        modelrecord = modelrecord + "," +"00"
    prevTok = curTok
    curTok = curTok + 1


     #Validate MNC
    if (curTok >= noToks):
        return True,"Records over",newrecord,IMEI,modelrecord + "," + "00"+ "," + "00"+ "," + "00" + "," + "00" + "," + "00"+ "," + "00"
    if (prevTok and is_float(records[curTok]) == True):
        newrecord = newrecord + "," + records[curTok]
        modelrecord = modelrecord + "," + records[curTok]
    else:
      newrecord = newrecord + ",MNC"
      modelrecord = modelrecord + "," + "00"
    prevTok = curTok
    curTok = curTok + 1
    
    if (curTok >= noToks):
        return True,"Records over",newrecord,IMEI,modelrecord + "," + "00"+ "," + "00"+ "," + "00" + "," + "00"+ "," + "00"
    if (is_float(records[curTok]) == True):
        newrecord = newrecord + "," + records[curTok]  # added LAC
        modelrecord = modelrecord + "," + records[curTok]
    else:
        newrecord = newrecord + ",No LAC"
        modelrecord = modelrecord + "," + "00"
    prevTok = curTok
    curTok = curTok + 1

    if (curTok >= noToks):
        return True,"Records over",newrecord,IMEI,modelrecord + "," + "00"+ "," + "00"+ "," + "00" + "," + "00"
    if (is_float(records[curTok]) == True):
        newrecord = newrecord + "," + records[curTok]  # added Cell ID
        modelrecord = modelrecord + "," + records[curTok]
    else:
        newrecord = newrecord + ",No cellID"
        modelrecord = modelrecord + "," + "00"
    prevTok = curTok
    curTok = curTok + 1

    if (curTok >= noToks):
        return True,"Records over",newrecord,IMEI,modelrecord + "," + "00"+ "," + "00"+"," + "00"
    newrecord = newrecord + "," + records[curTok]  # added LAC1
    curTok = curTok + 1
        
        
    if (curTok >= noToks):
        return True,"Records over",newrecord,IMEI,modelrecord + "," + "00"+"," + "00"+"," + "00"  
    newrecord = newrecord + "," + records[curTok]  # added NMR2
    curTok = curTok + 1
        
    if (curTok >= noToks):
        return True,"Records over",newrecord,IMEI,modelrecord + "," + "00"+"," + "00"+"," + "00"
    newrecord = newrecord + "," + records[curTok]  # added LAC2
    curTok = curTok + 1

    if (curTok >= noToks):
        return True,"Records over",newrecord,IMEI,modelrecord + "," + "00"+"," + "00"+"," + "00"
    newrecord = newrecord + "," + records[curTok]  # added NMR3
    curTok = curTok + 1
        
    if (curTok >= noToks):
        return True,"Records over",newrecord,IMEI,modelrecord + "," + "00"+"," + "00"+"," + "00"
    newrecord = newrecord + "," + records[curTok]  # added LAC 3
    curTok = curTok + 1
        
    if (curTok >= noToks):
        return True,"Records over",newrecord,IMEI,modelrecord + "," + "00"+"," + "00"+"," + "00"
    newrecord = newrecord + "," + records[curTok]  # added NMR4
    curTok = curTok + 1


    if (curTok >= noToks):
        return True,"Records over",newrecord,IMEI,modelrecord + "," + "00"+"," + "00"+"," + "00"
    newrecord = newrecord + "," + records[curTok]  # added LAC4
    curTok = curTok + 1


  #Check for digital input status
    if (curTok >= noToks):
        modelrecord = modelrecord + "," + "00"+"," + "00"+"," + "00"
        return True,"DigitalIntputStatus",newrecord,IMEI,modelrecord

    if(len(records[curTok]) >= 4):
        if(all(records[curTok]) in (0,1)):
            newrecord = newrecord + "," + records[curTok][0:4]  #Added digital input status
            modelrecord = modelrecord + "," + records[curTok]
            curTok = curTok + 1

    elif(len(records[curTok]) == 1 and records[curTok] in (0,1)):
                i1 = records[curTok]
                curTok = curTok + 1

                if(len(records[curTok]) == 1 and records[curTok] in (0,1)):
                 i2 = records[curTok]
                curTok = curTok + 1

                if(len(records[curTok]) == 1 and records[curTok] in (0,1)):
                 i3 = records[curTok]
                curTok = curTok + 1

                if(len(records[curTok]) == 1 and records[curTok] in (0,1)):
                 i4 = records[curTok]
                curTok = curTok + 1
                newrecord = newrecord + "," + i1 + i2 + i3 + i4
                modelrecord = modelrecord + "," + i1 + i2 + i3 + i4
    else:
         newrecord = newrecord + "," + "0000"
         modelrecord = modelrecord + "," + "0000"
         

  #Check for digital output status
    if (curTok >= noToks):
        modelrecord = modelrecord + "," + "00"+"," + "00"
        return True,"DigitalOutputStatus",newrecord,IMEI,modelrecord

    if(len(records[curTok]) >= 2):
        if(all(records[curTok]) in (0,1)):
            newrecord = newrecord + "," + records[curTok][0:2]  #Added digital output status
            modelrecord = modelrecord + "," + records[curTok][0:2]
            curTok = curTok + 1

    elif(len(records[curTok]) == 1 and records[curTok] in (0,1)):
                v1 = records[curTok]
                curTok = curTok + 1

                if(len(records[curTok]) == 1 and records[curTok] in (0,1)):
                 v2 = records[curTok]
                 curTok = curTok + 1
                 newrecord = newrecord + "," + v1 + v2
                 modelrecord = modelrecord + "," + v1 + v2
    else:
          newrecord = newrecord + "," + "00"
          modelrecord = modelrecord + "," + "00"



    #Validate frame number
    if (curTok >= noToks):
        modelrecord = modelrecord + "," + "00"
        return True,"No FrameNumber",newrecord,IMEI,modelrecord

    if not records[curTok].isdigit():
         newrecord = newrecord + "," + "00"
         modelrecord = modelrecord + "," + "00"
    else:
         newrecord = newrecord + "," + records[curTok] ## Added framenumber
         modelrecord = modelrecord + "," + records[curTok]

    prevTok = curTok
    curTok = curTok + 1

    newrecord = newrecord  + "*"


   #Return All Value validated
    return True,"Good_Record",newrecord,IMEI,modelrecord



def read_and_write_single_file(fileIn,dirIn):
       time1 = datetime.now()
#Initialising all variables
       Packets = 0
       Records = 0
       Good = 0
       Bad = 0

       f = fileIn
       recordcount = 0
       st1 = f.split("-")
       f_port = st1[1].strip('[\"\n]')
       f_date = (st1[2] + "-" + st1[3] + "-" + st1[4]).split(".")[0]


        #Creating a record dataframe
       IMEI_Files = dirIn + "\\Data\\"
       File_Summary = pd.DataFrame(columns=['Filename','Packet','Record','GoodRecord','ErrorRecord'])


       #Creating Error Summary file
       row_Error  = ['UID','Port','Recordindex','Desciption','Record']
       if not os.path.exists(dirIn + '\\'  + "Error\\" + "Error_Summary" + "_" + f_port + "_" + f_date +".csv"):
           with open(dirIn + '\\'  +"Error\\" + "Error_Summary" + "_" + f_port + "_" + f_date +".csv", 'a', encoding='utf8') as csvFileE:
               writer = csv.writer(csvFileE)
               writer.writerow(row_Error)
           df_Error = pd.DataFrame(columns = ['UID','Port','Recordindex','Desciption','Record'])
       else:
           print("Delete the error file of Error_Summary_" + str(f_port) + "_" + str(f_date) + " and try again with operation")


       #Creating track file
       row_Track = ['UID', 'Record']
       if not os.path.exists(dirIn + '\\' +  "Track\\" + "Track" + "_" + f_port + "_" + f_date +".csv"):
            with open(dirIn + '\\' +  "Track\\" + "Track" + "_" + f_port + "_" + f_date +".csv", 'a', encoding='utf8') as csvFileT:
               writer = csv.writer(csvFileT)
               writer.writerow(row_Track)
            df_Track = pd.DataFrame(columns = ['UID', 'Record'])
       else:
           print("Delete the track file of Error_Summary_" + str(f_port) + "_" + str(f_date) + " and try again with operation")

       #Creating IMEI dataframe for appending
       df_IMEI = pd.DataFrame(columns = ['Port','VendorID','PacketType','PacketStatus','IMEI','VehicleNumber','GPSFix','Date Time','Latitude','Latitude_Direction','Longitude','Longitude_Direction','Speed','Heading','SatelliteNumber','Altitude','PDOP','Operator','Ignition','MainPowerStatus','EmergencyStatus','TamperAlert','GSMSignal','MCC','MNC','LAC','CellID','DigitalInput','DigitalOutput','FrameNumber'])

       file = open(f,'r+', encoding='utf8')
       t = list([file.readline(),""])
       print("Doing file " + f)
       while (t[0] != ""):
             Packets = Packets + 1
             t = read_token(file,'\";',"")
             UID = t[0]
             t = read_token(file,'\";',t[1])
             t = read_token(file,'\";',t[1])
             packet = t[0]
             t = read_token(file,'\";',t[1])
             t = read_token(file,'',t[1])
             Port = t[0]

             if(f_port != Port):#Apending Error Summary File for port check failure
                  row = pd.DataFrame({'UID':UID,'Port':f_port,'Recordindex':-1,'Desciption':"Port Not Found",'Record':packet},index = [0])
                  df_Error = pd.concat([row, df_Error])
                  #Bad =  Bad + 1
                  continue


             toks = packet.split("*")
             Recordindex = 0

             #Processing each value of records in message
             for tok in toks:
                if(len(tok)!= 0):
                  Recordindex = Recordindex + 1
                  Records = Records + 1
                  rcd_string = process(tok)

            #Appending files on condition
                  if(rcd_string[0] == False): #When condition is false
                    #Appending in Error Summary Files
                      row = pd.DataFrame({'UID':UID,'Port':Port,'Recordindex':Bad,'Desciption':rcd_string[1],'Record':packet},index = [0])
                      df_Error = pd.concat([row, df_Error])
                      Bad = Bad + 1
                      


                  if(rcd_string[0] == True): # when condition is True
                      #Appending in Track files
                      row1 = pd.DataFrame({'UID':UID,'Record':rcd_string[2]},index = [0])
                      df_Track = pd.concat([row1, df_Track])
                      Good = Good + 1

                          #Appending fields of modelling in df of IMEI
                      for_model_data = rcd_string[4].split(",")
                      csvData = []
                      for i in range(len(for_model_data)):
                          csvData.append(for_model_data[i])
                          
                      IMEI = pd.DataFrame({'Port':Port,'VendorID':csvData[0],'PacketType':csvData[1],'PacketStatus':csvData[2],'IMEI':csvData[3],'VehicleNumber':csvData[4],'GPSFix':csvData[5],'Date Time':csvData[6],'Latitude':csvData[7],'Latitude_Direction':csvData[8],'Longitude':csvData[9],'Longitude_Direction':csvData[10],'Speed':csvData[11],'Heading':csvData[12],'SatelliteNumber':csvData[13],'Altitude':csvData[14],'PDOP':csvData[15],'Operator':csvData[16],'Ignition':csvData[17],'MainPowerStatus':csvData[18],'EmergencyStatus':csvData[19],'TamperAlert':csvData[20],'GSMSignal':csvData[21],'MCC':csvData[22],'MNC':csvData[23],'LAC':csvData[24],'CellID':csvData[25],'DigitalInput':csvData[26],'DigitalOutput':csvData[27],'FrameNumber':csvData[28]},index =[0]) 
                      df_IMEI = pd.concat([IMEI, df_IMEI])
                      

                    #Counting lines in this
                  recordcount +=  1
                  if ((recordcount % 100000)==0):
                    df_Track = df_Track.reset_index()
                    df_Error = df_Error.reset_index()
                    df_Track = df_Track[['UID', 'Record']]
                    df_Error = df_Error[['UID','Port','Recordindex','Desciption','Record']]
                    with open(dirIn + '\\'  +"Error\\" + "Error_Summary" + "_" + f_port + "_" + f_date +".csv", 'a', encoding='utf8') as csvFileE:
                      df_Error.to_csv(csvFileE,index=False,header=False)
                    with open(dirIn + '\\' +  "Track\\" + "Track" + "_" + f_port + "_" + f_date +".csv", 'a', encoding='utf8') as csvFileT:
                      df_Track.to_csv(csvFileT,index=False,header=False)
                    df_Track = pd.DataFrame(columns = ['UID', 'Record'])
                    df_Error = pd.DataFrame(columns = ['UID','Port','Recordindex','Desciption','Record'])

                    df_IMEI= df_IMEI.reset_index()
                    df_IMEI = df_IMEI[['Port','VendorID','PacketType','PacketStatus','IMEI','VehicleNumber','GPSFix','Date Time','Latitude','Latitude_Direction','Longitude','Longitude_Direction','Speed','Heading','SatelliteNumber','Altitude','PDOP','Operator','Ignition','MainPowerStatus','EmergencyStatus','TamperAlert','GSMSignal','MCC','MNC','LAC','CellID','DigitalInput','DigitalOutput','FrameNumber']]
                    IMEI_List = df_IMEI['IMEI'].unique().tolist()       
                    for t in IMEI_List:
                      if not os.path.exists(IMEI_Files + t + ".csv"):
                        column = ['Port','VendorID','PacketType','PacketStatus','IMEI','VehicleNumber','GPSFix','Date Time','Latitude','Latitude_Direction','Longitude','Longitude_Direction','Speed','Heading','SatelliteNumber','Altitude','PDOP','Operator','Ignition','MainPowerStatus','EmergencyStatus','TamperAlert','GSMSignal','MCC','MNC','LAC','CellID','DigitalInput','DigitalOutput','FrameNumber']
                        with open(IMEI_Files + t +".csv", 'a', encoding='utf8') as imei_file:
                            writer = csv.writer(imei_file)
                            writer.writerow(column)                    
                            temp_df = df_IMEI[df_IMEI['IMEI'] == t]
                            temp_df.to_csv(imei_file,index = False,header=False)
                      else:
                            temp_df = df_IMEI[df_IMEI['IMEI'] == t] 
                            with open(IMEI_Files + t +".csv", 'a', encoding='utf8') as imeifile:
                             temp_df.to_csv(imeifile,index = False,header=False)
                    df_IMEI = pd.DataFrame(columns = ['Port','VendorID','PacketType','PacketStatus','IMEI','VehicleNumber','GPSFix','Date Time','Latitude','Latitude_Direction','Longitude','Longitude_Direction','Speed','Heading','SatelliteNumber','Altitude','PDOP','Operator','Ignition','MainPowerStatus','EmergencyStatus','TamperAlert','GSMSignal','MCC','MNC','LAC','CellID','DigitalInput','DigitalOutput','FrameNumber'])
                    print("We are going better handled appending Track and Error file of %s and aggregrated %d lines in : "%(f,recordcount),str(datetime.now() - time1))




       # Adding To File Summary
       df_IMEI= df_IMEI.reset_index()
       df_IMEI = df_IMEI[['Port','VendorID','PacketType','PacketStatus','IMEI','VehicleNumber','GPSFix','Date Time','Latitude','Latitude_Direction','Longitude','Longitude_Direction','Speed','Heading','SatelliteNumber','Altitude','PDOP','Operator','Ignition','MainPowerStatus','EmergencyStatus','TamperAlert','GSMSignal','MCC','MNC','LAC','CellID','DigitalInput','DigitalOutput','FrameNumber']]
       IMEI_List = df_IMEI['IMEI'].unique().tolist()       
       for t in IMEI_List:
           if not os.path.exists(IMEI_Files + t + ".csv"):
                    column = ['Port','VendorID','PacketType','PacketStatus','IMEI','VehicleNumber','GPSFix','Date Time','Latitude','Latitude_Direction','Longitude','Longitude_Direction','Speed','Heading','SatelliteNumber','Altitude','PDOP','Operator','Ignition','MainPowerStatus','EmergencyStatus','TamperAlert','GSMSignal','MCC','MNC','LAC','CellID','DigitalInput','DigitalOutput','FrameNumber']
                    with open(IMEI_Files + t +".csv", 'a', encoding='utf8') as imei_file:
                            writer = csv.writer(imei_file)
                            writer.writerow(column)                    
                            temp_df = df_IMEI[df_IMEI['IMEI'] == t]
                            temp_df.to_csv(imei_file,index = False,header=False)
           else:
               temp_df = df_IMEI[df_IMEI['IMEI'] == t] 
               with open(IMEI_Files + t +".csv", 'a', encoding='utf8') as imeifile:
                  temp_df.to_csv(imeifile,index = False,header=False)

       df_Track = df_Track.reset_index()
       df_Error = df_Error.reset_index()
       df_Track = df_Track[['UID', 'Record']]
       df_Error = df_Error[['UID','Port','Recordindex','Desciption','Record']]
       print("***Generating File Summary Gentlemen***")
       print(datetime.now())
       print("File done %s lines handled in %d" %(f,recordcount))
       print("Time taken to handle file %s in hours is " %f,str(datetime.now() - time1))
       with open(dirIn + '\\'  +"Error\\" + "Error_Summary" + "_" + f_port + "_" + f_date +".csv", 'a', encoding='utf8') as csvFileE:
          df_Error.to_csv(csvFileE,index=False,header=False)
       with open(dirIn + '\\' +  "Track\\" + "Track" + "_" + f_port + "_" + f_date +".csv", 'a', encoding='utf8') as csvFileT:
          df_Track.to_csv(csvFileT,index=False,header=False)
       File_Summary = File_Summary.append({'Filename':"filesummary" + f_port + "_" + f_date +".csv",'Packet':Packets,'Record':Records,'GoodRecord':Good,'ErrorRecord':Bad},ignore_index=True)
       File_Summary.to_csv(dirIn + '\\'  +"FileSummary"+ "_" + f_port +"_"+ f_date +".csv")
       print("-----Just check the file, Hope it does not fails or screw up!-----")
