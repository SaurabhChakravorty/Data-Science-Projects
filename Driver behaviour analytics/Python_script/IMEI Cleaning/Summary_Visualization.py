# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:28:25 2019

@author: saura
"""

"""
Bokeh Visualization Template

This template is a general outline for turning your data into a 
visualization using Bokeh.

"""

import pandas as pd
import numpy as np
from math import pi
import math
import datetime

# Bokeh libraries
from bokeh.io import output_file, output_notebook
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, BoxSelectTool, HoverTool,Legend,FactorRange,LinearColorMapper,CDSView, GroupFilter
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel, TableColumn,DataTable, DateFormatter,RangeSlider
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral11


dir = "C:\\Users\\saura\\Documents\\Clean Records"

# Prepare the data and load it
days = pd.read_csv(dir + "\\IMEI_Description\\" + "Days_Description.csv")
hour = pd.read_csv(dir +  "\\IMEI_Description\\" +"Hour_Description.csv")
#imeidays = pd.read_csv(dir + "\\IMEI_Description\\" + "Days_Description.csv")

time = pd.read_csv(dir + "\\IMEI_Description\\" + "Time_Description.csv")
trip = pd.read_csv(dir + "\\IMEI_Description\\" + "Trip_Description.csv")
dist = pd.read_csv(dir + "\\IMEI_Description\\" + "Distance_Description.csv")

imeidescribe =  pd.read_csv(dir +  "\\IMEI_Description\\" +"IMEI_Description.csv")

cleanrecord = pd.read_csv(dir +  "\\IMEI_Description\\" +"Clean_Record_Summary.csv")

# Output to file
output_file('Summary Screen.html', 
title='VTS Data Description')

# Set up the figure(s) and draw the data
##Week of day description of vehicles in a segement##
cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
days['Day'] = pd.Categorical(days['Day'], categories=cats, ordered=True)
days = days.sort_values('Day')
bins = [0,26,27,28,29,30,51]
days['Week_Intervals'] = pd.cut(days['Week'],bins)
aggregation = {'count' : 'sum'}
weekday = days.groupby(['Week_Intervals','Day']).agg(aggregation).reset_index().sort_values(by = ['Week_Intervals','Day'])
x = weekday['count'].tolist()
#day = weekday['Day'].tolist()

#group = weekday.groupby(by=['Week_Intervals', 'Day'])

#index_cmap = factor_cmap('Day', palette=Spectral11, factors=sorted(weekday.Day.unique()), end=1)

factors = [
            ("0 - 26", "Monday"), ("0 - 26", "Tuesday"),("0 - 26", "Wednesday"),("0 - 26", "Thursday"),("0 - 26", "Friday"),("0 - 26", "Saturday"),("0 - 26", "Sunday"),
            ("27", "Monday"), ("27", "Tuesday"),("27", "Wednesday"),("27", "Thursday"),("27", "Friday"),("27", "Saturday"),("27", "Sunday"),
            ("28", "Monday"), ("28", "Tuesday"),("28", "Wednesday"),("28", "Thursday"),("28", "Friday"),("28", "Saturday"),("28", "Sunday"),
            ("29", "Monday"), ("29", "Tuesday"),("29", "Wednesday"),("29", "Thursday"),("29", "Friday"),("29", "Saturday"),("29", "Sunday"),
            ("30", "Monday"), ("30", "Tuesday"),("30", "Wednesday"),("30", "Thursday"),("30", "Friday"),("30", "Saturday"),("30", "Sunday"),
            ("31 - 52", "Monday"), ("31 - 52", "Tuesday"),("31 - 52", "Wednesday"),("31 - 52", "Thursday"),("31 - 52", "Friday"),("31 - 52", "Saturday"),("31 - 52", "Sunday")]

p1 = figure(x_range=FactorRange(*factors), plot_height=800,plot_width=1600,
           toolbar_location=None,title="Start day and week of vehicle in segemnt",tools = "hover", tooltips=[("Week and Day", "@x")])
p1.left[0].formatter.use_scientific = False
p1.y_range.start = 0
p1.vbar(x=factors, top=x, width=1, alpha=0.5,
        line_color="green")
p1.legend.orientation = "horizontal"
p1.legend.location = "top_right"
p1.legend.click_policy="mute"
p1.xaxis.major_label_orientation = pi/4   # standard styling code
p1.legend.label_text_font_size = "7pt"
p1.xaxis.axis_label_text_font_size = "50pt"
p1.yaxis.axis_label_text_font_size = "50pt"
p1.xaxis.axis_label_text_font = "times"
p1.xaxis.axis_label_text_color = "black"






##Time description of vehicles in a segemnt##
Time = time['Time'].tolist()
count2 = time['Count'].tolist()
percent = [(count2[i] / np.sum(count2)) * 100 for i in range(0,len(count2))]
source2 = ColumnDataSource(data=dict(count2=count2,Time = Time,percent = percent))
p2 = figure(x_range=Time, plot_height= 700,plot_width=800, toolbar_location=None, title="Time taken by vehicle in segment",tools = "hover", tooltips=[ ("Time", "@Time"),("Count", "@count2"),("Percent", "@percent")])
p2.vbar(x='Time', top='count2',width=1,source = source2, legend="Time",
        line_color="green",fill_color=factor_cmap('Time', palette=Spectral11, factors=Time))
p2.left[0].formatter.use_scientific = False
p2.y_range.start = 0
#p2.y_range.end = 260000
p2.legend.orientation = "vertical"
p2.legend.location = "top_right"
p2.legend.click_policy="mute"
p2.xaxis.major_label_orientation = pi/4   # standard styling code
p2.legend.label_text_font_size = "7pt"
p2.xaxis.axis_label_text_font_size = "250pt"
p2.yaxis.axis_label_text_font_size = "250pt"
p2.xaxis.axis_label_text_font = "times"
p2.xaxis.axis_label_text_color = "black"




##Distance description of vehicles in a segment##
Dist = dist['Distance'].tolist()
count3 = dist['Count'].tolist()
percent = [(count3[i] / np.sum(count3)) * 100 for i in range(0,len(count3))]
source3 = ColumnDataSource(data=dict(count3=count3,Dist = Dist,percent = percent))
p3 = figure(x_range=Dist, plot_height=700,plot_width=800, toolbar_location=None, title="Distance Travellled in Km in segment by Vehicle",tools = "hover",tooltips=[ ("Distance", "@Dist"),("Count", "@count3"),("Percent","@percent")])
p3.vbar(x='Dist', top='count3',width=1,source = source3, legend="Dist",
        line_color="green",fill_color=factor_cmap('Dist', palette=Spectral11, factors=Dist))
p3.left[0].formatter.use_scientific = False
p3.y_range.start = 0
#p3.y_range.end = 240000
p3.legend.orientation = "vertical"
p3.legend.location = "top_right"
p3.legend.click_policy="mute"
p3.xaxis.major_label_orientation = pi/4   # standard styling code
p3.legend.label_text_font_size = "7pt"
p3.xaxis.axis_label_text_font_size = "250pt"
p3.yaxis.axis_label_text_font_size = "250pt"
p3.xaxis.axis_label_text_font = "times"
p3.xaxis.axis_label_text_color = "black"



##Hour description of vehicles in a segment##
Hour = hour['Hour'].tolist()
count4 = hour['Count'].tolist()
percent = [(count4[i] / np.sum(count4)) * 100 for i in range(0,len(count4))]
source4 = ColumnDataSource(data=dict(count4=count4,Hour = Hour,percent = percent))
p4 = figure(x_range=Hour, plot_height=700,plot_width=800, toolbar_location=None, title="Start time of vehicle in segment",tools ="hover",tooltips=[ ("Hour", "@Hour"),("Count", "@count4"),("Percent","@percent")])
p4.vbar(x='Hour', top='count4',width=1,source = source4, legend="Hour",
        line_color="blue",fill_color=factor_cmap('Hour', palette=Spectral11, factors=Hour))
p4.left[0].formatter.use_scientific = False
p4.y_range.start = 0
#p4.y_range.end = 100000
p4.legend.orientation = "vertical"
p4.legend.location = "top_right"
p4.legend.click_policy="mute"
p4.xaxis.major_label_orientation = pi/4   # standard styling code
p4.legend.label_text_font_size = "7pt"
p4.xaxis.axis_label_text_font_size = "250pt"
p4.yaxis.axis_label_text_font_size = "250pt"
p4.xaxis.axis_label_text_font = "times"
p4.xaxis.axis_label_text_color = "black"



##Trip count of total IMEI i.e estimated trips of vehicle##
Trip = trip['Trip'].tolist()
count5 = trip['Count'].tolist()
percent = [(count5[i] / np.sum(count5)) * 100 for i in range(0,len(count5))]
source5 = ColumnDataSource(data=dict(count5=count5,Trip = Trip,percent = percent))
p5 = figure(x_range=Trip, plot_height=700,plot_width=800, toolbar_location=None, title="Trip counts in IMEI",tools ="hover",tooltips=[ ("Trip", "@Trip"),("Count", "@count5"),("Percent","@percent")])
p5.vbar(x='Trip', top='count5',width=1,source = source5, legend="Trip",
        line_color="yellow",fill_color=factor_cmap('Trip', palette=Spectral11, factors=Trip))
p5.left[0].formatter.use_scientific = False
p5.y_range.start = 0
#p5.y_range.end = 4000
p5.legend.orientation = "horizontal"
p5.legend.location = "top_right"
p5.legend.click_policy="mute"
p5.xaxis.major_label_orientation = pi/4   # standard styling code
p5.legend.label_text_font_size = "7pt"
p5.xaxis.axis_label_text_font_size = "250pt"
p5.yaxis.axis_label_text_font_size = "250pt"
p5.xaxis.axis_label_text_font = "times"
p5.xaxis.axis_label_text_color = "black"

"""
##Total days travelled by vehicle i.e Days where IMEI has gone##
imeicount = imeidays['Day'].tolist()
count6 = imeidays['count'].tolist()
percent = [(count6[i] / np.sum(count6)) * 100 for i in range(0,len(count6))]
source6 = ColumnDataSource(data=dict(count6=count6,imeicount = imeicount,percent = percent))
p6 = figure(x_range=imeicount, plot_height=700,plot_width=600, toolbar_location=None, title="Total days travelled by vehicle",tools = "hover",tooltips=[ ("Days", "@imeicount"),("Count", "@count6"),("Percent","@percent")])
p6.vbar(x='imeicount', top='count6',width=1,source = source6, legend="imeicount",
        line_color="pink",fill_color=factor_cmap('imeicount', palette=Spectral11, factors=imeicount))
p6.left[0].formatter.use_scientific = False
p6.y_range.start =0
#p6.y_range.end = 520000
p6.legend.orientation = "horizontal"
p6.legend.location = "top_right"
p6.legend.click_policy="mute"
p6.xaxis.major_label_orientation = pi/4   # standard styling code
p6.legend.label_text_font_size = "7pt"
p6.xaxis.axis_label_text_font_size = "250pt"
p6.yaxis.axis_label_text_font_size = "250pt"
p6.xaxis.axis_label_text_font = "times"
p6.xaxis.axis_label_text_color = "black"
"""



##Segement to trip ratio counts##
df = cleanrecord[['Total Trip','Total Segements','Small Segments']]
df['segtotrip'] = 0
df['segtotrip'] = [np.ceil(df['Total Segements'][i] / df['Total Trip'][i]) for i in range(0,len(df)) ]
segmenttotrip = pd.DataFrame(data = {'Ratio':['1','2','3','4','5','More than 5'],
                                    'Count' :[len([i for i in df['segtotrip'] if i == 1])
                                             ,len([i for i in df['segtotrip'] if i == 2])
                                             ,len([i for i in df['segtotrip'] if i == 3])
                                             ,len([i for i in df['segtotrip'] if i == 4])
                                             ,len([i for i in df['segtotrip'] if i == 5])
                                             ,len([i for i in df['segtotrip'] if i > 5])
                                             ]}) 
                                             

ratiocount = segmenttotrip['Ratio'].tolist()
count7 = segmenttotrip['Count'].tolist()
percent = [(count7[i] / np.sum(count7)) * 100 for i in range(0,len(count7))]
source7 = ColumnDataSource(data=dict(count7=count7,ratiocount = ratiocount,percent = percent))
p7 = figure(x_range=ratiocount, plot_height=700,plot_width=600, toolbar_location=None, title="Segment to trip ratio of IMEI",tools = "hover",tooltips=[ ("Ratio", "@ratiocount"),("Count", "@count7"),("Percent","@percent")])
p7.vbar(x='ratiocount', top='count7',width=1,source = source7, legend="ratiocount",
        line_color="green",fill_color=factor_cmap('ratiocount', palette=Spectral11, factors=ratiocount))
p7.left[0].formatter.use_scientific = False
p7.y_range.start =0
#p7.y_range.end = 520000
p7.legend.orientation = "horizontal"
p7.legend.location = "top_right"
p7.legend.click_policy="mute"
p7.xaxis.major_label_orientation = pi/4   # standard styling code
p7.legend.label_text_font_size = "7pt"
p7.xaxis.axis_label_text_font_size = "250pt"
p7.yaxis.axis_label_text_font_size = "250pt"
p7.xaxis.axis_label_text_font = "times"
p7.xaxis.axis_label_text_color = "black"


record = dict(imeidescribe[['Fields', 'Description']])
source = ColumnDataSource(record)
columns = [
        TableColumn(field="Fields", title="Fields"),
        TableColumn(field="Description", title="Description"),
    ]
data_table = DataTable(source=source, columns=columns, width=800, height=600)



# Organize the layout
# Create two panels, one for each conference
IMEI_summary_panel = Panel(child = data_table, title='IMEI Summary')
Days_panel = Panel(child=p1, title='Weekday and number')
Time_panel = Panel(child=p2, title='Time information')
Dist_panel = Panel(child=p3, title='Distance information')
Hour_panel = Panel(child=p4, title='Hour information')
Trip_panel = Panel(child=p5, title='Trip information')
#IMEI_Days_panel = Panel(child=p6, title='Days travelled information')
Trip_Segment_panel = Panel(child=p7, title='Segment to trip Ratio')


# Assign the panels to Tabs
tabs = Tabs(tabs=[IMEI_summary_panel,Days_panel,Time_panel,Dist_panel,Hour_panel,Trip_panel,Trip_Segment_panel])

# Show the tabbed layout
show(tabs)




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
    return d

def latlong_to_coordinate(x,y):
    
    """
    Converting latitude and longitude to x-y coordinate system using a reference lat and long
    
    In this case to project every distance as postitve we use reference latitude and longitude of Lakshwadeep
    
    """
    ref_lat  = 10.076011
    
    ref_long =  73.630345
    
    xdist = haversine((0,ref_lat),(x,y))
    
    ydist = haversine((ref_long,0),(x,y))
    
    return xdist/500,ydist/500


def Visualise_Segment(f,TripID):
    
    """
    
    Graphically show how vehciles move in a segment of a trip
    
    """
    
    df = pd.read_csv(dir + "IMEI_Files\\" +  f[0:15] + ".csv")
    dft = df[df['TripID'] == TripID].reset_index(drop=True)
    dft['Date Time'] = dft['Date Time'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    dft = dft.sort_values(by = ['Date Time']).reset_index(drop=True)
    dft['time'] = pd.Series(range(0,len(dft)))
    dft['X'] = 0
    dft['Y'] = 0
    dft['X'] = [latlong_to_coordinate(dft['Latitude'][i],dft['Longitude'][i])[0] for i in range(0,len(dft))]
    dft['Y'] = [latlong_to_coordinate(dft['Latitude'][i],dft['Longitude'][i])[1] for i in range(0,len(dft))]
    x = dft['X'].tolist()
    y = dft['Y'].tolist()
    time = dft['time'].tolist()    
    output_file('IMEI_Segment.html')
    source = ColumnDataSource(data=dict(x=x,y=y,time = time))
    
    color_mapper = LinearColorMapper(palette='Magma256', low=min(time), high=max(time))
    #color_mapper1 = CategoricalColorMapper(factors = ["0","1"],color = ['blue','red'])
    
    p = figure(plot_height=700,plot_width=1000, toolbar_location=None, title="Vehicle movement in a trip")
    
    p.circle(x='x',y = 'y',size = 20,color={'field': 'time', 'transform': color_mapper},source = source, legend="Segments Of Trip by time")
    
    #p.left[0].formatter.use_scientific = False
    p.legend.orientation = "horizontal"
    p.legend.location = "bottom_right"
    p.legend.click_policy="mute"
    p.xaxis.major_label_orientation = pi/4   # standard styling code
    p.legend.label_text_font_size = "7pt"
    Segment_panel = Panel(child = p, title='Segment plot i.e Vehcle movement vs Time')
    
    tabs = Tabs(tabs=[Segment_panel])

    show(tabs)













'''
# Import reset_output (only needed once) 
#from bokeh.plotting import reset_output

# Use reset_output() between subsequent show() calls, as needed
#reset_output()


# Preview and save 

show(p1)  # See what I made, and save if I like it
show(p2)
show(p3)
'''