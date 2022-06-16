# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:39:18 2019

@author: saura
"""


import pandas as pd
import numpy as np
from math import pi

# Bokeh libraries
from bokeh.io import output_file, output_notebook
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, BoxSelectTool, HoverTool,Legend,FactorRange,CategoricalColorMapper,LinearColorMapper
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel, TableColumn,DataTable, DateFormatter
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral11
from bokeh.models.markers import Asterisk
f = "357713009999993.csv"
df = pd.read_csv("C:\\Users\\saura\\Documents\\Clean Data\\IMEI_Files\\" + f)
dft = df[df['TripID'] == 54].reset_index(drop=True)
dft = dft.sort_values(by = ['Date Time']).reset_index(drop=True)
dft['time'] = pd.Series(range(0,len(dft)))
x = dft['Latitude'].tolist()
y = dft['Longitude'].tolist()
time = dft['time'].tolist()


output_file('IMEI.html')
source = ColumnDataSource(data=dict(x=x,y=y,time = time))

color_mapper = LinearColorMapper(palette='Magma256', low=min(time), high=max(time))
#color_mapper1 = CategoricalColorMapper(factors = ["0","1"],color = ['blue','red'])

p = figure(plot_height=700,plot_width=1000, toolbar_location=None, title="Trip counts in IMEI")

p.circle(x='x',y = 'y',size = 20,color={'field': 'time', 'transform': color_mapper},source = source, legend="Trip")

#p.left[0].formatter.use_scientific = False


show(p)


dft['unique'] = [dft['SegmentID'][i] + 1 for i in range(0,len(dft))]

ggplot(dft, aes(x='Latitude', y='Longitude',size = 'unique',color = "SegmentID")) +\
    geom_point() +\
    scale_color_brewer(type='diverging', palette=4) +\
    xlab("Latitude") + ylab("Longitude") + ggtitle("Segment in trips")












import seaborn as sns; sns.set(color_codes=True)
g = sns.catplot(x="Latitude", y="Longitude", hue="SegmentID", data=dft, sharex=False, sharey=False)
i = 10
 pd.cut(dft['SegmentID'], [0, 20000, 40000, 60000, 80000, 1000000], labels=sizes)
while i < len(dft):
    dt = dft[0:i].reset_index(drop=True)
    g = sns.catplot(x="Latitude", y="Longitude", hue="SegmentID", data=dt, sharex=False, sharey=False)
    i = i + 10

g.set_xticklabels(range(np.min(dft['Latitude']),np.max(dft['Longitude']),0.5))
g.set_yticklabels(np.unique(dft['Longitude']))
fig = g.fig
ax = g.axes[0,0]
scatters = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
txt = ax.text(0.1,0.9,'frame=0', transform=ax.transAxes)


def animate(i):
    for c in scatters:
        # do whatever do get the new data to plot
        x = np.random.random(size=(50,1))*50
        y = np.random.random(size=(50,1))*10
        xy = np.hstack([x,y])
        # update PathCollection offsets
        c.set_offsets(xy)
    txt.set_text('frame={:d}'.format(i))
    return scatters+[txt]


# initialize samples
sampleStats = []

plt.tight_layout()

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=1, blit=True)














# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:17:29 2019

@author: saura
"""

from bokeh.io import curdoc,output_file
from bokeh.layouts import column
from bokeh.plotting import figure,show
from bokeh.models import Button, ColumnDataSource, ColorBar
from bokeh.palettes import Spectral6
from bokeh.transform import linear_cmap

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
import matplotlib
import numpy as np


f = "357713009999993.csv"
df = pd.read_csv("C:\\Users\\saura\\Documents\\Clean Data\\IMEI_Files\\" + f)
dft = df[df['TripID'] == 54].reset_index(drop=True)


output_file("animation.html")


i = 10
x = dft['Latitude'][0:i]
y = dft['Latitude'][0:i]
z = dft['SegmentID'][0:i]

#Use the field name of the column source
mapper = linear_cmap(field_name='x', palette=Spectral6 ,low=min(y) ,high=max(y))

source = ColumnDataSource(dict(x=x, y=y, z=z))

p = figure(plot_width=300, plot_height=300, title="Linear Color Map Based on Y")
p.circle(x='x', y='y', line_color=mapper,color=mapper, fill_alpha=1, size=12, source=source)



glyph = Asterisk(x="x", y="y", line_color="#f0027f", fill_color=None, line_width=2)
plot.add_glyph(source, glyph)












color_bar = ColorBar(color_mapper=mapper['transform'], width=8,  location=(0,0))

b = Button()

def update():
    i = 10
    x = dft['Latitude'][0:i].reset_index(drop=True)
    y = dft['Latitude'][0:i].reset_index(drop=True)
    new_z = dft['SegmentID'][0:i].reset_index(drop=True)
    i =i + 10   

    source.data = dict(x=x, y=y, z=new_z)

b.on_click(update)


curdoc().add_root(column(b, p))

show(p)



















i=10
while i < len(dft):
    
        x=dft['Latitude'][0:i]
        y=dft['Longitude'][0:i]
        p = figure(plot_width=400, plot_height=400,)
        p.circle('x', 'y', size=5)
        show(p)
        i = i + 10
        a = raw_input() # just press any key to continue
        p.line(range(10),range(10))
        save(p)

callback = CustomJS(args=dict(source=source), code="""
        // get data source from Callback args
        var data = source.data;
        var geometry = cb_data['geometries'];



        /// calculate x and y
        var x = geometry[0].x
        var y = geometry[0].y


        /// update data source
        data['x'].push(x);
        data['y'].push(y);
        data['color'].push('gold')
        
        // trigger update of data source
        source.trigger('change');
    """)



taptool = TapTool(callback=callback)
p = figure(plot_width=400, plot_height=400,)
p.circle('x', 'y', size=20, source=source)
show(p)

