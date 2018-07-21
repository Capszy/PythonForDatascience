import numpy as np
from numpy.random import randn
import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb

rcParams['figure.figsize']=5,4
sb.set_style='whitegrid'
x=range(1,10)
y=[1,2,3,4,0.5,4,3,2,1]

wide=(0.5,0.5,0.5,0.9,0.9,0.5,0.5,0.5,0.5)
color=['salmon']
#plt.bar(x,y,width=wide,color=color,align='center')

#linechart
#color_theme=['darkgray','lightsalmon','powderblue']
#df.plot(color=color_theme)

z=[1,2,3,4,0.5]
color_theme=['#A9A9A9','#FFA07A','#B0E0E6','#FF34C4','#BDB768']
plt.pie(z,colors=color_theme)
plt.show()


#plt.plot(x,y,ls='steps',lw=5)
#plt.plot(x1,y1,ls='--',lw=10)

#plt.plot(x,y,marker='1',mew=20)
#plt.plot(x1,y1,marker='+',mew=15)