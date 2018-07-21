import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import scipy
from scipy import stats

#using cars dataset
cars.head()

cars.sum()#for column
cars.sum(axis=1)#for rows
cars.median()
cars.mean()
cars.max()
mpg=cars.mpg
mpg.idmax()#row id of max value

cars.std()#standard deviation
cars.var()#variance
dear=cars.gear
gear.value_counts()#to get unique values

cars.describe()# to get all basic info



