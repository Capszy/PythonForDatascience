import numpy as np
import pandas as pd

#cars dataset
cars.head(15)#15 rows

carb=cars.carb
carb.value_counts()

cars_cat=cars[['cyl','vs','am','gear','carb']]#subset of cars
cars_cat.head()

gears_group=cars_cat.groupby('gear')
gears_group.describe()

cars['group']=pd.Series(cars.gears,dtype='category')
cars['group'].dtypes

cars['group'].value_counts()

pd.crosstab(car['am'],cars['gear'])



