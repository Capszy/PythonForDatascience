import numpy as np
import pandas as pd
from pandas import Series,DataFrame

series_obj=Series(np.arange(8),index=['row 1','row 2','row 3','row 4','row 5','row 6','row 7','row 8'])
print(series_obj[[0,2,7]])

np.random.seed(25)
DF_Obj=DataFrame(np.random.rand(36).reshape(6,6))
print(DF_Obj.iloc[[0,1],[0,1]])
print(DF_Obj>0.2)
print(series_obj[series_obj>6])