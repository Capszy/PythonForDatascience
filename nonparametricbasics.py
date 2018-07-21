import numpy as np
import pandas as pd
from matplot.lib.pyplot as plt
import seaborn as sb
from pylab import rcParams
import scipy
from scipy.stats import spearmanr
rcParams['figure.figsize']=8,4
sb.set_style='whitegrid'

#cars dataset
x=cars[['cyl','vs','am','gear']]
sb.pairplot(x)

#isolating variables

spearman_coefficient,p_value=spearmanr(cyl,vs)
print 'Spearman Rank Correlation Coefficient %0.3f'%(spearmanr_coefficient)


#chisquare test
table=pd.crosstab(cyl,am)
from scipy import chi2_contingency
chi2,p,dof,expected=chi2_contingency(table.values)
print 'Chi-Square Statistic %0.3f p_value %0.3f'%(chi2,p)




