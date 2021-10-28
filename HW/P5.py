import numpy as np
from scipy import stats

klonopin = np.genfromtxt('/Users/kierensinghgill/Desktop/klonopin.txt', delimiter = '', skip_header = 0)
rats = np.genfromtxt('/Users/kierensinghgill/Desktop/rats.txt', delimiter = '', skip_header = 0)
births = np.genfromtxt('/Users/kierensinghgill/Desktop/births.txt', delimiter = '', skip_header = 0)
blogData = np.genfromtxt('/Users/kierensinghgill/Desktop/blogData.txt', delimiter = '', skip_header = 0)
happiness = np.genfromtxt('/Users/kierensinghgill/Desktop/happiness.txt', delimiter = '', skip_header = 0)
socialstress = np.genfromtxt('/Users/kierensinghgill/Desktop/socialstress.txt', delimiter = '', skip_header = 0)

dose0 = klonopin[:25,0]
dose1 = klonopin[25:50,0]
dose2 = klonopin[50:75,0]
dose3 = klonopin[75:100,0]
dose4 = klonopin[100:125,0]
dose5 = klonopin[125:150,0]

f,p = stats.f_oneway(dose0, dose1, dose2, dose3, dose4, dose5)

def catCounter(inputData):
    uniqueValues = []
    frequency = []
    for i in inputData:
        if i not in uniqueValues:
            uniqueValues.append(i)
    uniqueValues.sort()
    for value in uniqueValues:
        counter = 0
        for data in inputData:
            if data == value:
                counter += 1
        frequency.append(counter)
    
    output = np.array([uniqueValues,frequency])
    output = output.transpose()    
        
    return output

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot as meansPlot

# df = pd.read_csv('/Users/kierensinghgill/Desktop/klonopin.csv')
# model = ols('anxiety ~ dose', data=df).fit()
# anova_table = sm.stats.anova_lm(model, typ=1)
# print(anova_table) #Show the ANOVA table
# fig = meansPlot(x=df['dose'], trace=df['dose'], response=df['anxiety'])

# df = pd.read_csv('/Users/kierensinghgill/Desktop/rats.csv')
# model = ols('rat ~ companion + exercise + companion:exercise', data=df).fit()
# anova_table = sm.stats.anova_lm(model, typ=2)
# print(anova_table) #Show the ANOVA table
# fig = meansPlot(x=df['companion'], trace=df['exercise'], response=df['rat'])

# df = pd.read_csv('/Users/kierensinghgill/Desktop/socialstress.csv')
# model = ols('cortisol ~ sleep + classes + disposition + setting + sleep:classes + sleep:disposition + sleep:setting + classes:disposition + classes:setting + disposition:setting + classes:setting:disposition + classes:setting:sleep + setting:sleep:disposition + classes:disposition:sleep + classes:setting:sleep:disposition', data=df).fit()
# anova_table = sm.stats.anova_lm(model, typ=1)
# print(anova_table) #Show the ANOVA table
# # fig = meansPlot(x=df['sleep'], trace=df['setting'], response=df['cortisol'])

# explained = 412.806250 + 436.195630 + 20.199376 + 5.668255 + 2.009539 + 0.002647 + 1.097118 + 19.323960 + 2.799989 + 521.790279 + 1.466072 + 11.862391 + 20.341150 + 8.772300 + 8.246320
# total = explained + 1896.762474
# print (explained/total)

#print(catCounter(blogData))
#qq = stats.chisquare([4884, 5378, 6599, 6420, 7255, 1314, 9456, 10180, 10965, 10932, 13278, 11259], f_exp=[9145.5, 9145.5, 9145.5, 9145.5, 9145.5, 9145.5, 9145.5, 9145.5, 9145.5, 9145.5, 9145.5, 9145.5])


x = catCounter(births)
actual = x[:,1]

y = []
expected = 100.0712328767
for i in range(365):
    y.append(expected)

y=np.array(y)
z = stats.chisquare(actual, f_exp=y)
