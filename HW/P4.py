#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:34:47 2021

@author: kierensinghgill
"""

import numpy as np
from scipy import stats

Sadex1 = np.genfromtxt('/Users/kierensinghgill/Desktop/Homework/2021 Spring/Intro to DataSci/HW/Sadex1.txt')
Sadex2 = np.genfromtxt('/Users/kierensinghgill/Desktop/Homework/2021 Spring/Intro to DataSci/HW/Sadex2.txt')
Sadex3 = np.genfromtxt('/Users/kierensinghgill/Desktop/Homework/2021 Spring/Intro to DataSci/HW/Sadex3.txt')
Sadex4 = np.genfromtxt('/Users/kierensinghgill/Desktop/Homework/2021 Spring/Intro to DataSci/HW/Sadex4.txt')

Sadex1 = Sadex1.transpose()
Sadex2 = Sadex2.transpose()
Sadex3 = Sadex3.transpose()
Sadex4 = Sadex4.transpose()

G1 = Sadex1[0,:30]
G2 = Sadex1[0,30:]
mean1 = (np.nanmean(G1))
mean2 = (np.nanmean(G2))

t1,p1 = stats.ttest_ind(G1,G2)

G1 = Sadex2[0,:]
G2 = Sadex2[1,:]
mean1 = (np.nanmean(G1))
mean2 = (np.nanmean(G2))

t1,p1 = stats.ttest_rel(G1,G2)

G1 = Sadex3[0,:90]
G2 = Sadex3[0,90:]
mean1 = (np.nanmean(G1))
mean2 = (np.nanmean(G2))

t1,p1 = stats.ttest_ind(G1,G2)

G1 = Sadex2[0,:]
G2 = Sadex2[1,:]
mean1 = (np.nanmean(G1))
mean2 = (np.nanmean(G2))
meanDiff = mean1-mean2
t1,p1 = stats.ttest_rel(G1,G2)



#21: 2.9
#22: 58
#23: 30
#24: 0.21
#25: We have no evidence to believe that the drug works because p > 0.05
#26: 30
#27: 2.9
#28: We believe that the drug works because the difference is highly significant at p < 0.01
#29: We believe that the drug works because the difference is significant at p < 0.05
#30: We believe that the effect of the drug is statistically significant because p < 0.05 
    #but not clinically significant because the scores improve by less than 5 points, on average.
