import numpy as np
import pandas as pd
from scipy import stats

pandaSata = pd.read_csv('/Users/kierensinghgill/Desktop/Homework/2021 Spring/Intro to DataSci/Datasets/movieRatingsDeidentified.csv', delimiter = ',')
sata = np.genfromtxt('/Users/kierensinghgill/Desktop/Homework/2021 Spring/Intro to DataSci/Datasets/movieRatingsDeidentified.csv', delimiter=',', skip_header=1)

#separate headings
headings = pandaSata.columns


k1 = sata[:,109] 
k2 = sata[:,110]
k3 = sata[:,136] 


Arr = np.vstack((k1,k2,k3)).T
Arr = Arr[~np.isnan(Arr).any(axis=1)]

mean1 = (np.nanmean(k1))
mean2 = (np.nanmean(k2))

t1,p1 = stats.ttest_ind(Arr[:,0],Arr[:,1])
t2,p2 = stats.ttest_rel(Arr[:,0],Arr[:,1])
t3,p3 = stats.ttest_ind(Arr[:,0],Arr[:,2])

u1,p4 = stats.mannwhitneyu(Arr[:,1],Arr[:,2])

median1 = np.nanmedian(Arr[:,0])
median2 = np.nanmean(Arr[:,2])

#What are the degrees of freedom for an *independent samples* t-test between Kill Bill 1 and Kill Bill 2? 
    #2620
    
#Doing an independent samples t-test between ratings from Kill Bill 1 and Kill Bill 2, what is the p-value?
    #0.053021327459578244
    
#When using a paired samples t-test, is there a significant difference between the ratings for Kill Bill 1 
#and Kill Bill 2?
    #Yes, we can reject the null hypothesis
    
#Doing an independent samples t-test between ratings from Kill Bill 1 and Kill Bill 2, what is the t-value?
    #1.9356232385154668

#Doing an independent samples t-test between ratings from Kill Bill 1 and Pulp Fiction, 
#what is the absolute value of the t-value?
    #8.4683711289626

#What are the degrees of freedom for a *paired samples* t-test between Kill Bill 1 and Kill Bill 2?
    #1309

#Doing a paired samples t-test between ratings from Kill Bill 1 and Kill Bill 2, what is the p-value?
    #5.646322558291671e-06

#When using a Mann-Whitney U-test, is there a significant difference between the ratings for Kill Bill 2 
#and Pulp Fiction?
    # Yes, we can reject the null hypothesis
    
#Doing a paired samples t-test between ratings from Kill Bill 1 and Kill Bill 2, what is the t-value?
    #4.558001060041254
    
#When using an independent samples t-test, is there a significant difference between the ratings 
#for Kill Bill 1 and Pulp Fiction?
    #Yes, we can reject the null hypothesis
    
#When using an independent samples t-test, is there a significant difference between the ratings 
#for Kill Bill 1 and Kill Bill 2?
    #No, we fail to reject the null hypothesis

#Doing an independent samples t-test between ratings from Kill Bill 1 and Pulp Fiction, what is the p-value?
    #4.083848614270736e-17
    
#How many users have rated all 3 of these movies?
    #1311

#What is the median rating of Kill Bill 1 in this population (of people who have rated all 3 movies)?
    #3.5

#What is the mean rating of Pulp Fiction in this population (of people who have rated all 3 movies)? 
    #3.403203661327231
    

    
    
    