#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 05:20:56 2021

@author: pascalwallisch
"""

# Calculating correlation
import numpy as np # load numpy
howMany = 7 #How long will the random array be?
A = np.random.rand(howMany,2) #Draw the random numbers from a uniform distribution

r = np.corrcoef(A[:,0],A[:,1]) #Calculate Pearson correlation between column 1 and column 2
print('r =',r) #The correlation matrix. So far, so good

#%% Introducing nans into the array by replacing some values with nans
A[2,0] = np.nan 
A[2,1] = np.nan
A[4,0] = np.nan
A[5,1] = np.nan

r = np.corrcoef(A[:,0],A[:,1]) #Calculate Pearson correlation between column 1 and column 2
print('r =',r) #Now there are nans. 
#As soon as there is even one in a calculation like that, the results are also nan
#So we have to handle that. Handling missing data in the form of nans is a major part of the job of data scientists
#How they are handled is critical and there are a lot of intricacies. 
#We'll deal with that later in detail. For now, let's just introduce a simple way to handle nans in numpy
#So you can do the assignment.

#%%
#There are many ways to handle this - for instance, pandas dataframes more readily ignore nans by default (but some of these defaults are not what you want or need)
#Pandas hides a lot of this plumbing from the user, so it is useful to first understand what is going on under the hood, e.g. with numpy. That's what we're doing here.
#You could also use masked arrays in numpy. For now, let's use logic.
temp = np.copy(A) #Make a copy of A
whereAreTheNansAt = np.isnan(temp) #Determine the nan status of entries in temp
print(whereAreTheNansAt)
nanCoordinates = np.where(whereAreTheNansAt==True) #Coordinates in rows and columns
#Here, we will eliminate the entire row of the temporary array if the value in either (or both) column(s) is missing
temp = np.delete(temp,nanCoordinates[0],0) #Delete all rows (axis 0, the 3rd argument) of the input array temp, where there is a nan in a row (from nanCoordinates)
r = np.corrcoef(temp[:,0],temp[:,1]) #Calculate the Pearson correlation between the surviving 4 rows of column 1 and column 2
print('r =',r) #No more nans. Note the value is different than in line 14, as we deleted some rows

#So for now, the advice is to c
#1) reate a temporary array with the values of the two vectors you are correlating at a given time
#2) Then eliminate the nans. 
#3) Then take the correlation of the remaining array (e.g. containing the movies that have a rating from both users in question)
#Then go back to 1), for new data (e.g. a new pair of users or movies to correlate)

#%% We could have also done it directly with logic:
temp = np.copy(A) #Make a copy of A
temp = temp[~np.isnan(temp).any(axis=1)] #Remove all rows where there are missing values
#Then do the same as in lines 41-42:
r = np.corrcoef(temp[:,0],temp[:,1]) #Calculate the Pearson correlation between the surviving 4 rows of column 1 and column 2
print('r =',r) #No more nans. Note the value is different than in line 14, as we deleted some rows
#Same outcome as in 42.
 
