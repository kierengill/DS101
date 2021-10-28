writtenBy = "KierenSinghGill"
dateWritten = "May 03 2021"

#FUNCTION DESCRIPTION
    #This function determines the bounds of an empirical distribution
#This function takes in 2 inputs:
    #dataset: A variable that represents the dataset/distribution/sample
    #bounds: The probability mass bounds that define the center of the distribution 
        #(e.g. 95, 99 or 50) – in contrast to the tails.
#This function returns:
    #The lower bound (where the left tail starts)
    #The upper bound (where the right tail starts)
#Assumptions:
    #The first input argument can be anything – a list, dataframe or numpy array 
    #containing real numbered values, but you can assume it to be a 1D numpy array 
    #with arbitrary length, by default.
    #The second input argument should be a real number from 0.01 to 99.99 
    #that determines the location of the bounds (where the tails start).

import numpy as np

def empiricalSampleBounds(dataset, bounds):
    sata = np.sort(dataset)
    length = len(sata)
    percentile = length/100.0
    p_mass = (100 - bounds)/2
    upper = 100 - p_mass
    lower = 0 + p_mass
    upper = int(upper * percentile) - 1
    lower = int(lower * percentile) - 1
    lowerBound = sata[lower]
    upperBound = sata[upper]
    return (lowerBound, upperBound)