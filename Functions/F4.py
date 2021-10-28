writtenBy = "KierenSinghGill"
dateWritten = "April 8 2021"

#FUNCTION DESCRIPTION
    #This function that counts the absolute frequency of how often a given number 
    #(usually representing a category) is present in an array
#This function takes in one parameter:
    #inputData: assume input is a 1D numpy array by default
#The function returns:
    #The unique values sorted in ascending order
    #Their frequency
    #These are returned in a 2D array
    
import numpy as np

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