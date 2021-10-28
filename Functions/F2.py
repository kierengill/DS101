writtenBy = "KierenSinghGill"

import numpy as np

#FUNCTION DESCRIPTION
#The function takes in three parameters
    #Parameter 1: An input dataset, either a 1D or 2D numpy array
    #Parameter 2: A flag for which parameter to calculate, where 1 = mean, 2 = SD and 3 = correlation
    #Parameter 3: The window size (the subset of how many numbers of the dataset to compute the parameter indicated in 2 over)
#Assumptions
    #Assume that the input will be a 2D array if the Parameter 2 flag is correlation
    #Assume that the input will be a 1D array if the Parameter 2 flag is mean or sd
    #Assume variables are in columns and measures/numbers in rows
    
def slidWinDescStats(inputDataset,inputFlag,inputWindow):
    if inputFlag == 1:
        resultArray = np.array([])
        for i in range (0,len(inputDataset)):    
            meanAcrossWindow = (np.nanmean(inputDataset[i:inputWindow+i]))
            resultArray = np.append(resultArray, meanAcrossWindow)
            if inputWindow + i == len(inputDataset):
                break
    elif inputFlag == 2:
        resultArray = np.array([])
        for i in range (0,len(inputDataset)):    
            stdAcrossWindow = (np.nanstd(inputDataset[i:inputWindow+i], ddof = 1))
            resultArray = np.append(resultArray, stdAcrossWindow)
            if inputWindow + i == len(inputDataset):
                break
    elif inputFlag == 3:
        resultArray = np.empty([(len(inputDataset)-inputWindow)+1,1])
        transposedArray = np.transpose(inputDataset)
        for i in range (0,len(inputDataset)):    
            r = (np.corrcoef(transposedArray[0][i:inputWindow+i], transposedArray[1][i:inputWindow+i]))[1,0]
            resultArray[i,0]=r
            if inputWindow + i == len(inputDataset):
                break
    return resultArray

B=np.array([[1,1],[3,2],[5,4],[7,3],[9,5]])
A=np.array([1,3,5,7,9])
