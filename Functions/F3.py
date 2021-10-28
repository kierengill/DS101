writtenBy = "KierenSinghGill"

#FUNCTION DESCRIPTION
#The function takes in two parameters
    #inputData: 2D numpy array
    #flag: a flag by which power to normalize the error
        #for instance 1 for the mean absolute error, 2 for the RMSE, 
        #3 for the cubic root mean cubed error and so on.
#Output: Normalized error as a single, scalar, real number

import numpy as np


def normalizedError(inputData, flag):
    yHat = inputData[:,0]
    y = inputData[:,1]
    n = len(y)
    residuals = (abs(yHat - y))**flag
    sumOfResiduals = int(np.sum(residuals))
    result = (sumOfResiduals/n)**(1/flag)
    return result