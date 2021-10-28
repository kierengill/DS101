writtenBy = "KierenSinghGill"
dateWritten = "May 11 2021"

import numpy as np
import pandas as pd

dataset = np.genfromtxt('/Users/kierensinghgill/Desktop/Homework/2021 Spring/Intro to DataSci/Functions/F7/sampleImput2.csv')


listOfNans = []
for i in range (0,len(dataset)):
    if np.isnan(dataset[i]):
        listOfNans.append(i)
        
    
        
    