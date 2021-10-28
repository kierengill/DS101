writtenBy = "KierenSinghGill"

import numpy as np

def medAbsDev(data):
    median = np.median(data)
    allDeviations = []
    for number in data:
        deviation = abs(number - median)
        allDeviations.append(deviation)
    
    medianAbsoluteDeviation = np.median(allDeviations)
    return medianAbsoluteDeviation

