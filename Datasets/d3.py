import numpy as np
import pandas as pd

pandaSata = pd.read_csv('/Users/kierensinghgill/Desktop/Homework/2021 Spring/Intro to DataSci/Datasets/movieRatingsDeidentified.csv', delimiter = ',')
sata = np.genfromtxt('/Users/kierensinghgill/Desktop/Homework/2021 Spring/Intro to DataSci/Datasets/movieRatingsDeidentified.csv', delimiter=',', skip_header=1)

#separate headings
headings = pandaSata.columns

def simple_linear_regress_func(data):
    # Load numpy:
    import numpy as np
    # Initialize container:
    temp_cont = np.empty([len(data),5]) # initialize empty container, n x 5
    temp_cont[:] = np.NaN # convert to NaN
    # Store data in container:
    for i in range(len(data)):
        temp_cont[i,0] = data[i,0] # x 
        temp_cont[i,1] = data[i,1] # y 
        temp_cont[i,2] = data[i,0]*data[i,1] # x*y
        temp_cont[i,3] = data[i,0]**2 #x^2
        temp_cont[i,4] = data[i,1]**2 # y^2
    # Compute numerator and denominator for m:
    m_numer = len(data)*sum(temp_cont[:,2]) - sum(temp_cont[:,0])*sum(temp_cont[:,1])
    m_denom = len(data)*sum(temp_cont[:,3]) - (sum(temp_cont[:,0]))**2
    m = m_numer/m_denom
    # Compute numerator and denominator for b:
    b_numer = sum(temp_cont[:,1]) - m * sum(temp_cont[:,0])
    b_denom = len(data)
    b = b_numer/b_denom
    # Compute r^2:
    temp = np.corrcoef(data[:,0],data[:,1]) # pearson r
    r_sqr = temp[0,1]**2 # L-7 weenie it (square)
    # Output m, b & r^2:
    output = np.array([m, b, r_sqr])
    return output 


def normalizedError(inputData, flag):
    yHat = inputData[:,0]
    y = inputData[:,1]
    n = len(y)
    residuals = (abs(yHat - y))**flag
    sumOfResiduals = int(np.sum(residuals))
    result = (sumOfResiduals/n)**(1/flag)
    return result

star1v2 = sata[:,156:158]

star1v2cleaned = star1v2[~np.isnan(star1v2).any(axis=1)]
#1625 for star wars 1 and 2

star1v2output = simple_linear_regress_func(star1v2cleaned)
#0.808275 - Betas
#0.652412 - R^2


star1vtitanic = sata[:,156:198:41]

star1vtitaniccleaned = star1vtitanic[~np.isnan(star1vtitanic).any(axis=1)]
#1612

star1vtitanicoutput = simple_linear_regress_func(star1vtitaniccleaned)
#0.235529 - Betas
#0.062494 - R^2

star1v2outputRMSE = normalizedError(star1v2cleaned, 2)


#predict star wars 1 from star wars 2
#0.67

s1 = sata[:,156] 
s2 = sata[:,157] 

sArr = np.vstack((s2,s1)).T
sArr = sArr[~np.isnan(sArr).any(axis=1)]

r1 = simple_linear_regress_func(sArr)
yHat = sArr[:,0]* r1[0] + r1[1]
sRMSE = np.vstack((yHat,sArr[:,1])).T
RMSE1 = normalizedError(sRMSE, 2)
#0.6670255444307038


#predict titanic from star wars 1
#1.05

s1 = sata[:,156] 
t1 = sata[:,197] 

tArr = np.vstack((s1,t1)).T
tArr = tArr[~np.isnan(tArr).any(axis=1)]

r2 = simple_linear_regress_func(tArr)
yHat = tArr[:,0]* r2[0] + r2[1]
tRMSE = np.vstack((yHat,tArr[:,1])).T
RMSE2 = normalizedError(tRMSE, 2)
#1.0499320549917768










