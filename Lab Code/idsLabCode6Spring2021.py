# Overfitting, Cross Validation and Regularization Lab
# Code by Pascal Wallisch and Stephen Spivack

#%% 1) Overfitting
# Modeling error that occurs when a function fits the set of data points too well
# 1 Sample points from a curve
# 2 Fit a polynomial to those points until we overfit
# 3 Leave one out cross-validation to demonstrate the escalation of RMSE when overfitting

# Import libraries:
import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters:
noiseMagnitude = 2 # how much random noise is there?
numData = 8 # how many measurements (samples) of the signal?
numPoints = 1001 
leftRange = -5
rightRange = 5
x = np.linspace(leftRange,rightRange,num=numPoints) # determine the location of evenly spaced points from -5 to 5 to use as an x-base

# Determine the functional relationship between x and y in reality (ground truth):
sig = 1 # user determines whether the signal is quadratic (1) or cubic (2)
if sig == 1:
    y1 = x**2 # quadratic function
elif sig == 2:
    y1 = x**3 # cubic function
    
# Compute signal plus noise:
y = y1 + noiseMagnitude * np.random.normal(0,1,len(x)) # signal + noise

# Plot data:
plt.figure(1)
plt.plot(x,y1,color='blue',linewidth=5)
plt.xlabel('X') 
plt.ylabel('Y')  
plt.title('Ground Truth')

plt.figure(2)
plt.plot(x,y,color='blue',linewidth=1)
plt.xlabel('X') 
plt.ylabel('Y')  
plt.title('Signal plus noise')

#Ground truth with noise in one plot
plt.plot(x,y,color='blue',linewidth=1)
plt.plot(x,y1,color='black',linewidth=5)

#%% Determine the location of the sampling (measuring) points 

# Randomly draw points to sample:
samplingIndices = np.random.randint(1,len(x),numData) # random points, from anywhere on the signal

# Plot data as a subsample of the noisy signal:
plt.figure(2)
plt.plot(x,y,color='blue',linewidth=1)
plt.plot(x,y1,color='black',linewidth=5)
plt.plot(x[samplingIndices],y[samplingIndices],'ro','col','c1',markersize=4)
plt.xlim(-5,5) # keep it on the same x-range as before

# Note: Parabola doesn't fit perfectly because there is noise (measurement error). We are
# overfitting to noise. The more noise, the worse this effect is
# In real life, all measurements are contaminated with noise, so overfitting
# to noise is always a concern.

#%% (Over)fitting successive polynomials and calculating RMSE at each point

rmse = np.array([]) # capture RMSE for each polynomial degree
for ii in range(numData): # loop through each sampling point
    plt.subplot(2,4,ii+1)
    numDegrees = ii+1
    p = np.polyfit(x[samplingIndices],y[samplingIndices],numDegrees) # minimizes squared error
    y_hat = np.polyval(p,x) # evaluate the polynomial at specific values
    plt.plot(x,y_hat,color='blue',linewidth=1)
    plt.plot(x[samplingIndices],y[samplingIndices],'ro',markersize=3)
    error = np.sqrt(np.mean((y[samplingIndices] - y_hat[samplingIndices])**2))
    plt.title('Degrees: {}'.format(numDegrees) + ', RMSE = {:.3f}'.format(error))
    rmse = np.append(rmse,error) # keep track of RMSE - we will use this later
    
#%% Plotting RMSE of the training set as function of polynomial degree
plt.plot(np.linspace(1,numPoints,8),rmse)
plt.title('Apparent RMSE as a function of degree of polynomial')
plt.xlabel('Degree of polynomial')
plt.ylabel('RMSE')

#%% Leave one out procedure to cross-validate the number of terms in the
# model. Note: We randomly pick one of the test points to use to calculate
# the RMSE with. We use the other data points to fit the model
# This method is called "leave one out" and is very computationally
# expensive, as one has to fit the model n-1 times

# Initialize parameters:
numRepeats = 100 # Number of samples - how often are we doing this?
rmse = np.zeros([numRepeats,numData-1]) # Reinitialize RMSE (100x7)
# For each polynomial degree, 100x we are going to randomly pick one of
# the points from the set of 8 and compute the RMSE
# We are then going to fit the model from the remaining (7) points

# Compute RMSE on test set:
for ii in range(numRepeats): # Loop from 0 to 99
    testIndex = np.random.randint(0,numData,1) # Randomize test index - pick randint from 0 to 7
    testSet = samplingIndices[testIndex] # Find the test set
    trainingSet = np.copy([samplingIndices]) # Make copy of sampling indices
    trainingSet = np.delete(trainingSet,testIndex) # Delete the test subset
    for jj in range(numData-1): # Loop from 0 to 7 - for each poly degree
        numDegrees = jj+1 # degrees are from 1 to 8
        p = np.polyfit(x[trainingSet],y[trainingSet],numDegrees)
        yHat = np.polyval(p,x) 
        # Calculate RMSE with the test set:
        rmse[ii,jj] = np.sqrt(np.mean((y[testSet] - yHat[testSet])**2))

# Plot data:
plt.plot(np.linspace(1,numData-1,7),np.mean(rmse,axis=0))
plt.title('Real RMSE as a function of degree of polynomial')
plt.xlabel('Degree of polynomial')
plt.ylabel('RMSE measured only at points left out from building model')   

# The solution? Where RMSE is minimal
solution = np.amin(np.mean(rmse,axis=0)) # value
index = np.argmin(np.mean(rmse,axis=0)) # index
print('The RMSE is minimal at polynomial of degree: {}'.format(index+1)) 

#Note - the console will give you warnings that the polyfit is poorly conditioned sometimes. 
#That's another dead giveaway that you are overfitting. Too many parameters, not enough data.

#%% Revisiting multiple regression
# Predicting class performance from SAT math, SAT verbal, hours studied, GPA, 
# appreciation of statistics, fear of math, and fear of teacher
# There are 200 students in this case.

# 0. Load libraries:
from sklearn import linear_model

# 1. Load data:
x = np.genfromtxt('mRegDataX.csv',delimiter=',') # satM satV hoursS gpa appreciation fearM fearT
y = np.genfromtxt('mRegDataY.csv',delimiter=',') # outcome: class score

# 2. Doing the full model and calculating the yhats:
regr = linear_model.LinearRegression()
regr.fit(x,y) # use fit method 
betas = regr.coef_ 
yInt = regr.intercept_
yHat = betas[0]*x[:,0] + betas[1]*x[:,1] + betas[2]*x[:,2] + betas[3]*x[:,3] + betas[4]*x[:,4] + betas[5]*x[:,5] + betas[6]*x[:,6] + yInt
       
# 3. Scatter plot between predicted and actual score of full model:
r = np.corrcoef(yHat,y)
plt.plot(yHat,y,'o',markersize=5)
plt.xlabel('Predicted grade score')
plt.ylabel('Actual grade score')
plt.title('R: {:.3f}'.format(r[0,1])) 

# 4. Splitting the dataset for cross-validation:
x1 = np.copy(x[0:100,:])
y1 = np.copy(y[0:100])
regr = linear_model.LinearRegression()
regr.fit(x1,y1) 
betas1 = regr.coef_ 
yInt1 = regr.intercept_

x2 = np.copy(x[100:,:])
y2 = np.copy(y[100:])
regr = linear_model.LinearRegression()
regr.fit(x2,y2) 
betas2 = regr.coef_ 
yInt2 = regr.intercept_

# 5) Cross-validation. Using the betas from the first (training) dataset, but
#    measuring the error with the second (test) dataset
yHat1 = betas1[0]*x1[:,0] + betas1[1]*x1[:,1] + betas1[2]*x1[:,2] + betas1[3]*x1[:,3] + betas1[4]*x1[:,4] + betas1[5]*x1[:,5] + betas1[6]*x1[:,6] + yInt1
yHat2 = betas1[0]*x2[:,0] + betas1[1]*x2[:,1] + betas1[2]*x2[:,2] + betas1[3]*x2[:,3] + betas1[4]*x2[:,4] + betas1[5]*x2[:,5] + betas1[6]*x2[:,6] + yInt1
rmse = np.sqrt(np.mean((yHat2 - y2)**2))

