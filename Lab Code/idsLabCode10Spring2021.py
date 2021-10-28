# Python Session 10
# Effect Size and Power Lab
# Code by Pascal Wallisch and Stephen Spivack

#%% 0. Initialize

# Import libraries:
import random 
import numpy.matlib # for repmat function (line 185)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Set RGN seed to system clock:
random.seed() 

#%% 1. Distributions of p values assuming true vs. false null hypothesis (H0)

# Initialize variables:
sampleSize = 2000
numReps = int(1e4) # Number of repetitions in our simulation
meanDifference = 0.25 # Actual difference in sample means. Try 0 and other values.

# Draw from a random normal distribution with zero mean:
sata1 = np.random.normal(0,1,[sampleSize,numReps])

# Our 2nd sample. Same distribution, different mean:
sata2 = np.random.normal(0,1,[sampleSize,numReps]) + meanDifference

# Run a t-test, a lot of times:
t = np.empty([numReps,1]) # initialize empty 2D array for t
t[:] = np.NaN # then convert to NaN
p = np.empty([numReps,1]) # initialize empty 2D array for p
p[:] = np.NaN # then convert to NaN
for i in range(numReps): # loop through each rep
    t[i],p[i] = stats.ttest_ind(sata1[:,i],sata2[:,i]) # do the t-test
    
# Plot the data:
plt.hist(p,100)
plt.xlabel('p-value')
plt.ylabel('frequency')

#%% 2. Mean differences, effect sizes and significance
# Effect sizes as salvation?
# Central take home message: The same p value can correspond to dramatically
# different effect sizes.

# A. Initialize variables:
numReps = 500 # Number of repetitions in our simulation
sata = np.empty([numReps,3,2]) # Initialize empty 3D array for sata
sata[:] = np.NaN # then convert to NaN 
PvsE = np.empty([numReps,5]) # Initialize empty 2D array for PvsE
PvsE[:] = np.NaN # then convert to NaN

# B. Generate and analyze sata:
for i in range(numReps): # loop through each rep
    p = 0 # set p to 0
    while abs(p - 0.04) > 0.01: # Find datasets that are just about significant
        temp = np.random.normal(0,1,[3,2]) # Draw n = 3 (mu = 0, sigma = 1)
        t,p = stats.ttest_rel(temp[:,0],temp[:,1]) # paired t-test
    sata[i] = temp # store temp in sata array
    
    # sample size:
    PvsE[i,0] = len(sata[i,:,:]) # take the length of the z-stack dimension
    
    # significance level:
    t,p = stats.ttest_rel(sata[i,:,0],sata[i,:,1]) # paired t-test
    PvsE[i,1] = p 
    
    # effect size (computing cohen's d by hand):
    mean1 = np.mean(sata[i,:,0]) # mean of sample 1
    mean2 = np.mean(sata[i,:,1]) # mean of sample 2
    std1 = np.std(sata[i,:,0]) # std of sample 1
    std2 = np.std(sata[i,:,1]) # std of sample 2
    n1 = len(sata[i,:,0]) # size of sample 1
    n2 = len(sata[i,:,1]) # size of sample 2
    numerator = abs(mean1-mean2) # absolute value of mean difference
    denominator = np.sqrt((std1**2)/2 + (std2**2)/2) # pooled std
    d = numerator/denominator
    PvsE[i,2] = d
    
    # mean differences:
    PvsE[i,3] = abs(np.mean(sata[i,:,0]) - np.mean(sata[i,:,1]))
    
    # pooled standard deviation:
    PvsE[i,4] = np.sqrt((std1**2)/2 + (std2**2)/2)
    
# C. Plot it:
plt.subplot(2,3,1)
plt.hist(PvsE[:,1],20)
plt.title('p value')
plt.subplot(2,3,2)
plt.hist(PvsE[:,2],20)
plt.title('cohens d')
plt.subplot(2,3,3)
plt.hist(PvsE[:,3],20)
plt.title('mean diff')
plt.subplot(2,3,4)
plt.hist(PvsE[:,4],20)
plt.title('pooled sd')
plt.subplot(2,3,5)
plt.plot(PvsE[:,2],PvsE[:,3],'o',markersize=.5)
plt.xlabel('cohens d')
plt.ylabel('abs mean diff')
plt.subplot(2,3,6)
plt.plot(PvsE[:,2],PvsE[:,4],'o',markersize=.5)
plt.xlabel('cohens d')
plt.ylabel('pooled sd')

#%% 3. PPV - "positive predictive value" - what we actually want to know
# p that something that is significant is true
# 1 case

# Initialize variables:
alpha = 0.05 # fisher's choice
beta = 0.2 # classic choice
R = 0.5 # we want to know. this is our prior belief
ppv = ((1-beta)*R)/(R-beta*R+alpha)

# What if we are agnostic - let's explore all Rs:
R = np.linspace(0,1,101) # 0 to 1 in .01 increments
ppv = np.empty([len(R),1]) # initialize empty array
ppv[:] = np.NaN # convert to NaN
for i in range(len(R)): # loop through each R
    ppv[i] = ((1-beta)*R[i])/(R[i]-beta*R[i]+alpha)
    
# Plot it:
plt.plot(R,ppv)
plt.xlabel('R')
plt.ylabel('ppv')


#%% So far, we power-clamped at 0.8 - what if we vary power too?

beta = np.linspace(1,0,101) # power = 1 - beta
R = np.linspace(0,1,101) 
ppv = np.empty([len(R),len(beta)]) # initialize empty array
ppv[:] = np.NaN # convert to NaN
for i in range(len(R)): # loop through each R
    for j in range(len(beta)): # loop through each beta
        ppv[i,j] = ((1-beta[j])*R[i])/(R[i]-beta[j]*R[i]+alpha)
        
# Summarizing the Iannidis paper in one figure:
x = R # 1d array
y = beta # 1d array
x, y = np.meshgrid(x, y) # make a meshgrid out of x and y
z = ppv # 2d array
fig = plt.figure() # init figure
ax = fig.gca(projection='3d') # project into 3d space
surf = ax.plot_surface(x,y,z) # make surface plot
ax.set_xlabel('R') # add xlabel 
ax.set_ylabel('beta') # add ylabel 
ax.set_zlabel('ppv') # add zlabel 

# To see: Tradeoff between alpha, beta, and R.

#%% 4. Funnel plots - these are a meta-analytic tool. 
# Plotting effect size vs. power. 

# As you increase power, the effects cluster around the "real" effect
# If you run low-powered stuff, you will be at the bottom of the funnel and
# your effect sizes will jump all over the place. 

# Initialize variables:
sampleSize = np.linspace(5,250,246) # We vary sample size from 5 to 250
effectSize = 0 # The real effect size
repeats = int(1e2) # In reality, you would do many more than that
meanDifference = np.zeros([repeats,len(sampleSize)]) # preallocate

# Calculations:
for r in range(repeats): # loop through each repeat
    for i in range(len(sampleSize)): # loop through each sample size
        tempSample = int(sampleSize[i]) # what is our n this time?
        temp = np.random.normal(0,1,tempSample) + effectSize
        temp2 = np.random.normal(0,1,tempSample)
        meanDifference[r,i] = np.mean(temp) - np.mean(temp2)
        
# Learning effect: Larger n will converge to real effect
# Lower n allows fishing for noise. Instead of doing 1 high powered
# experiment, n = 250, do 10 n = 25 experiments and publish the one
# that becomes significant. Brutal.

# To save time, let's linearize that:
linearizedMeanDiff = np.ndarray.flatten(meanDifference) 
temp = np.matlib.repmat(sampleSize,repeats,1) 
linearizedSample = np.ndarray.flatten(temp)

# Now we can do the funnel plot:
plt.plot(linearizedMeanDiff,linearizedSample,'o',markersize=.5)
plt.xlabel('Observed effect size')
plt.ylabel('Sample size')

#%% 5. Powerscape

# Initialize variables:
popSize = int(1e3) # Size of the population
nnMax = 250 # Maximal sample size to be considered
nnMin = 5 # Minimal sample size to be considered
sampleSize = np.linspace(nnMin,nnMax,246) # array of each sample size
effectSize = np.linspace(0,1.5,31) # From 0 to 1.5, in .05 increments
# As std is one, effect_size will be in units of std
pwr = np.empty([len(sampleSize),len(effectSize)]) # initialize power array
pwr[:] = np.NaN # then conver to nan

# Run calculations:
for es in range(len(effectSize)): # loop through each effect size (31 total)
    A = np.random.normal(0,1,[popSize,2]) # Get the population of random 
    # numbers for each effect size - 2 columns, 1000 rows
    A[:,1] = A[:,1] + effectSize[es] # Add effect size to 2nd one
    for n in range(len(sampleSize)): # loop through each sample size
        mm = int(2e1) # Number of repeats
        significances = np.empty([mm,1]) # preallocate
        significances[:] = np.NaN
        for i in range(mm): # Do this mm times for each one
            sampInd = np.random.randint(0,popSize,[n+5,2]) # subsample
            # we add 5 to n because n is indexed at 0 but our min n is 5
            drawnSample = np.empty([n+5,2]) # initialize empty
            # drawn_sample starts as 5x2 and with each iteration adds one row
            drawnSample[:] = np.NaN # convert to NaN
            drawnSample[:,0] = A[sampInd[:,0],0] 
            drawnSample[:,1] = A[sampInd[:,0],1]
            t,p = stats.ttest_ind(drawnSample[:,0],drawnSample[:,1])
            if p < .05: # assuming our alpha is 0.05
                significances[i,0] = 1 # if significant, assign 1
            else:
                significances[i,0] = 0 # if ~significant, assign 0
        pwr[n,es] = sum(significances)/mm*100 # compute power
        
# Plot it:
plt.pcolor(pwr) #create a pseudocolor plot with a non-regular rectangular grid
plt.xlabel('real effect size (mean diff in SD')
plt.ylabel('sample size (n)')
plt.title('powerscape t-test') # color represents significant effects