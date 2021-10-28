import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sklearn as sk


dataset = pd.read_csv('/Users/kierensinghgill/Desktop/middleSchoolData.csv', delimiter = ',')
data = dataset.to_numpy()

#%%

applications = (data[:,2]).astype(float)
admissions = (data[:,3]).astype(float)

temp = np.vstack((applications , admissions))
temp = np.transpose(temp)

r = np.corrcoef(applications, admissions)[0,1]
plt.scatter(applications, admissions)
plt.xlabel('Applications')
plt.ylabel('Acceptances')

#1) Correlation between applications and admissions
    #0.8017265370719315
    
#%%
#Application rate = number of application per school size
schoolSize = (data[:,20]).astype(float)

indexTracker = []
for i in range(0, len(schoolSize)):
    if np.isnan(schoolSize[i]):
        indexTracker.append(i)
        

temp = np.vstack((applications,schoolSize))
temp = np.transpose(temp)
temp = temp[~np.isnan(temp).any(axis=1)]

applicationRate = np.divide(temp[:,0],temp[:,1])

#%%
#Remove rows in the admissions column which corresspond to  NaN rows in the class size column
temp = np.vstack((admissions , schoolSize))
temp = np.transpose(temp)
temp = temp[~np.isnan(temp).any(axis=1)]
admissions_cleaned = temp[:,0]
r = np.corrcoef(applicationRate, admissions_cleaned)[0,1]

#2) Correlation between application rate and admissions
    #0.658750752900268

#%%
#Per student odds
    #number of students who were accepted / school size
    
acceptanceRate = np.divide(temp[:,0],temp[:,1])

largest = 0   
     
#Loop through the array    
for i in range(0, len(acceptanceRate)):    
    #Compare elements of array with max    
   if(acceptanceRate[i] > largest):    
       largest = acceptanceRate[i];    

empty_list = []
for i in range(0, len(acceptanceRate)):
    if acceptanceRate[i] == largest:
        empty_list.append(i)

odds = largest / (1-largest)

schools = (data[:,1]).astype(str)

bestPerStudent = schools[304]

#3) School with best per student odds
    #THE CHRISTA MCAULIFFE SCHOOL\I.S. 187
    
#%%

#Dimension reduction using PCA for factors
newdata = (data[:,11:17]).astype(float)

newdata = newdata[~np.isnan(newdata).any(axis=1)]

# You're in luck. There is now a function that does it all in one
# One catch: The PCA expects normally distributed DATA
# So that is why we z-score the data first

# Z-score the data:
zscoredData = stats.zscore(newdata)

# Run the PCA:
pca = PCA()
pca.fit(zscoredData)

# eig_vals: Single vector of eigenvalues in decreasing order of magnitude
eigVals = pca.explained_variance_

# loadings: Weights per factor in terms of the original data. Where do the
# principal components point, in terms of the 17 questions?
loadings = pca.components_

# rotated-data: Simply the transformed data - we had 40 courses (rows) in
# terms of 17 variables (columns), now we have 40 courses in terms of 17
# factors ordered by decreasing eigenvalue
rotatedData = pca.fit_transform(zscoredData)

# For the purposes of this, you can think of eigenvalues in terms of 
# (co)variance explained
covarExplained = eigVals/sum(eigVals)*100

# We note that there is a single factor (!) - something like class quality 
# or overall experience that explains most of the data

eigenValueFactors = 3.8106

plt.bar(np.linspace(1,6,6),loadings[:,3])
plt.xlabel('School perception metric')
plt.ylabel('Effect on component')

numClasses = 5

# =============================================================================
# #plt.bar(np.linspace(1,num_classes,num_classes),eig_vals)
# plt.plot(eigVals)
# plt.xlabel('Principal component')
# plt.ylabel('Eigenvalue')
# plt.plot([0,numClasses],[1,1],color='red',linewidth=1) 
# =============================================================================


#%%

#Dimension reduction using PCA for achievements
newdata = (data[:,21:24]).astype(float)

newdata = newdata[~np.isnan(newdata).any(axis=1)]

# You're in luck. There is now a function that does it all in one
# One catch: The PCA expects normally distributed DATA
# So that is why we z-score the data first

# Z-score the data:
zscoredData = stats.zscore(newdata)

# Run the PCA:
pca = PCA()
pca.fit(zscoredData)

# eig_vals: Single vector of eigenvalues in decreasing order of magnitude
eigVals = pca.explained_variance_

# loadings: Weights per factor in terms of the original data. Where do the
# principal components point, in terms of the 17 questions?
loadings = pca.components_

# rotated-data: Simply the transformed data - we had 40 courses (rows) in
# terms of 17 variables (columns), now we have 40 courses in terms of 17
# factors ordered by decreasing eigenvalue
rotatedData = pca.fit_transform(zscoredData)

# For the purposes of this, you can think of eigenvalues in terms of 
# (co)variance explained
covarExplained = eigVals/sum(eigVals)*100

# We note that there is a single factor (!) - something like class quality 
# or overall experience that explains most of the data

eigenValueAchievements = 2.20659

#plt.bar(np.linspace(1,3,3),loadings[:,2])
plt.xlabel('Achievement Metric')
plt.ylabel('Loading')

# =============================================================================
# numClasses = 2
# #plt.bar(np.linspace(1,num_classes,num_classes),eig_vals)
# plt.plot(eigVals)
# plt.xlabel('Principal component')
# plt.ylabel('Eigenvalue')
# plt.plot([0,numClasses],[1,1],color='red',linewidth=1) 
# 
# =============================================================================


#%%

#I picked the 6th feature based on the graphs of the 4 principal components, because 
#it seemed to carry the most weight

#I picked the second achievement based on the graph of the 2 principal components
#for the same reason

#Now, I'll do correlation between feature 6 and achivement 2
#Trust and Reading Scores

trust = (data[:,16]).astype(float)
collabTeachers = (data[:,12]).astype(float)
rigorTeaching = (data[:,11]).astype(float)

readingScores = (data[:,22]).astype(float)

temp = np.vstack((trust , readingScores))
temp = np.transpose(temp)
temp = temp[~np.isnan(temp).any(axis=1)]

r = np.corrcoef(temp[:,0],temp[:,1])[0,1]
plt.scatter(trust, readingScores)

#r with feature 6 (trust) and achievement 2 (reading scores) = 0.044768692957930054
#r with feature 1 (rigorous instruction) and achievement 2 (reading scores) = 0.44517063688687525
#r with feature 2 (collaborative teachers) and achievement 2 (reading scores) = 0.2977188101745054

#4) Is there a correlation between how students perceive school vs. how school's perform?
    #No, weak correlation scores throughout
    
#%%

#Now, the previous cell isn't correct, we need to use multiple regression.
#I am taking features 1,2, and 6 and achievement 2

from sklearn import linear_model # library for multiple linear regression

def normalizedError(inputData, flag):
    yHat = inputData[:,0]
    y = inputData[:,1]
    n = len(y)
    residuals = (abs(yHat - y))**flag
    sumOfResiduals = int(np.sum(residuals))
    result = (sumOfResiduals/n)**(1/flag)
    return result


temp = np.vstack((trust, collabTeachers, rigorTeaching, readingScores))
temp = np.transpose(temp)
temp = temp[~np.isnan(temp).any(axis=1)]

X = temp[:,0:3]
Y = temp[:,3] # income

regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # use fit method 
rSqr = regr.score(X,Y) #0.23381138853362726
R = 0.23381138853362726**0.5 #0.48354047248769905
betas = regr.coef_ #m
yInt = regr.intercept_  #b

yHats = (temp[:,0] * betas[0]) + (temp[:,1] * betas[1]) + (temp[:,2] * betas[2]) + yInt
RMSEArray = np.vstack((yHats,temp[:,3])).T
RMSE = normalizedError(RMSEArray,2)
#RMSE = 0.16071794642348758

#%%

#Q5
#Hypothesis: Smaller schools perform better on reading scores 

size = (data[:,20]).astype(float)
reading = (data[:,22]).astype(float)

temp = np.vstack((size, reading))
temp = np.transpose(temp)
temp = temp[~np.isnan(temp).any(axis=1)]

size = temp[:,0]
reading = temp[:,1]
median = np.median(size)
#539.0

#small
for i in range(len(size)):
    if size[i] <= 539:
        size[i] = 0

#big
for i in range(len(size)):
    if size[i] > 539:
        size[i] = 1

temp = temp = np.vstack((size, reading))
temp = np.transpose(temp)

#small
smallSchools = []
for i in range(len(temp[:,0])):
    if temp[:,0][i] == 0:
        smallSchools.append(temp[:,1][i])


largeSchools = []
for i in range(len(temp[:,0])):
    if temp[:,0][i] == 1:
        largeSchools.append(temp[:,1][i])
        
smallSchools = np.array(smallSchools)
largeSchools = np.array(largeSchools)
median1 = np.median(smallSchools)
median2 = np.median(largeSchools)
answer = stats.mannwhitneyu(smallSchools,largeSchools)[1]
#7.934557472038052e-22

#Statistically significant because P value less than 0.05

#%%


pupilSpending = (data[:,4]).astype(float)
classSize = (data[:,5]).astype(float)

temp = np.vstack((size, reading))
temp = np.transpose(temp)
temp = temp[~np.isnan(temp).any(axis=1)]

median = np.nanmedian(pupilSpending)
#20147.0
median = np.nanmedian(classSize)
#22.05


#%%

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot as meansPlot

#Do a two-way ANOVA and show an ANOVA table
pupilSpending = (data[:,4]).astype(float)
classSize = (data[:,5]).astype(float)

#Do a two-way ANOVA and show an ANOVA table
filename = 'anova1.csv'
df = pd.read_csv(filename) #Import the data from the csv file into a dataframe
df.info() #What is the structure of the data frame?

model = ols('ReadingScores ~ PupilSpending + ClassSize + PupilSpending:ClassSize', data=df).fit() #Build the two-way ANOVA model. Value = y, X1,X2 = Main effects. X1:X2 = interaction effect
anova_table = sm.stats.anova_lm(model, typ=2) #Create the ANOVA table. Residual = Within
print(anova_table) #Show the ANOVA table


#Show the corresponding means plot
fig = meansPlot(x=df['PupilSpending'], trace=df['ClassSize'], response=df['ReadingScores'])


#%%

#admissions
acceptances = (data[:,3]).astype(float)
schoolSize = (data[:,20]).astype(float)
pps = (data[:,4]).astype(float)
classSize = (data[:,5]).astype(float)

temp = np.vstack((acceptances, schoolSize, pps, classSize))
temp = np.transpose(temp)
temp = temp[~np.isnan(temp).any(axis=1)]


#%%

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot as meansPlot

#Do a two-way ANOVA and show an ANOVA table
pupilSpending = (data[:,4]).astype(float)
classSize = (data[:,5]).astype(float)



#Do a two-way ANOVA and show an ANOVA table
filename = 'anova2.csv'
df = pd.read_csv(filename) #Import the data from the csv file into a dataframe
df.info() #What is the structure of the data frame?

model = ols('AcceptanceRate ~ PPS + ClassSize + PPS:ClassSize', data=df).fit() #Build the two-way ANOVA model. Value = y, X1,X2 = Main effects. X1:X2 = interaction effect
anova_table = sm.stats.anova_lm(model, typ=2) #Create the ANOVA table. Residual = Within
print(anova_table) #Show the ANOVA table


#Show the corresponding means plot
fig = meansPlot(x=df['PPS'], trace=df['ClassSize'], response=df['AcceptanceRate'])

#%%
#Question 7

sumA = sum(acceptances)
percent90 = 0.9*sumA
#4014.9

sortedAccept = np.sort(acceptances)[::-1]
total = 0
counter = 0
for i in sortedAccept:
    total += i
    counter += 1
    if total >= 4014.9:
        break
    
proportion = 123/594

plt.bar(np.linspace(1,594,594),sortedAccept[0:594])
plt.axvline(x=123, color="red")
plt.xlabel('School perception metric')
plt.ylabel('Effect on component')

#%%
#Question 8

#do PCA, then do lasso regression

temp = np.vstack((admissions , schoolSize))
temp = np.transpose(temp)

#Dimension reduction using PCA for factors
newdata = (data[:,4:21]).astype(float)
newdata = np.transpose(newdata)
temp = np.vstack((newdata, admissions , schoolSize))
temp = np.transpose(temp)

temp = temp[~np.isnan(temp).any(axis=1)]

newdata = temp[:,0:17]
acceptanceRateCalc = temp[:,17:]
acceptanceRate = np.divide(acceptanceRateCalc[:,0],acceptanceRateCalc[:,1])


#%%
# 8a
# Z-score the data:
zscoredData = stats.zscore(newdata)

# Run the PCA:
pca = PCA()
pca.fit(zscoredData)

# eig_vals: Single vector of eigenvalues in decreasing order of magnitude
eigVals = pca.explained_variance_

# loadings: Weights per factor in terms of the original data. Where do the
# principal components point, in terms of the 17 questions?
loadings = pca.components_

# rotated-data: Simply the transformed data - we had 40 courses (rows) in
# terms of 17 variables (columns), now we have 40 courses in terms of 17
# factors ordered by decreasing eigenvalue
rotatedData = pca.fit_transform(zscoredData)

# For the purposes of this, you can think of eigenvalues in terms of 
# (co)variance explained
covarExplained = eigVals/sum(eigVals)*100

# We note that there is a single factor (!) - something like class quality 
# or overall experience that explains most of the data

eigenSum = sum(eigVals)

#pick the first 4 components because > 1 (Kaiser criterion)
#loading 0:
    #13, 4, 6
#loading 1:
    #9, 8
#loading 2:
    #5, 17, 6, 8 
#loading 3:
    #17, 4, 3

numClasses = 17



plt.bar(np.linspace(1,17,17),loadings[:,3])
plt.xlabel('School perception metric')
plt.ylabel('Effect on component')



newdataT = np.transpose(newdata)
temp = np.vstack((newdataT , acceptanceRate))
temp = np.transpose(temp)

X = np.vstack((temp[:,3],temp[:,5],temp[:,7],temp[:,12]))
X = np.transpose(X)
Y = temp[:,17] 

regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # use fit method 
rSqr = regr.score(X,Y) #0.183004252517029
R = 0.183004252517029**0.5 #0.4277899630858922
betas = regr.coef_ #m
sorted_betas = np.sort(betas)
yInt = regr.intercept_  #b


#%%
#8b

#do PCA, then do lasso regression

#Dimension reduction using PCA for factors
newdata = (data[:,4:21]).astype(float)
newdata = np.transpose(newdata)
readingScores = (data[:,22]).astype(float)
temp = np.vstack((newdata, readingScores))
temp = np.transpose(temp)

temp = temp[~np.isnan(temp).any(axis=1)]

#%%

# Z-score the data:
zscoredData = stats.zscore(temp[:,0:17])

# Run the PCA:
pca = PCA()
pca.fit(zscoredData)

# eig_vals: Single vector of eigenvalues in decreasing order of magnitude
eigVals = pca.explained_variance_

# loadings: Weights per factor in terms of the original data. Where do the
# principal components point, in terms of the 17 questions?
loadings = pca.components_

# rotated-data: Simply the transformed data - we had 40 courses (rows) in
# terms of 17 variables (columns), now we have 40 courses in terms of 17
# factors ordered by decreasing eigenvalue
rotatedData = pca.fit_transform(zscoredData)

# For the purposes of this, you can think of eigenvalues in terms of 
# (co)variance explained
covarExplained = eigVals/sum(eigVals)*100

# We note that there is a single factor (!) - something like class quality 
# or overall experience that explains most of the data

eigenSum = sum(eigVals)

#pick the first 4 components because > 1 (Kaiser criterion)
#loading 0:
    #13, 4, 6
#loading 1:
    #9, 8
#loading 2:
    #5, 17, 6, 8 
#loading 3:
    #17, 4, 3

plt.bar(np.linspace(1,17,17),loadings[:,3])
plt.xlabel('School perception metric')
plt.ylabel('Effect on component')

numClasses = 17

X = np.vstack((temp[:,3],temp[:,5],temp[:,7],temp[:,12]))
X = np.transpose(X)
Y = temp[:,17] 

regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # use fit method 
rSqr = regr.score(X,Y) #0.183004252517029
R = 0.183004252517029**0.5 #0.4277899630858922
betas = regr.coef_ #m
sorted_betas = np.sort(betas)
yInt = regr.intercept_  #b


