import numpy as np

sata = np.genfromtxt('/Users/kierensinghgill/Desktop/kepler.txt')

temp = np.copy(sata)

casteMembership = temp[:,0]
IQ = temp[:,1]
brainMass = temp[:,2]
hoursWorked = temp[:,3]
annualIncome = temp[:,4]

#Correlation between IQ and Caste Membership
corr_caste_IQ = np.corrcoef(casteMembership,IQ)[1,0]

from simple_linear_regress_func import simple_linear_regress_func

#Correlation between IQ and Caste Membership while accounting for Brain Mass
inputData = np.transpose([brainMass,casteMembership])
                          
output = simple_linear_regress_func(inputData) # output returns m,b,r^2
y = casteMembership # actual income
yHat = output[0]* brainMass + output[1] # predicted income
residuals1 = y - yHat # compute residuals

inputData = np.transpose([brainMass,IQ])
                          
output = simple_linear_regress_func(inputData) # output returns m,b,r^2
y = casteMembership # actual income
yHat = output[0]* brainMass + output[1] # predicted income
residuals2 = y - yHat # compute residuals

part_corr = np.corrcoef(residuals1,residuals2)


inputData = np.transpose([brainMass,IQ])
output = simple_linear_regress_func(inputData) # output returns m,b,r^2

corr_caste_income = np.corrcoef(casteMembership,annualIncome)[1,0]

inputData = np.transpose([IQ,annualIncome])
output = simple_linear_regress_func(inputData)

inputData = np.transpose([hoursWorked,annualIncome])
output = simple_linear_regress_func(inputData)

#%%
from sklearn import linear_model # library for multiple linear regression
X = np.transpose([IQ,hoursWorked]) # IQ, hours worked
Y = annualIncome # income
regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # use fit method 
rSqr = regr.score(X,Y) # 
y = regr.predict([[120,50]])

#%%
from sklearn import linear_model # library for multiple linear regression


#First, do a Multiple regression for the change in casteMembership depending on IQ and hours worked
#get the betas of the multiple regression plane equation
#calculate y_hat using those betas
#then get the residuals
X = np.transpose([IQ,hoursWorked]) # IQ, hours worked
Y = casteMembership # casteMembership
regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # use fit method 
rSqr = regr.score(X,Y) # 
betas = regr.coef_ # m
yInt = regr.intercept_
y_hat = betas[0]* IQ + betas[1]* hoursWorked + yInt
residuals1 = Y - y_hat

#Then, do a Multiple regression for the change in income depending on IQ and hours worked
#get the betas of the multiple regression plane equation
#calculate y_hat using those betas
#then get the residuals
X2 = np.transpose([IQ,hoursWorked]) # IQ, hours worked
Y2 = annualIncome # income
regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X2,Y2) # use fit method 
rSqr = regr.score(X2,Y2) # 
betas2 = regr.coef_ # m
yInt2 = regr.intercept_
y_hat2 = betas2[0]* IQ + betas2[1]* hoursWorked + yInt2
residuals2 = Y2 - y_hat2

#Finally, do a partial correlation between the residuals
part_corr = np.corrcoef(residuals1,residuals2)
#-0.0146259







