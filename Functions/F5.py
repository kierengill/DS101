writtenBy = "KierenSinghGill"
dateWritten = "April 24 2021"

#FUNCTION DESCRIPTION
    #This function is a Bayesian calculator that can be used to calculate Bayesian probabilities.
#This function takes in 4 parameters:
    #a: The prior probability of A, b: The prior probability of B, c: The likelihood
    #(the probability of B given A) and d: A flag (the number 1 or 2) whether
    #this function will implement 1) the simple or 2) the explicit version of Bayes theorem.
    #If the flag is set to 2 (the explicit version), input argument b should be interpreted as 
    #“the probability of B given not A”, instead of the prior probability of B.
#This function returns: 
    #The posterior using Bayes Theorem

def bayesCalculator(prior_A, prior_B, likelihood, flag):
    if flag == 1:
        posterior = prior_A * likelihood / float(prior_B)
        return posterior
    
    elif flag == 2:
        posterior = likelihood * prior_A / float(likelihood * prior_A + prior_B * (1-prior_A))
        return posterior


print(bayesCalculator(0.01,0.1,0.99,1))
print(bayesCalculator(0.01,0.1,0.99,2))
print(bayesCalculator(0.0001,0.01,1,1))
print(bayesCalculator(0.06,0.1,0.95,2))
print(bayesCalculator(0.3775,0.33,0.9,2))
print(bayesCalculator(0.05,0.0625,1,1))
print(bayesCalculator(0.05,0.0625,0.8,1))
print(bayesCalculator(0.3,0.01,0.95,2))
print(bayesCalculator(0.3,0.25,0.5,2))