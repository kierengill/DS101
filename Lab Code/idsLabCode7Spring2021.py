# Python Session 7
# Hypothesis testing lab
# Code by Pascal Wallisch and Stephen Spivack

#%% This script introduces how to analyze data with the canonical data
# analysis cascade covering classical (descriptive and inferential)
# statistics for behavioral data. 

# Specifically, we will analyze a subset of the data that Pascal recorded
# in the early 2000s. The paper pertains to movie ratings. And perhaps
# fittingly, it is ratings for the  movies "Matrix" I to III. 

# (Null) Hypothesis: People like the 3 movies in the trilogy equally well.
# Why might this be? People who are into it are into it. Others are not.

# Alternative hypothesis: The 1st one is rated best.
# Why might people like the 1st one the most? It was new, and after that,
# expectations are set, and they are hard to beat. 
 
# Alternative hypothesis 2: They will rise, actually. 
# Why might they rise? Dropout - only "fanboys" will stick around (experimental mortality). 
# Let's find out which one it is

#%% 0 Init (Birth)

# a. Load/import - libraries/packages:
import numpy as np
import matplotlib.pyplot as plt

# b. Predispositions:
startColumn = 177 # This is the column that contains data from matrix I
numMovies = 3 # Data from how many movies
pruningMode = 1 # This flag determines how we will remove nans. 1 = element-wise, 2 = row-wise, 3 = imputation

#%% 1 Loader/Transducer - "Retina"

# This step puts the data from whatever format it comes in into a matrix.
# Once we have it in a matrix, we own it. [Can be a dataframe if you want - 
# more on this later].
# We do assume/presume we're in the Code folder
# With Excel

data = np.genfromtxt('movieRatingsDeidentified.csv', delimiter = ',', skip_header = 1)
# We want just the data, so skip the first row / header
dataMatrix = data[:,startColumn:startColumn+numMovies] # Should yield a n x 3 matrix
M1 = dataMatrix[:,0] # Separate them, for 
M2 = dataMatrix[:,1]
M3 = dataMatrix[:,2]

#%% 2 Pruner/Filter/Preprocessor - "Thalamus"

# By visual inspection, we realize that there are a lot of "nans", which is
# missing information. Given that these are movie ratings, it probably
# represents people who didn't watch the movie. In other words, the data is
# missing systematically, not randomly (probably). 
# This is a challenge because the way we handle this here - at this critical
# strategic juncture sets up the entire rest of the analysis. Depending on
# our choices, some analyses won't make sense or even be possible.

# The easiest way to deal with the missing data is to remove the nans. We can't just leave them. 
# That would invalidate the entire rest of the analysis. 
# But how? 

# a) Element-wise: Just dropping all nans wherever they occur.
# Challenge: There is an unequal number of nans per movie. Some people saw
# only 1 of the movies, some 2, some 3, etc.
# If we do element-wise exclusion, this will yield matrices of unequal
# length (n). We cannot do a paired anything test if we do that.

if pruningMode == 1:
    M1 = M1[np.logical_not(np.isnan(M1))] 
   #M1 = M1[np.isfinite(M1)] # another way to do the same thing
    M2 = M2[np.logical_not(np.isnan(M2))] 
   #M2 = M1[np.isfinite(M2)]
    M3 = M3[np.logical_not(np.isnan(M3))]
   #M3 = M3[np.isfinite(M3)]

# b) Row(participant)-wise elimination of missing data.
# Next time

# c) Imputation - but here that probably makes no sense because of self-selection
# Once we do matrix factorization

#%% 3 Formatting the data into a representation that we can actually use - "V1"

# This is where I usually spend most of my time. Usually this is
# something like: The data came in, in one long chronological stream (say
# trial numbers or straight up time), unless you study timing effects, that
# is often not that useful. Here, I already did this step. This already is
# in a nice format
# But let's do one thing - which will help with the analysis: Putting the 3
# disparate data snippets in one variable that contains all data of
# interest.
# We could do this if n was equal. But our pruning of the data by removing 
# missing data element wise precludes that. 

# So let's combine our data into a single array called, which will 
# consist of an array of arrays:
combinedData = np.array([M1,M2,M3])

# Now that the data is combined, let's open the first array, reach inside
# and grab the 10th element:
example = combinedData[0][10]

#%% 4 Actual data analysis - several things are reasonble - "Extrastriate Cortex"

# a) Descriptive statistics - we are looking for very special numbers that
# capture the essence of the entire dataset - the typical number (central
# tendency) and the dispersion. Typically mean and SD. 

descriptivesContainer = np.empty([numMovies,4])
descriptivesContainer[:] = np.NaN 
# 4 numbers per movie that summarize the entire dataset.
# Note, you can use the "all descriptives" function you wrote here.

if pruningMode == 1:
    for i in range(numMovies):
        descriptivesContainer[i,0] = np.mean(combinedData[i]) # mu
        descriptivesContainer[i,1] = np.std(combinedData[i]) # sigma
        descriptivesContainer[i,2] = len(combinedData[i]) # n
        descriptivesContainer[i,3] = descriptivesContainer[i,1]/np.sqrt(descriptivesContainer[i,2]) # sem


# b) Inferential statistics
# The question here is whether the differences we saw in the descriptives -
# usually the means - are "statistically significant". 
# Whether it is plausibly consistent with chance (that the numbers came out
# of a RGN)

# 1) Assert a null hypothesis: We assume that the data came out of a RNG -
# strictly by chance.

# 2) We compute the probability that this is the case - assuming chance. 

# 3) If this probability is implausibly low, we DECIDE to concede that we
# were wrong in 1) - that it is probably not solely due to chance. 

# It's a choice, it's a decision. You have not falsified the null hypothesis
# or "proven" to be wrong. We made a choice that it is implausible. But we
# could be wrong (type I and type II errors). 

# In science, we consider things that happen only 1 in 20 times to be "too
# implausible" to be consistent with chance. This corresponds to getting
# heads (or tails) 5 times in a row (if you flip coins). Is that terribly
# implausible? There is now a movement afoot to lower this to 1 in 200. To
# avoid false positives. So results won't reproduce. This gave p values a
# bad reputation. But there is nothing wrong with them, if they are
# understood and used properly. Properly = conservative enough criterion of
# implausibility and high enough power. 

# Let's start with t-tests, they are the easiest (and most common) thing to
# do. This has to be an independent samples t-test because our n is unequal.
# t-test has lots of assumptions: Normality, homogeniety of variances, etc. 

from scipy import stats # we need this module to perform our t-test

if pruningMode == 1:
    t1,p1 = stats.ttest_ind(M1,M2)
    t2,p2 = stats.ttest_ind(M1,M3)
    t3,p3 = stats.ttest_ind(M2,M3)
    
# p-value - in this case it is below 0.05
# Remember: We do a t-test in the first place
# because we don't know how the data distributes. We have to calculate the
# SD from the data itself. That calculation uses some data to calculate the
# mean. And we calculate 2 means in an independent samples t-test, so we
# "lose" 2 df. 

#%% 5 Now let's plot the outcome of our analysis - "Motor Cortex"

x = ['Matrix 1', 'Matrix 2', 'Matrix 3'] # labels for the bars
xPos = np.array([1,2,3]) # x-values for the bars
plt.bar(xPos,descriptivesContainer[:,0],width=0.5,yerr=descriptivesContainer[:,3]) # bars + error
plt.xticks(xPos, x) # label the x_pos with the labels
plt.ylabel('Mean rating') # add y-label
plt.title('t = {:.3f}'.format(t1) + ', p = {:.3f}'.format(p1)) # title is the test stat and p-value

#%% 6 Epilogue: Pandas DataFrame

# 0. Import library:
import pandas as pd

# 1. Load data:
df = pd.read_csv('movieRatingsDeidentified.csv',skipinitialspace=True)
# Fill empty strings with NaN
# Now we have the headers AND the data in one object
# This is a DataFrame. For handling tabular data.

# 2. Let's get a handle on our movie titles:
titles = df.columns 
print(titles)
# We won't use this for subsequent analyses
# but it's nice to see all the titles at once

# 3. Find the Matrix data:
title = 'Matrix' # or any other title, for that matter
theMatrix = df.loc[:,df.columns.str.contains(title)]

# 4. Perform descriptives:
magic = theMatrix.describe()
# We don't have to run a loop or initialize a container
# We are still missing the SEM, so let's add it:
temp = magic.iloc[2,:]/np.sqrt(magic.iloc[0,:])
magic.loc['sem'] = temp
