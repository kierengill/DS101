# Python Session 11
# PCA Lab
# Code by Pascal Wallisch and Stephen Spivack

#%% In this script, we will illustrate dimension reduction principles with a 
# toy - but real - example. Both in terms of use case and data.

# Example: Real student evaluation data from a NYU Department in the last year

# Issue: We ask students a lot of questions per class, so many don't respond.
# So the Response rate is low. This is a problem due to representativeness concerns. 

# Suspicion: It is low because a typical student has to answer close to a hundred 
# questions per semester if they take a handful of classes. 
# Can we reduce the number of questions we ask? 

#%% 0. Init

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sqlalchemy import create_engine

#%% 1. Load data

# Load questions into a dataframe:
# In Latin-1, every character fits into a single byte
questions = pd.read_csv('/Users/kierensinghgill/Desktop/Homework/2021 Spring/Intro to DataSci/Lab Code/evaluationQuestions.csv', encoding='latin-1',header=None)

# Looking at this, we can confirm that we ask a lot of these students.
# For instance, we ask both about the course and the instructor
# Whether the students differentiate is an empirical question. if they do, 
# we should expect 2 factors, and if they don't, we shouldn't. 

# Load data:
data = np.genfromtxt('/Users/kierensinghgill/Desktop/Homework/2021 Spring/Intro to DataSci/Lab Code/courseEvaluationData.csv', delimiter=',')
# We have data from 40 courses and 17 measures (variables) per course

# Here, we will introduce the mqSQL - the natural habitat of data is a
# database. So it makes sense to introduce that now. Convert the data to a table
# in a relational database, then get them back with a couple of sample
# queries

#%% Adding the questions dataframe to mysql database

# Create connection with the MySQL server host, username and password can be set from MySQL installer:
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='root')

# Create cursor:
my_cursor = connection.cursor()

# Create database in MySQL server:
# Capital letters are commands and small letter is name of database
my_cursor.execute("CREATE DATABASE testdatabase")


# Adding the questions dataframe to MySQL database:
# Create sqlalchemy engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",
                               pw="root",
                               db="testdatabase"))

questions.to_sql('questions_table', con = engine, if_exists = 'replace', chunksize = 1000)


# Querying the data from MySQL server:
# The connctor needs to be defined again with the created database for querying/importing data
connection = pymysql.connect(host="localhost",
                             user="root",
                             password="root",
                             db="testdatabase"
                             )

my_cursor = connection.cursor()

# Execute Query:
my_cursor.execute("SELECT * from questions_table")

# Fetch the records:
result = my_cursor.fetchall()

for i in result:
    print(i)
    
# Conditional queries:
# Querying only first five questions
my_cursor.execute("SELECT * FROM questions_table LIMIT 5 ")

for j in my_cursor:
    print(j)

connection.close()


#%% 2 Looking at the raw data (Exploratory data analysis)

plt.imshow(data) # Display an image, i.e. data, on a 2D regular raster.
plt.colorbar() # Add color bar 

# Some observations
# 1) There is a lot of variability. This is a good thing. No variability -->
# Fail

# 2) We note that there are some courses (e.g. #7 or #35) where there is very
# little variability. Those happen to be courses with very low enrollment,
# so there isn't enough data (as per CLT) to be useful. We should probably
# exclude them. For now, we'll keep them.

#%% 3 Create the correlation matrix and look at it. 

# The reason this matters is so you can evalute the output of the PCA.
# Nothing in the PCA will not already be foreshadowed by this. 
# If there are clusters in the data here, there will be factors in the PCA

# Compute correlation between each measure across all courses:
r = np.corrcoef(data,rowvar=False)

# Plot the data:
plt.imshow(r) 
plt.colorbar()

# Observation 1: Most variables are very highly correlated with each other
# Observation 2: There is probably going to be a 2nd factor, but that one
# will be very narrow, basically question 10
# Observation 3: There might be a 3rd factor, but it's not as clear, around
# question 6

#%% 4 Actually doing the PCA

# You're in luck. There is now a function that does it all in one
# One catch: The PCA expects normally distributed DATA
# So that is why we z-score the data first

# Z-score the data:
zscoredData = stats.zscore(data)

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

#%% 5 Scree plot

# It is up to the researcher how many factors to interpret meaningfully
# All dimension reduction methods are exhaustive, i.e. if you put 17
# variables in, you get 17 factors back. If you put 100 variables in, you
# get 100 factors back. But there are not all created equal. Some explain a
# lot more of the covariability than others. 

# What a scree plot is: Plotting a bar graph of the sorted Eigenvalues
numClasses = 17
#plt.bar(np.linspace(1,num_classes,num_classes),eig_vals)
plt.plot(eigVals)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,numClasses],[1,1],color='red',linewidth=1) # Kaiser criterion line


# There are 4 criteria by which people commonly pick the number of factors
# they interpret meaningfully:
    
# 1) Kaiser criterion: Keep all factors with an eigenvalue > 1
# Rationale: Each variable adds 1 to the sum of eigenvalues. The eigensum. 
# We expect each factor to explain at least as much as it adds to what needs
# to be explained. The factors have to carry their weight.
# By this criterion, we would report 2 meaningful factors. Generally speaking, this is
# a liberal criterion. You will end up with many factors, using this
# criterion. If you put in 256 EEG channels, 10 factors might exceed this
# threshold. 

# 2) The "elbow" criterion: Pick only factors left of the biggest/sharpest
# drop. This would yield 1 factor

# 3) Number of factors that account for 90% of the variance (Eigenvalues that 
# add up to 90% of the Eigensum. To account for 90% of the variability in this 
# data, we need 3 factors

# 4) "Horn's method". Simulate noise distributions to see which factors exceed
# what you would expect from noise. Resampling/Bootstrap-based. This is 
# explained in the NDS book, in the PCA chapter. 

#%% 6 Horn's parallel method (Horn, 1965) as described on page 239ff of the NDS book

# Initialize variables:
nDraws = 10000 # How many repetitions per resampling?
numRows = 40 # How many rows to recreate the dimensionality of the original data?
numColumns = 17 # How many columns to recreate the dimensionality of the original data?
eigSata = np.empty([nDraws,numColumns]) # Initialize array to keep eigenvalues of sata
eigSata[:] = np.NaN # Convert to NaN

for i in range(nDraws):
    # Draw the sata from a normal distribution:
    sata = np.random.normal(0,1,[numRows,numColumns]) 
    # Run the PCA on the sata:
    pca = PCA()
    pca.fit(sata)
    # Keep the eigenvalues:
    temp = pca.explained_variance_
    eigSata[i] = temp

#%% That was fast. And we did it 10,000 times. I bet Horn would have loved to
# do that. He had to wait months for something like this.

# Make a plot of that and superimpose the real data on top of the sata:
plt.plot(np.linspace(1,numColumns,numColumns),eigVals,color='blue') # plot eig_vals from section 4
plt.plot(np.linspace(1,numColumns,numColumns),np.transpose(eigSata),color='black') # plot eig_sata
plt.plot([1,numColumns],[1,1],color='red') # Kaiser criterion line
plt.xlabel('Principal component (SATA)')
plt.ylabel('Eigenvalue of SATA')
plt.legend(['data','sata'])

# By this method, as you can see, only the first factor exceeds the noise
# distribution. If you want to be fancy, calculate the empirical confidence
# interval of the 10,000 SATA eigenfactors and note the empirical values that exceed it 

#%% 7 Interpreting the factors
# Now that we realize that 1, 2 or 3 are reasonable solutions to the course
# evaluation issue, we have to interpret the factors.
# This is perhaps where researchers have the most leeway.
# You do this - in principle - by looking at the loadings - in which
# direction does the factor point? 

whichPrincipalComponent = 0 # Try a few possibilities (at least 1,2,3)

# 1: The first one accounts for almost everything, so it will probably point 
# in all directions at once
# 2: Challenging/informative - how much information?
# 3: Organization/clarity: Pointing to 6 and 5, and away from 16 - structure?

plt.bar(np.linspace(1,17,17),loadings[:,whichPrincipalComponent])
plt.xlabel('Question')
plt.ylabel('Loading')

# General principle: Looking at the highest loadings (positive or negative)
# and looking for commonalities.

#%% 8 Usually, we want to look at the old data in the new coordinate system
# For instance, let's say the school wants to figure out which courses are
# good or needlessly hard, we can now look at that

plt.plot(rotatedData[:,0],rotatedData[:,1],'o',markersize=10)
plt.xlabel('Overall course quality')
plt.ylabel('Hardness of course')

# In this sense, PCA can help in decision making - are there some classes
# that are under/over-performing, given their characteristics?
# If we had more than 40 courses, looking at the 3rd dimension would be
# interesting too. As is, it is a bit sparse.
