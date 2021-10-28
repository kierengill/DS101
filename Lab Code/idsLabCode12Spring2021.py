# Python Session 12
# Machine Learning Lab
# Code by Pascal Wallisch and Stephen Spivack

#%% Machine learning methods are all the rage because they allow us to
# automate otherwise cognitive tasks. For instance, spam detection, virus
# detection, making a self-driving car, robots, etc. 
# We could call it "AI". It is driven by "big data" to work properly.

# Today, we'll use a toy example. Imagine you work for the wellness exchange
# at NYU. You want to predict who will get depressed in college. 
# So we can allocate resources to improve academic success rates.
# To make such a prediction, we have to use relevant (!) data and use ML
# methods to discern patterns - relationships in the data. Then apply it to
# a new set of data. 

#%% 0. Init

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

#%% 1. Loader
students = pd.read_csv('mlDataset.csv')

#%% 2. Exploring data frames
# We know about matrices and arrays.
# Those work fine, but imagine we have a matrix with 100 columns that
# represent the variables. Immediate question: What does each column
# represent. That's where data frames come in
# The data frame "students" is an object. It has fields and each field for
# each member of the data frame has a value

students.loc[21] 
students.loc[1]
students.age.loc[1713] # Just the value of that field
ages = students.age # This is a series
ages = ages.to_numpy() # Convert to array

# Coding principle: Organisms store information with DNA. But organisms
# compute with RNA - that's what makes the actual proteins.
# So: Strong advice: STORE your data in data frames. They will be inherently
# labeled. But do your computations with arrays. Because either many
# functions will flat out not work with data frames. Or it will be very
# solution.

#%% 3. Extract what we need from the students structure to do the job
# We're going to extract 6 predictors and 1 outcome
# For the time being, we won't extract gender because it is categorical. It
# can be used, but it complicates thing. 
# We also don't extract age because we will have a restricted age problem. 
# In general, ML methods are not magic. If the data going in has
# limitations, ML won't be able to rescue you. If the data is problematic,
# ML methods could make the situation worse. 

yOutcomes = students.depression.to_numpy()
predictors = students[["friends","fbfriends","extraversion",
                       "neuroticism","stress","cortisol"]].to_numpy()

# If this was the 20th century, we would now do classical statistics
# Null hypothesis significance testing. But: What will that tell you you?
# You could test 6 relationships here. Say we find that mean number of
# friends is significantly different between people with or without
# depression.

# This approach can still be meaningfully done, but it tells us about the
# population, i.e. the relationship between the constructs, i.e. between
# friendship and depression.
# What do we want to know?
# Our question is different. Of these specific students, who is likely to 
# become depressed?
# Related: We want to use all predictors at once, for a given individual

# Taking a closer look: Are the predictors uncorrelated?
# By common sense, these variables are very unlikely to be independent

#%% 4. To ascertain whether a PCA is indicated, let's look at the correlation
# heatmap
r = np.corrcoef(predictors,rowvar=False)
plt.imshow(r) 
plt.colorbar()

# The variables are not uncorrelated. There is a correlation structure
# The correlation structure suggests that there will be 2 meaningful.
# factors. 1-3 are correlated (1 cluster) and 4-6 are correlated (2nd
# cluster), but they are not correlated between clusters.
# The intercorrelations in one cluster are slightly higher than in
# another, so we predict that eigenvalues in 1 are going to be slightly
# higher.

# PCA is indicated, and we have an expectation of the results
# So let's do a PCA

# Run the PCA:
pca = PCA()
pca.fit(predictors)

# eigValues: Single vector of eigenvalues in decreasing order of magnitude
eigValues = pca.explained_variance_ 

# loadings: Weights per factor in terms of the original data. Where do the
# principal components point, in terms of the 6 predictors?
loadings = pca.components_ # sorted by explained_variance_

# rotated data: Simply the transformed data - we had 2000 students (rows) in
# terms of 6 predictors (columns), now we have 2000 students in terms of 6
# predictors ordered by decreasing eigenvalue
origDataNewCoordinates = pca.fit_transform(predictors)

#%% scree plot:
numPredictors = 6
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues)
plt.title('Scree plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalues')

# Problem: PCA does not technically work with the correlation matrix. It
# works with the covariance matrix. That is not independent of units. 
# It also assumes normal distributions. 
# Both is violated here. This didn't really rear its head with the evals
# because they were all bounded and on the same scale.
# Normalization: By z-scoring. --> Mean (0) and STD (1) - normalize by
# putting things on a STANDARD NORMAL DISTRIBUTION:

#%% Do it again, now z-scored

zscoredData = stats.zscore(predictors)
pca = PCA()
pca.fit(zscoredData)
eigValues = pca.explained_variance_ 
loadings = pca.components_
origDataNewCoordinates = pca.fit_transform(zscoredData)

numPredictors = 6
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues)
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')

#%% Looking at the corrected scree plot, we get 2 factors, both by 
# Kaiser criterion and Elbow 

# Next step: Look at the loadings to figure out meaning
# Factor 1: 
loadings[0,:] # "Pressed" or "Challenges"
# Factor 2:
loadings[1,:] # "Social support" or just "support"

#%% Old wine in new bottles:
plt.plot(origDataNewCoordinates[:,0],origDataNewCoordinates[:,1],'o',markersize=5)
plt.xlabel('Challenges')
plt.ylabel('Support')

#%% 5. Clustering - doing quantitatively what can be seen intuitively
# Clustering answers - in a data-driven way - which subgroup a datapoint
# belongs to.
# The "kMeans clustering" is like pca of clustering. It's not the only
# clustering method, but it is the most commonly used one
# Algorithm: Minimize the summed distances between a cluster center and its
# members. Once the minimum has been found (regardless of starting
# position), it stops. "Converging"
X = np.transpose(np.array([origDataNewCoordinates[:,0],origDataNewCoordinates[:,1]]))

#%% We extracted two meaningful predictors out of the raw PREDICTORS matrix
# Silhouette: How similar to points in cluster vs. others, arbitrariness

# Init:
numClusters = 9 # how many clusters are we looping over? (from 2 to 10)
Q = np.empty([numClusters,1]) # init container to store sums
Q[:] = np.NaN # convert to NaN

# Compute kMeans:
for ii in range(2, 11): # Loop through each cluster
    kMeans = KMeans(n_clusters = int(ii)).fit(X)
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(X,cId) # compute the mean silhouette coefficient of all samples
    Q[ii-2] = sum(s) # take sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,500)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(Q[ii-2]))) # sum rounded to nearest integer

#%% Plot this to make it clearer what is going on
plt.plot(np.linspace(2,10,numClusters),Q)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
# kMeans gives you the center coordinates of the clusters, assuming a number
# of clusters. Silhouette gives you how many are most unamigously described
# by the clusters. Most likely "real" number: Where the sum of the
# silhouette scores peaks. In reality, they are complementary. Use together

#%% Now let's plot and color code the data
indexVector = np.linspace(1,len(np.unique(cId)),len(np.unique(cId))) 
for ii in indexVector:
    plotIndex = np.argwhere(cId == int(ii-1))
    plt.plot(origDataNewCoordinates[plotIndex,0],origDataNewCoordinates[plotIndex,1],'o',markersize=5)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=10,color='black')  
    plt.xlabel('Challenges')
    plt.ylabel('Support')

# As you can see, kMeans returns as many clusters as you ask for. 
# What it does is, is returns the optimal center that minimizes the summed
# distance from all centers. But it requires - as an input (!) how many
# clusters to look for. Basically, you find what you look for in terms of
# cluster number. And the sum of the summed distances is only going down

# Solution: "Silhouette"
# Silhouette takes distances nearest neighbor clusters into account

# Exercise: Plot correct solution for clustering example (4 clusters - the 
# peak of the Silhouette).

#%% 6. Classification: Using the predictors to predict an outcome
# In general, you use some data to build a model
# Then you use other data to test the model and check how accurate it is
# Intuition
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0],X[:,1],yOutcomes)

#%% This visualizes the issue: Some people do get depressed, others don't 
plt.plot(X[np.argwhere(yOutcomes==0),0],X[np.argwhere(yOutcomes==0),1],'o',markersize=5,color='green')
plt.plot(X[np.argwhere(yOutcomes==1),0],X[np.argwhere(yOutcomes==1),1],'o',markersize=5,color='blue')
plt.xlabel('Challenges')
plt.ylabel('Support')

#%%  Another view on this, but the outcome is represented by color, not a 3rd
# dimension. To make it clear that you have to draw a line in predictor
# space, not outcome space
# Name of the game: Draw a straight line ("linear separator") that optimally
# separates people with 1 outcome from people with another outcome. In
# higher dimensions: "Hyperplane", 2D: Line, 3D: Plane, 4D: Hyperplane

# Insight: 
# Given this data, it is impossible to draw a line that perfectly separates
# the subgroups (depressed vs. not). This is normal. There will be
# misclassifications (people who should be depressed, given how terrible
# everything is, but are not, and vice versa). 
# Does something have to be perfect to be good? The SVM finds what is called
# the "widest margin classifier" - that seperatates the two outcomes - in
# predictor space as best as possible. Widest possible margin that is
# spanned by the "support vectors". 

# Step 1: Fit the model to the data
svmModel = svm.SVC()
svmModel.fit(X,yOutcomes)

# Step 2: Visualize the support vectors (you can skip this once you know
# what you're doing). For now, I want to add that to the scatter plot
sV = svmModel.support_vectors_ # Retrieve the support vectors from the model
plt.plot(sV[:,0],sV[:,1],'o',markersize=5,color='green')
# These "support vectors" span the decision boundary

# Step 3: Use model to make predictions and assess accuracy of model
# You will want to do this on new data to avoid overfitting. If you test the
# model on the same data that you fit it on, you will overestimate the
# accuracy of the model. It will not generalize because you are fitting to
# noise
decision = svmModel.predict(X) # Decision reflects who the model thinks will be depressed

# Step 4: Assess model accuracy by comparing predictions with reality
comp = np.transpose(np.array([decision,yOutcomes])) 
modelAccuracy = sum(comp[:,0] == comp[:,1])/len(comp)

# This model would predict the depression status of ~95% of the students
# correctly. Given an overfit model. 
# Baseline: 75% - we guess "not depressed" - we could weight outcomes
# Correctly guessing depression might be more valuable than the other
# Also, the "errors" are not equal

# SVM are a cardinal example of linear classifiers. They are used very
# often. Their advantage is that they are easily understood. Directly
# theoretically interpretable. Problem: They basically known to be too
# simple to really model complex phenomena perfectly.

# There is a large number of nonlinear classifiers, like CNNs, RNNs, ANNs,
# and so on. Here, we will show one commonly used one. The random forest.

#%% 7. The random forest. 
# Advantage: Very powerful. Allows to learn and model complex behavior of data. 
# Disadvantage: Very hard to interpret the output

treeModel = DecisionTreeClassifier().fit(X,yOutcomes) # build the model
pred = treeModel.predict(X) # make predictions
empiricalVsPrediction = np.transpose(np.array([yOutcomes,pred])) 
modelAccuracy = sum(empiricalVsPrediction[:,0] == empiricalVsPrediction[:,1])/len(comp)*100

# We are able to predict 100% of the outcomes with this model. There are no
# errors. Even the strange cases, we got. The problem is that if you have
# results that are too good to be true, they probably are not true. 
# We committed the sin of "overfitting", due to the fact that we used the
# same data to both fit ("train") the model and test it.
# Prescription: "Don't do that". Use one set of data to build the model and
# another to train it.

# The problem is that results from overfit models won't generalize because
# some proportion of the data is due to noise. If you fit perfectly, you fit
# to the noise. The noise will - by definition - not replicate. 
# Best solution: Get new data. Rarely practical.
# Most common solution: Split the dataset. There are many ways to do this,
# e.g. 50/50, 80/20 (at random. Most powerful (most computationally
# intensive): "Leave one out": Use the entire dataset to build the model,
# expect for one point. Predict that point from n-1 data. Do that many
# times, at random, and average the results. 


