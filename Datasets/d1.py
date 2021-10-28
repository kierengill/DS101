import numpy as np
import pandas as pd

pandaSata = pd.read_csv('/Users/kierensinghgill/Desktop/Homework/2021 Spring/Intro to DataSci/Datasets/movieRatingsDeidentified.csv', delimiter = ',')
sata = np.genfromtxt('/Users/kierensinghgill/Desktop/Homework/2021 Spring/Intro to DataSci/Datasets/movieRatingsDeidentified.csv', delimiter=',', skip_header=1)

#Movies
meanMovieRating = np.nanmean(sata,axis=0) #mean rating of each movie, across all participants
medianMovieRating = np.nanmedian(sata,axis=0) #median rating of each movie, across all participants
stdRatingMovie = np.nanstd(sata,axis=0) #standard deviation of the ratings of each movie, across all participants

#Participants
meanParticipantRating = np.nanmean(sata, axis=1) #mean rating of each research participant
medianParticipantRating = np.nanmedian(sata,axis=1) #median rating of each research participant
stdParticipantRating = np.nanstd(sata,axis=1) #standard deviation of the ratings of each participant

#mean of means
meanOfMeans = np.nanmean(meanParticipantRating)

#mean of Stds
meanOfStds = np.nanstd(stdParticipantRating)

#separate headings
headings = pandaSata.columns

#get index of the desired movie, and use the index to locate corresponding mean/median for the movie
location = headings.get_loc("""insert movie name here""")