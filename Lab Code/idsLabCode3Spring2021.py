# Lab session 3
# Objective: Introducing while loops and functions, as well as continuing linear algebra, introducing descriptive stats, some file i/o
# Code by Pascal Wallisch and Stephen Spivack

# Last time, we introduced for loops, if statements and mixing if statements with for loops
# The reason we did this is precisely why people use computers to analyze
# data in the first place. Briefly, complicated operations are usually
# broken down into many *simple* steps that are done iteratively. 
# Iterative: Repeatedly, with small adjustments. 
# Often recursively. Recursively: Invoking a prior version of itself
# If statements: Implement applied logic. 


#%% 1a) While loops

# Fundamental assumption / downside of the for loop: You have to know how
# many times you need to iterate for. 
# Why might you not know how often you have to do that? 
# Say you run some code that depends on user input, e.g. runs until a user presses some button. 
# We have no idea how long that will take. 
# Another key use case is if the input is unknown, and it is unclear a priori how much
# processing is needed until a desired output is reached. 
# For use cases like that, while loops are indicated. 

# Warning: While loops are a key reason for critical logical errors, if done
# wrong. Here is why: The for loop just runs for n iterations, and you tell
# it what n is. The worst that usually happens is that you're off by some number.
# You asked for the wrong n. Fine. In while loops, it can be much worse, as
# the loop runs as long as the condition is true.

number = 1 # initialize counter
while 1 != 2: # condition: As long as 1 is not equal to 2 (this is always true, in this reality)
    number = number + 1 # add 1 to counter
    print(number) # print output in console
    
# Observation: the loop will never end, as 1 is never equal to 2
# You can manually override a "runaway" or "infinite" loop by stopping the current
# command in the console (red stop button, top right)
# Be careful. If the loop uses up all of your processor and all of your RAM,
# it might not let you do that. 
# In general, you should do everything "programmatically" - in other words, 
# the program should control itself, not rely on user input  

# A while loop nicely illustrates the "strangeness" of computers. 
# The computer will do exactly as told. It will never think to question it.
# If you - unintentionally - tell a computer to do something forever, it will. 
    

#%% 1b) Self-regulating infinite loop
weirdThreshold = 10000 # set threshold to identify that something went wrong
counter = 0 # initialize counter
while 0 != 1: # while 0 is not equal to 1 (this condition is always true)
    if counter > weirdThreshold: # If this has gone on for too long
        print('Something weird is going on. You might be stuck in a runaway loop.')
        print('We terminated execution of the loop for safety reasons. Check your code.')
        break # this terminates the loop
    counter = counter + 1 # incrementing the counter
    print(counter) # print counter status for each iteration
    
# This is self-regulating code. Just make sure the break statement is
# actually in the right place. If the if condition is never reached, e.g. by
# moving the incrementation of the counter inside of the if statement, if
# will never be run. Try it - by moving line 55 inside of the if statement.

# The computer will not tell you about this, because it is not a syntax
# error. It thinks that this is what you wanted. It takes you 100%
# literally. It doesn't get intentions. It takes the code at face value. 

# The other problem with while statements is that it is possible to write conditionals that are
# never true, so the loop is never executed. Suggestion for exploration: Change the condition in 
# line 50 to zero being larger than 1, which is never true and see if the loop runs.


#%% 1c) Have while loops do something useful with linear algebra:
import numpy as np # Import numpy to work with matrices

#Let's recall that the dot product between two vectors is 0, if the vectors are orthogonal.
#You can check this here:
# dot product = m1 * m2 * cos(angle2) %Result of trig identities
# Solve this for angle between
vec1 = np.array([0,0.5])  # vector v
vec2 = np.array([1,1])  # vector u
magVec1 = np.sqrt(vec1[0]**2 + vec1[1]**2) # magnitude of vector 1
magVec2 = np.sqrt(vec2[0]**2 + vec2[1]**2) # magnitude of vector 2
dotProduct = np.dot(vec1,vec2) # Using a function 
angleBetween = np.degrees(np.arccos(dotProduct/(magVec1*magVec2))) #What is the angle between the vectors?
print('Angle between vectors is:', angleBetween, 'degrees')

#Suggestion for exploration: Do this for other vectors to see if you get other angles
    
#%% If the dot product between two vectors is 0, the vectors are orthogonal
# How easy is it to find orthogonal vectors randomly - how rare are orthogonal vectors?
# Let's find 10 orthogonal vectors by random trial and error
orthoCounter = 1 # This is our counter of orthogonal vectors
loopCounter = 0 # This counts the number of loops it took to get them
numOrthoVectors = 10 # How many orthogonal pairs do we want to draw?
epsilon = 0.00001 #How close is close enough to 0?
rowContainer = np.empty([numOrthoVectors,2]) # Preallocate the row vector container
columnContainer = np.empty([numOrthoVectors,2]) # Preallocate the column vector container
while orthoCounter < (numOrthoVectors + 1): # We don't know how many tries we 
# need, so while is suitable
    A = np.random.normal(0,1,2) # Draw a row vector randomly from a normal distribution
    B = np.random.normal(0,1,2) # Draw a column vector randomly from a normal distribution
    testThem = np.dot(A,B.T) # Take the inner product
    loopCounter = loopCounter + 1 # Increment the loopCounter
    if abs(testThem) < epsilon: # Ideally, we want dot products that are exactly 0, 
    # due to Python's numerical precision, this will never happen, so we need 
    # to check for close enough to zero. 
        print('Orthogonal!') # Expression of joy
        rowContainer[orthoCounter-1,:] = A # Capture the row vector
        columnContainer[orthoCounter-1,:] = B # Capture the column vector
        orthoCounter = orthoCounter + 1 # increment the orthoCounter
print('It took ',loopCounter, 'repeats to find 10 orthogonal vectors') # Wow

#Suggestion for exploration: Try different values of epsilon

#%% Now that we found them, let's plot them
import matplotlib.pyplot as plt # Import Matlab plotting library
plt.plot() # Unless you call this explicitly and there is not one open, it 
# will open one for you. Here, it is good to open one explictly
# If you have more than one figure you draw to, it helps to open new figures
for ii in range(numOrthoVectors):
    ax = plt.subplot(2,5,ii+1) #Create a subplot and name it "ax", so we can reference it later
    plt.plot([0,rowContainer[ii,0]],[0,rowContainer[ii,1]],color='purple',linewidth=1) #Plot vector 1
    plt.plot([0,columnContainer[ii,0]],[0,columnContainer[ii,1]],color='magenta',linewidth=1) #Plot vector 2
    ax.axis('equal') #Set aspect ratio to equal. If you don't do this, angles will be orthogonal, but not look like it


#%% 1d) Since we are on the topic of linear algebra: Let's introduce nested loops by recreating matrix multiplication with nested loops
#Loops are a good way to understand what an algorithm is doing, step by step.
#Last time, we did this with dot products.
#Here, we try this with nested loops, to understand matrix multiplication:

A = np.array([[1,2,3],[4,5,6]]) # define 2x3 matrix A with these values
B = np.transpose(np.copy(A)) # define 3x2 matrix B as a transposed copy of matrix A   
#We would like to get matrix C as the result of a matrix multiplication of A*B
#As the dimensionalities match where the matrices touch, this should work, yielding a 2x2 matrix C
    
#Pseudocode of the algorithm    
# 1 Determine the number of rows of matrix A by using the length function
# 2 Determine the number of columns of matrix B by dividing the size by the length
# 3 Initialize a matrix with the right number of rows and columns with 0s.
# In the future, we will preallocate with nans because if we have a 0 in the
# resulting matrix, we don't know if one element wasn't assigned
# (overwritten) or whether the 0 is genuine. We don't hardcode - here or in
# general because we don't want this to work just for these specific
# matrices. 
# 4 Go through all the rows - you can call the counter variables anything
# you want. Make use of this, you don't have to call them i or j, like
# mathematicians do. Make it something good. Like rr for row
# 5 Go through all the columns, using cc as indices
# 6 Replace the suitable zero in matrix C with the correct dot product 
# 7 Show C at the end

numRows = len(A) #1
numColumns = int(np.size(B)/len(B)) #2
C = np.zeros([numRows,numColumns]) #3
for rr in range(numRows): #4
    for cc in range(numColumns): #5
        C[rr,cc] = np.dot(A[rr,:],B[:,cc]) #6
print(C) #7

# Suggestion for exploration 1: Try this for 2 other matrices 
# Suggestion for exploration 2: Compare the outcome of this algorithm with 
# the built-in matrix multiplication operator


#%% 1e) For the sake of completeness, here is vector projection, both mathematically and visually:
# Dot products can be used (as per lecture on essentials of linear algebra) to yield the length of the piece of v 
# in the direction of the unit vector u 
# (v * u) * u
# Magnitude = Length of projection times direction of unit vector, so
# v * u = p = magnitude of v times cos of the angle between
# p = magnitude of v * cos angle
# Then: p * u

#Using the vectors from 1c):

uVec = vec2/magVec2 # Making it a unit vector

p = magVec1 * np.cos(np.deg2rad(angleBetween)) # Getting the magnitude of the projected piece
projVec = p * uVec # That's the actual projected vector by getting the direction from multiplication with the unit vector

#Plotting it
plt.plot([0,vec1[0]],[0,vec1[1]],color='purple',linewidth=2) # Plot original vector 1 in purple
plt.plot([0,uVec[0]],[0,uVec[1]],color='blue',linewidth=2) # Plot the unit vector uVec in blue
plt.plot([0,projVec[0]],[0,projVec[1]],color='red',linewidth=2) # Plot vector 1 projected onto the unit vector in red

#Suggestion for exploration: As you change the vectors in 1c, how do these projections change?    


#%% 2a) Functions: logic and how to write them

# A. What is a function?:
# A function is something that takes an input and produces/maps it to an output
# Usually, I recommend functions to be as simple as possible:
# Take a bunch of inputs, do ONE thing to them to implement a computation, 
# then return the output of this process. 
# Why is it wise that one function does one thing? 
# Because of the concept of "modularity". Strong advice: Build up a toolkit of
# functions where each of them does ONE thing. Then you can mix and match
# extremely flexibly (have combinatorics work in your favor). FLEXIBILITY matters!

# B. Use cases - why use functions?
# Just like one - in principle - never has to use a script - one could type all commands 
# in the command line, one could never write functions and do everything with scripts. 
# However, there are several good reasons to use functions. 
# Here are the three main reasons to use functions:
# 1. If you have the same kind of code that you use over and over again and
#    want to deploy in a modular fashion (mix and match in different scripts
#    without copying and pasting, and while taking variable inputs)
# 2. If you want to work out some complicated code logic once, then debug it
#    and make sure it works in general (regardless of specific inputs), 
#    then forget about it (--> 1 line of code instead of lots of complicated code)
# 3. Not cluttering up the workspace with temporary variables

# C. Scope:
# Scripts have access to the "global" workspace.
# This is sometimes good - everything is in one place. Sometimes this is
# bad. Why? Very quickly, it will cluttered up with all kinds of stuff that
# is completely useless long term. It also makes it hard to see what matters.
# Regular scripts have a global "scope". They have access to the workspace
# and write to the workspace. 
# FUNCTIONS have a limited scope. They only know the inputs you pass to them and
# only output what you output from it. Other than that, whatever happens in
# the function stays in the function and they only know what you tell them. 
# So functions help to control information flow in the program and don't clutter up/mess 
# with the global workspace. Consider the global workspace sacred. Only put stuff
# there that matters globally. 

numRows = 100
numColumns = 5
A = np.random.uniform(0,1,[numRows,numColumns]) # Makes a matrix with 100 rows and 5 columns with random numbers from 0 to 1 from a uniform distribution

# Say you want to know what the smallest dimension, i.e. width, of A is 

# len will give you the number of rows (usually largest dimension):
numRrows = len(A)
print(numRrows)

# np.size will give the total number of elements (num_rows * num_columns):
numElements = np.size(A)
print(numElements)

# Note: Although it is true that A.shape will output a *tuple* of the dimensions
# of A, tuples are immutable and therefore must be converted to other data
# types in order to perform operations on it. Oh no.
# So it make make sense to write a function that computes the width
# It saves typing
# It saves thinking
# This is generally advisable for functions. I write functions where I work
# out the - usually complicated - logic ONCE, then test it (we'll talk about
# this), then package it as a function and then I forget about it. As long
# as I remember the function, I don't have to derive the logic again.

# There is no "width" function that applies to an array. 
# What is a length? The longest dimension
# What is a width? The smallest dimension

# Careful: 
# 1) The function only knows what you tell it. So even if the variable
# exists in the workspace, the function won't know that, unless you tell it.
# 2) You have to explicitly declare an output, otherwise, the function will run -
# not error out - but not produce an output.
# 3) Even though you might have made many temporary variables on your way to
# the output, you will not have access to them in general - outside of the
# function unless you output them as well. 

inputArray = np.copy(A) # let's copy and rename our previously definied array, A
# This function yields the smallest dimension of a matrix
# In analogy to "len", which yields the largest dimension
# Input: A matrix of arbitrary dimensionality
# Output: The smallest dimension
def width_func(inputArray): # and use it as the input to our function
    step1 = np.size(inputArray) # total number of elements
    step2 = len(inputArray) # total number of rows
    return int(step1/step2) # return total number of elements divided by number of rows
howManyColumns = width_func(inputArray) # use our function to output how_many_rows
print('Number of columns:',howManyColumns) 


# Let's build another useful function. 
# This time, we are interested in the digitsum of a number
# All we want to do is add up all the digits in a number. Easy.
# The digitsum of 17 = 1 + 7 = 8. You get the idea
# This is a good use case for a while loop, as we don't know how many digits the input has
num = 17 # input any integer greater than or equal to 10
def digitsum_func(num):
    temp = str(num) # convert int to a string
    while len(temp) > 1: # as long as it is not a single digit
        temp2 = 0 # initialize sum
        for ii in range(len(temp)): # go through all digits
            temp2 = temp2 + int(temp[ii]) # add the ith digit as a number to the cumulative sum
        return temp2 
digitSum = digitsum_func(num)
print('Our digitsum:',digitSum) # place at end of function to automatically print

#%% 2b) Calling functions

# So far, all our functions were written directly in the script
# To save space and for overall cleaner code, we can save them to a 
# separate script and then call them directly here.

# First, we need to copy our function to a new script
# Then, we need to save that script
# Note: make sure the function script is in the same directory as your main sctipt
# Otherwise you will have to navigate to it manually using Unix commands


# Now, let's revisit our width function, this time calling our function:
from width import width_func # from file (no .py) import function
inputArray = np.copy(A) # let's copy and rename our previously definied array, A
howManyColumns = width_func(inputArray) # use our function to output number of columns
print('Number of columns:',howManyColumns)  
    

# Now, let's revisit our digitsum function, this time also calling our function:
from digitsum import digitsum_func # from file (no .py) import function
num = 1982 # pick a new number
digitSum = digitsum_func(num) # use our function to compute the digitsum
print('Our digitsum:',digitSum) # print the results

# For the duration of this course we will be building our own functions
# so that you can compile your 'toolbox' of data science functions for 
# future use


#%% 3a) Mean Absolute Deviation (MAD) vs. Standard Deviation (SD): Simulation, as seen in lecture

# 0. Import relevant libraries:
import matplotlib.pyplot as plt # import matlab plotting library
from mean_absolute_deviation import mean_absolute_deviation_func # from file (no .py) import function

# 1. Preallocate variables:
numElements = 10000
numIterations = 30
M = np.zeros((numElements,numIterations))
S = np.zeros((numElements,numIterations))

# 2. Compute MAD and SD:
for jj in range(numIterations):
    X = np.random.normal(0,1,[numElements,jj+2]) # (mean, sd, [number of draws, number of columns])
    for ii in range(numElements):
        data = X[ii,:]
        M[ii,jj] = mean_absolute_deviation_func(data) # custom M.A.D. function
        S[ii,jj] = np.std(data) #Take the standard deviation, using the numpy function
    print(jj)
        
# 3. Plot data:
for ii in range(numIterations):
    # Subplot 1:
    plt.subplot(1,2,1)
    plt.plot(M[:,ii],S[:,ii],'o',markersize=.5)
    maxi = np.max(np.array([M[:,ii],S[:,ii]]))
    line = np.array([0,maxi])
    plt.plot(line,line,color='red',linewidth=0.5)
    plt.title('N = {}'.format(ii+2))
    plt.xlabel('MAD')
    plt.xlim([0,maxi])
    plt.ylabel('SD')  
    plt.ylim([0,maxi])
    # Subplot 2:
    plt.subplot(1,2,2)
    plt.hist(S[:,ii]/M[:,ii],100)
    meanToPlot = np.mean(S[:,ii]/M[:,ii]) #Take the mean, using the numpy function
    medianToPlot = np.median(S[:,ii]/M[:,ii]) #Take the median, using the numpy function
    plt.title('Mean = {:.3f}'.format(meanToPlot) + ', Median = {:.3f}'.format(medianToPlot))
    plt.xlabel('SD/MAD')
    plt.xlim(1,2)
    plt.ylabel('Relative Bin Count')
    ax = plt.gca()
    ax.axes.yaxis.set_ticks([])
    # Timelapse between each plot:
    plt.pause(.01)
    
    
#%% 3b) A closer look at MAD vs. SD
# Given a normal distribution, the MAD to SD ratio is 0.8
# This also means that the SD to MAD ratio is 1.25 (as shown in the simulation)
# Let's compute and plot the MAD against the SD for different values of the SD

# 0. Initialize parameters:
numStds = 100 # integer values, 1 to 100
numElements = 100000 # number of random numbers drawn per SD

# 1. Compute MAD for each SD (1 to 100):
meanAbsDev = np.zeros(numStds) # initialize array of zeros to store MAD
for ii in range(numStds): # loop through each SD value
     data = np.random.normal(0,ii+1,[numElements]) # draw numElements random numbers from a normal distribution
     meanAbsDev[ii] = mean_absolute_deviation_func(data) # then compute MAD

# 2. Print ratio of MAD to SD:
print('Ratio: ',np.mean(meanAbsDev/np.linspace(1,numStds,numStds)))
      
# 3. Line plot of MAD against SD:
plt.plot(meanAbsDev)
plt.ylabel('MAD')
plt.xlabel('SD')
plt.title('MAD vs. SD for normal distributions')


#%% 4) Basic file i/o - we'll do much more on this later. 
#For now - that's all you need to do for the problem set is to import data from a csv file with numpy:

sata = np.genfromtxt('firstSata.csv',delimiter=',') # load file as sata, 1000 rows, 2 columns

#The 4 chords of descriptive statistics: 
    
D1 = np.mean(sata,axis=0) # take mean of each column, across all rows
D2 = np.median(sata,axis=0) # take median of each column, across all rows
D3 = np.std(sata,axis=0) # take std of each column, across all rows

#Suggestion for exploration: numpy doesn't have a mean absolute deviation function. Other libraries that we will introduce later do.
#In the meantime, you could write your own. If you wanted to
    
#Now - just for the sake of completeness - the same, but for each row, across columns
D1b = np.mean(sata,axis=1) # take mean of each row, across all columns
D2b = np.median(sata,axis=1) # take median of each row, across all columns
D3b = np.std(sata,axis=1) # take std of each row, across all columns
