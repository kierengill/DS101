# Lab session 2
# Objective: Understanding program flow (for loops and if statements), some probability and some linear algebra
# Code by Pascal Wallisch and Stephen Spivack

# Last time, we introduced the numpy library and the array data type.
# In our probability distribution example at the end, we manually spun a probability gear 10 times.
# Oh no - how tedious!
# What if we wanted to do this 100 times, 1000 times or even more than 9000 times?
# Anything that is repetitive in nature, you should not do by hand.
# You should let the computer do that for you. That is the whole point of coding. 
# So, this allows us to naturally pivot to the concept of loops.

# Thus far, we have only used the editor as a repository for commands.
# We could have typed them all in the command window. 
# We simply used the editor to keep track of things and to quickly re-run a set of commands. 
# Now, we will introduce what is called "program flow" or "program control".
# Program control is all about applied logic, and doing so very, very fast. 
# In practice, two forms of program control are particularly fundamental: 1) loops and 2) switches
# Let's start with for loops

#%% 1 For loops
    
import numpy as np  # import the numpy library - note: You need to do this only once,
                    # so we do it before playing around with loops

# A for loop literally just executes a set of commands repeatedly, as many times as you want

# a) The simplest possible loop:
startNum = 1
endNum = 100
steps = 100    
consecutiveNumbers = np.linspace(startNum,endNum,steps) # creates an array of integers from 1 to 100

for ii in consecutiveNumbers: # declare a loop to iterate over each item in numbers
  print(ii) # print the number in the command window after each iteration
# Output is 1, 2, 3, ... - for everytime we go through the loop
# The colon (:) indicates the start of the loop

#To unpack this loop, we are going through all statements in the loop (for now, only line 35) repeatedly
#How often? As many times as is indicated in line 34. Note that the words "for" and "in" are reserved keywords, 
#rendered in purple in Spyder. This means that we go through all elements in the consecutiveNumbers array, updating
#the indexing variable ii, as we go. In other words, the first time we go through the loop, ii will be assigned the value
#1. The second time we go through the loop, the same statements (line 35) are executed, but ii will now have the value 2
#and so on, until the end of the consecutiveNumbers vector is reached. So looping implements the classic philosophical 
#notion of "mutatis mutandis" - repeat with the necessary adjustments. 
#Updating the indexing variable is that necessary adjustment. 

# Aside on naming of indexing variables: Mathematicians like to use i and j as indices. 
# This is presumably in an attempt to highlight the arbitary nature of the index. 
# However, in coding, this can get you into trouble: 
    #1) Many people - when coding at night - have confused i with 1, which will lead to unexpected behavior
    #2) In several technical languages (although not Python), i and j are already pre-defined as an imaginary number
    
# For these reasons, we suggest (although you can do whatever you want) to never use "i" or "j" as indices. 
# We use ii and jj instead. In fact, we usually use a descriptive name, such as rr or cc to indicate that 
# we loop over rows or columns. This is not a big issue right now, because we only have 1 loop, but soon
# we will use nested loops, and getting confused about one's loop level is a primary cause of logical errors.

# Aside on what constitutes a loop: In Python, all indented statements are part of the same loop. 
# So unlike many other languages, loops do not need to be closed with an "end" statement in Python. 

# Suggestion for exploration: Create loops that start at a different number, end at a different number
# and have different number of steps; can you loop through all odd numbers? Even numbers? 


#%% b) Exercise a) was just to illustrate the concept of a loop at its most bare
# Let's now make a loop that does something useful. How about we use a loop to compute a cumulative sum?
# Cumulative sum: Adding up all the numbers we've seen so far
# We need to declare a new variable that represents the cumulative sum, which we will call cumSum 
cumSum = 0 # Declaring the variable to keep track of the cumulative sum
# Note that  we initialize it as 0 because we start from nothing (ex nihilo, although it is controversial whether zero qualifies as such)
# Also note that - importantly - we need to initalize OUTSIDE and BEFORE the loop to avoid logical errors
for ii in consecutiveNumbers: # declare a loop to iterate over each element in consecutiveNumbers
    cumSum = cumSum + ii # add the current value of the updated indexing variable to the cumulative sum
    print(cumSum) #Output current value of the cumulative sum to the console

#So far the loop. Note that every for loop can be replaced with a closed form expression. In this case, 
#we can use a well known equation to check if our loop did what we wanted it to. If you don't know of 
#such an equation, use simple test cases (e.g. cumulative sum of 3 numbers) instead.    
safetyCheck = ii*(ii+1)/2 #5050 - checks out -  Gauss would be proud
print(safetyCheck)

# Strong suggestion for exploration: Take line 70 and put it inside of the loop - right after line 73
# You will realize that the cumSum is no longer correct. Instead, every time we go through the loop, we 
# start from 0 - and add the current value of ii to it. Of course this would be a bizarre thing to do, 
# but Python doesn't glean your intent - it does exactly what you asked of it. 
# Even worse, you could insert line 70 after the computation in line 74, but before line 75. That way, you would
# nullify the value of the variable just after you computed it, but the output to the console would always
# be 0. Try it. Python will never question you. That is good on some level, but presumes that you 
# know what you are doing.   
# Beware! Initializing the counter at the wrong level of the hierarchy is
# one of the top 5 causes of logical errors. Again, logical errors are the worst
# kind of error because Python is doing what you told it to do, it's just
# not what you thought it was. And it won't tell you, because it doesn't
# know what you wanted to do. 
# Solution: *Always* check that your code did/does what you think it should
# do. Unlike society - which necessitates trust - there is no trust in coding.
# Check everything, then check again. Maybe have someone else check too? 

# What if we had never initalized cumSum to begin with? 
# This will either trigger a syntax error (Python will complain that it can't add to something undefined),
# or - even worse - if you ran a program with the same variable name previously, its value will
# still be floating around in memory. Oh no.
# Note this would not trigger an error - try it: Comment out line 70 after having run the program before
# The cumSum variable will no longer reflect the cumulative sum, but simply add to its previous value. 
# Hidden problems like that sink projects, papers and prospects.    
# Thus, *always* initalize your variables, and always do this at the right point in
# the loop hierarchy. Usually, this is before the loop begins.



#%% c) After introducing the concept of a loop and doing a toy example, let's do something useful
# Say you work in the user interface design department of a big company.  
# Your job is determine how fast operators respond to an incoming signal by pressing a button 
# You want to know the total reaction time (how long it took to respond to all signals). 
# You also want to know how reaction time evolves by plotting it over time.
# In principle, there are three possibilities - the operator might be getting faster over time,
# due to practice. Or maybe getting slower due to fatigue. Or the effects might cancel out. 
import matplotlib.pyplot as plt # import pyplot from the matlab plotting library, calling it plt - we'll explain how it works later, for now, we need this to plot
numTrials = 100 # number of "trials" the operator does (responses to incoming signals) 
lowerBound = 0 # Lower bound of interval we draw from, here: 0 seconds - in this universe, time runs forward
upperBound = 1 # Upper bound of interval we draw from - no one is too slow
rt = np.random.uniform(lowerBound,upperBound,numTrials) # simulate reaction times - drawn from a uniform distribution between 0 and 1 seconds
# I didn't bring any primate data today. If you're in a pinch and want to
# get realistic looking numbers, you often get it from a random number generator. 
# We will talk about randomness and where it comes from at length throughout this class.
totalRT = 0 # We use this variable to keep track of the total time the operator spent, initializing it with zero
data = np.zeros([numTrials,3]) # Initialize an array with 100 rows and 3 columns that will hold our data 
for ii in range(numTrials): #Start the loop (as indicated by the colon)
#Note that we use the built-in function "range" here, to create our indexing variable.
    #Now, here is what we we would like loop to do, each iteration:
    data[ii,0] = ii+1 # Put the current trial number (+1, as range starts from 0) into the 1st column, iith row of "data" 
    totalRT = totalRT + rt[ii] # Calculate the cumulative reaction time up to the iith trial
    data[ii,1] = totalRT # Put this value in the matrix at the right spot
    data[ii,2] = rt[ii]; # Keep track of the rt that was added in a given trial
averageRT = np.mean(data[:,2]) # This is the average reaction time across all trials
plt.figure() #Open a new plot - appropriately named command "figure", to indicate where this function came from (Matlab)
plt.plot(data[:,0],data[:,1]) # plot cumulative reaction time as a function of trial number 
plt.title('Average RT: {}'.format(averageRT) + ' seconds')
plt.xlabel('Trial number')
plt.ylabel('Total RT (s)')
#Note: The title is not wrong, but it's a lot. How about we introduce rounding? 
sigDig = 3 #How many significant digits do we want?
roundedRT = round(averageRT,sigDig)
plt.title('Average RT: {}'.format(roundedRT) + ' seconds') #Much better
#As we drew the reaction times from a uniform distribution between 0 and 1, we would expect the average reaction time to be
#close to 0.5 seconds and the cumulative reaction time to rise with a slope of 0.5 seconds per trial, on average.
#This seems to be the case here. 
#Suggestion for exploration: What happens to average and slope if you change lower and upper Bound (lines 121 and 122)?
#Also, if you reduce the number of Trials, does the average reaction time get closer to the expected value of 0.5 or 
#farther away? What happens if you increase the number of trials, say to a million?

#%% 2 Switches
# Let's mix switches into this toy example 
# If these numbers represent reaction times, we might only want to count valid
# ones. The operator might jump the gun - not react to the stimulus
# onset, but to their expectation of the stimulus onset, so we need some
# kind of filter where we say: "This number is probably not legit"
cutoff = 0.3 # Pick whatever you want, but I would be suspicious of faster times
#than that - people can only react so fast - 300 ms is not unreasonable

# What needs to be done to now only count "valid" trials, where the operator
# did not jump the gun? 

# We repurpose and modify our existing code, as follows:
# Briefly, before we do the "code surgery", here is what needs to be done logically:
# We need to place an if statement within the loop *before* doing anything
# else to check if the reaction time in the current trial is larger than the
# minimum cutoff we consider plausible (0.3, in this case)
# We also need to only increment the counter variable if the conditions are met,
# so below is the - modified - code from 1c, with switches thrown in at the suitable point:

rt = np.random.uniform(0,1,numTrials) # Same as before
totalRT = 0; #Same as before
data = np.empty([numTrials,3]) # Initialize the same array, but now empty 
data[:] = np.NaN # We now populate it with nans, which is better than 0s in this case. Exercise for reader: Why? 
validTrialNumber = 0; # This variable counts valid trials. 
for ii in range(numTrials): #Start the same counter as before
    if rt[ii] > cutoff: #But now - whoa - a switch! - implemented with an "if" statement, then the conditional, then a colon
        data[ii,0] = ii+1 # Same as before
        totalRT = totalRT + rt[ii] # Calculate the cumulative reaction time, as before
        data[ii,1] = totalRT # Same as before
        data[ii,2] = rt[ii] # Same as before
        validTrialNumber = validTrialNumber + 1

# Issue: If a value is less than the cutoff, it will be skipped, but we are
# indexing a later row of the data matrix with values that are valid. 
# This is why it was critical to index with nans, otherwise, the average reaction time would be lower, not higher.
# As we are taking out values lower than the cutoff, we expect the average reaction time of valid trials to rise

averageRT = np.nanmean(data[:,2]) # This is the average reaction time, but only for locations in data that are NOT nans
plt.figure() #Open a new plot
plt.plot(data[:,0],data[:,1]) # let's see how reation time builds over time
plt.title('Average RT: {}'.format(averageRT) + ' seconds')
plt.xlabel('Trial number')
plt.ylabel('Total RT (s)')

#Note that the average reaction of valid trials (only) has increased. This makes sense, 
#as we systematically eliminated short reaction times. Also note that there are now "holes"
#in the plot - as not every trial will have a valid time. We will address this issue - of how to 
#handle missing data with imputation later. For now, we have to learn to live with it.
#This is just an example - focused on how to use switches. 

#Suggestion for exploration: Change the cutoff value in line 159. How does the average value 
#systematically depend on the cutoff? Can you can write a nested loop where you increment the cutoff value 
#say from 0.05 to 0.95 and plot the average reaction time of valid trials as a function of that. Note that you might 
#want to run that with 1000 trials per cutoff, as so many will be eliminated for the higher cutoffs.

#%% 3 Some linear algebra 

#%% a) Vector length
arbVec = np.array([1,5]) # Define some arbitrary vector. This one is pretty arbitrary
magVec = np.sqrt(arbVec[0]**2 + arbVec[1]**2) # Magnitude of vector from Pythagoras

# Trust, but verify - always check whether your code does what is supposed to.
knownVec = np.array([3,4]) # This is used in every middle school class when introducing Pythagoras. We would expect 5 as the answer 
magVec2 = np.sqrt(knownVec[0]**2 + knownVec[1]**2) # Answer checks out. So this is legit.

# If you reuse code logic more than once, it is probably wise to write a
# function, *NOT* copy and paste (like we did here). We'll do that next time. 
#We will also generalize this to vectors with more numbers. 

#%% b) Creating a unit vector
newVec = np.array([7,4]) # Arbitrary new vector
magVec3 = np.sqrt(newVec[0]**2 + newVec[1]**2) # Same logic as before, lines 213 and 217
uniVec = newVec/magVec3 # compute the unit vector by dividing the vector through its length
magVec4 = np.sqrt(uniVec[0]**2 + uniVec[1]**2) # Checking that this worked - we expect an answer of 1

#%% c) Scalar multiplication of a vector
# Let's say we have our unit vector from before, but we now want to make it longer
scal = 5 #Scale factor
scaledVec = scal * uniVec #Implementing scalar multiplication
magVec5 = np.sqrt(scaledVec[0]**2 + scaledVec[1]**2) # Checking that this worked

#%% d) Creating a null vector by multiplying with 0
nullify = 0 #Zero defeats all
nullVec = nullify * uniVec #Multiplying with zero 
magVec6 = np.sqrt(nullVec[0]**2 + nullVec[1]**2) # Checking that this worked - the zero is undefeated

#%% e) Vector addition
vec1 = np.array([1,4]) #Create an arbitrary vector
vec2 = np.array([5,7]) #Create another arbitrary vector
vec3 = vec1 + vec2 # Add them

#%% f) Vector plotting 
# Plot each vector:
plt.plot([0,vec1[0]],[0,vec1[1]],color='purple',linewidth=0.25) # Plot Vec1 in purple
plt.plot([0,vec2[0]],[0,vec2[1]],color='blue',linewidth=0.25) # Plot Vec2 in blue
plt.plot([0,vec3[0]],[0,vec3[1]],color='red',linewidth=0.25) # Plot Vec3 in red
#Suggestion for exploration: Create different vectors in lines 241 and 242 and see what their sum plotted looks like

#%% g) The dot product
# Much of data science relies on the dot product in various guises. 
# You're invoking dot products implicitly every day, e.g. every time you do a web search
# We will explore these in later sessions. 
# For now, just an immediate illustration as to how it is useful, and what it is - intuitively. 
# Imagine you are running a web store and you made 302 sales last month. 
# Say you sell 5 different kinds of products. 
# We represent thes sales numbers of these 5 products with an array
productSales = np.array([100,50,50,75,27]) # Declare a variable to represent how many products of each kind were sold?
priceList = np.array([[5],[20],[3],[6],[1]]) # Declare a variable to represent how much each type of product costs in dollars?
overallRevenue = np.dot(productSales,priceList) # This is a dot product!
print('Revenue last month from all sales:',overallRevenue[0], 'dollars')
# Imagine you were running Amazon. In principle, you could represent all of their sales last months like this.
# And compute their overall revenue with 1 line of code.
# That is powerful.

#%% h) The dot product as a loop
# Whereas matrix operations like the dot product in line 262 are powerful - efficient and fast - it is often
# useful to implement the same operation with a loop. Not because it is faster, but because the logic is easier to understand.
# We are doing this here, using the same vectors from before - the sales vector from line 260 and the price vector from line 261
# It is also useful to write pseudocode to explain what a code segment does, like so:
# 1. Measure the number of elements of the vector. As they must the same number in both row and column vector, 
# it is ok to measure just one. Later, when writing error-checking code, we will check whether they match before doing anything. 

# 2. Preallocate memory to speed up assignment of products to resulting product vector. We will use this vector to capture the products we make. 
# In general, Python loses a *lot* of time if you don't preallocate. 

# 3. Go through all the elements of the vectors, one by one. What ii is changes every time you go through the loop.

# 4. For each iteration, multiply the element from the row vector with the corresponding one from the column vector and 
# assign it to the right place in the products vector, replacing the 0s we preallocated

# 5. Once we done looping, sum up all the products
numElements = len(productSales) #1
products = np.zeros(numElements) #2
for ii in range(numElements): #3
    products[ii] = productSales[ii] * priceList[ii] #4
dotProd2 = sum(products) #5
print('Revenue calculated this way is:',dotProd2,'dollars')

#%% 4) The Monty Hall problem: Implementing nested loops to solve a counter-intuitive problem
# In the Monty Hall problem, you are on a game show and asked to pick between 3 doors, one of which contains a prize.
# After making your initial pick, Monty Hall shows you a door you did't pick and gives you the option to switch.
# Assumption: Monty Hall is fair, i.e. he always gives you the choice to
# switch, not only if you're on the prize. Otherwise, this doesn't make sense.

# For the purposes of this example, we assuming the contestant always switches. 
# We want to know what the probability of winning is, given this behavior, i.e. if the contestant is
# switching. As the sample space is closed and known, we can calculate
# p(not switching) as 1-p(switching). Here, we go through all possible cases - all positions where the prize can be, and all doors the constestant can pick


# 1. Initialize parameters:
numDoors = 3 
trial = 0 # initialize trial counter to 0. Trial = each iteration (combination of prize door and contestant pick - there are 9 unique ones total)
data = np.empty([numDoors**2,5]) # Initializing container where Rows = trials. Columns = trial number, prize behind door, picked door, strategy (switch or stay), outcome (win or loss)
data[:] = np.NaN # convert empty to NaN - we could have done this initially by calling for nans to begin with
    
# 2. Compute all possible outcomes:
for ii in range(numDoors): # Put the prize behind all doors, one after the other
    doorStatus = np.zeros((1,numDoors)) # Reinitialize the doors for every run, in terms of closing them. 1 = open, 0 = closed. This variable represents door status
    prizeStatus = np.zeros((1,numDoors)) # Re-initialize the doors for every run, in terms of prize status, so this variable represents whether there is a prize behind a given door 
    prizeStatus[0,ii] = 1 # Actually putting the prize behind a given door, implementing the comment in line 311 
    for jj in range(numDoors): # Going through all initial picks by the contestant
        trial = trial + 1 # Keep track of the trial number
        pickStatus = np.zeros((1,numDoors)) # Reinitializing pick status with zeros for every nested loop run
        pickStatus[0,jj] = 1 # Actually making the contestant pick of a door
        temp = prizeStatus + pickStatus # Represent doors that are either picked or are doors where the prize is. Only zeroes here have neither prize, nor are open
        montyOptions = np.array(np.where(temp[0,:]==0))+1 # Find doors that Monty Hall can actually open (not picked, no prize)
        whichDoorsCanMontyOpen = montyOptions[0,np.random.permutation(len(montyOptions[0,:]))] # Randomly permute the possible choices Monty has
        whichDoorWillMontyOpen = whichDoorsCanMontyOpen[0] # Decide on the one he actually opens
        doorStatus[0,whichDoorWillMontyOpen-1] = 1 # Open a door, reveal that it doesn't have a prize, update door status
        # Now the participant switches:
        temp = pickStatus + doorStatus # Participant can't pick a door that is open (door status = 1) or already picked (pick status = 1)
        whichDoorCanParticipantSwitchTo = np.array(np.where(temp[0,:]==0))+1 # Which ones are available?
        pickStatus[0,jj] = 0
        if len(whichDoorCanParticipantSwitchTo[0,:]) > 0: # If there is a door to pick
            pickStatus[0,whichDoorCanParticipantSwitchTo-1] = 1 # Pick the door
        temp = pickStatus + prizeStatus # See if we have a match - if pick and prize matches, they should add up to a value of 2
        data[trial-1,0] = trial # What trial is this?
        data[trial-1,1] = ii+1 # Where was the prize put? 
        data[trial-1,2] = jj+1 # What was the initial pick by the contestant?
        data[trial-1,3] = 1 # Switched? This is what needs to be changed if we allow for more flexible strategies
        # Print and store results of each trial:
        if 2 in temp: # Is there a  winner? (temp == 2)
            print('trial',trial,'Winner!')
            data[trial-1,4] = 1 # Recording a win
        else:
            print('trial',trial,'You lost')
            data[trial-1,4] = 0 # Recording a loss
winningProportion = sum(data[:,4])/len(data) # Compute proportion of wins
print('Proportion wins given switching:', winningProportion)
plt.hist(data[:,4]) # Plot the data of number of wins vs losses, given switching - given all possible outcomes
        
#%% 5) The Monty Carlo Hall problem: A simulation
# Same as before, but now with many doors and many trials, playing at random (switching or staying)
# We reuse much/most of the code from the previous section, but we modify/generalize it. 
# This is common. You could call this Monty Python Carlo Hall

minDoors = 3 # How many doors do we implement minimally?
maxDoors = 20 # How many doors do we implement maximally?
numGames = 1000 # How many games are played per door number?
metaData = np.empty([maxDoors,3]) # Initialize a container for the data. Rows = trials. Columns = trial, prize, pick, strategy, outcome
metaData[:] = np.NaN # convert empty to NaN - nan is safe to initialize with here. 
# Let's reuse most of the code, just repurpose it a bit
for numDoors in range(minDoors,maxDoors+1): #Looping through the doors
    data = np.empty([numGames,5]) # Rows = trials. Columns = trial, prize, pick, strategy, outcome
    data[:] = np.NaN # convert empty to NaN
    print(numDoors) # Where are we now? How many doors are we up to?
    for game in range(numGames): # Play all the games, one after the next
        doorStatus = np.zeros((1,numDoors)) # Reinitialize the doors for every run, in terms of closing them. 1 = open, 0 = closed
        prizeStatus = np.zeros((1,numDoors)) # Re-initialize the doors for every run, in terms of prizes
        pz = np.random.randint(1,numDoors+1,1) # Where should the prize be put?
        prizeStatus[0,pz-1] = 1 # Actually putting the prize behind a given door
        pickStatus = np.zeros((1,numDoors)) # Reinitializing pick status
        pk = np.random.randint(1,numDoors+1,1) # What should the initial pick be?
        pickStatus[0,pk-1] = 1 # Actually picking a door
        for pd in range(1,numDoors-2+1): # Monty can open doors until there are n-2 left open - otherwise, the game doesn't work. So he opens one after the other
            temp = prizeStatus + pickStatus + doorStatus # Represent doors that are either picked or where the prize is, or that are already open
            montyOptions = np.array(np.where(temp[0,:]==0))+1 # Find doors that monty Hall can open (not picked, no prize)
            whichDoorsCanMontyOpen = montyOptions[0,np.random.permutation(len(montyOptions[0,:]))] # Pick the door he opens at random, out of the possible choices
            whichDoorWillMontyOpen = whichDoorsCanMontyOpen[0] # Decide on the one he actually opens
            doorStatus[0,whichDoorWillMontyOpen-1] = 1 # Open a door, and reveal that it doesn't have a prize.
            # Opening up all these doors will take time, so later iterations will take a lot longer. My fans just came on.
        # Now the contestant decides what to do, on the spot (this is new):
        contestantStrategy = np.random.randint(2) # 0 = Stay, 1 = Switch
        if contestantStrategy == 1: #If contestant switched
            temp = pickStatus + doorStatus # Participant can't pick a door that is open (door status = 1) or already picked (pick status = 1)
            whichDoorCanParticipantSwitchTo = np.array(np.where(temp[0,:]==0))+1 # Which one is it?
            pickStatus[0,pk-1] = 0
            pickStatus[0,whichDoorCanParticipantSwitchTo-1] = 1 # Move the pick
        temp = pickStatus + prizeStatus # See if we have a match!
        data[game,0] = game + 1 # What game is this?
        data[game,1] = pz # Where was the prize put?
        data[game,2] = pk # What was the initial pick by the contestant?
        data[game,3] = contestantStrategy # Switched?
        # Print and store results of each trial:
        if 2 in temp: # Is there a  winner? (temp == 2)
            #print('Winner!')
            data[game,4] = 1 # Recording a win - printing all of this would be overwhelming
        else:
            #print('You lost')
            data[game,4] = 0 # Recording a loss - so we just log it

    # Keeping track of everything in the  metadata:
    switchingIndices = np.array(np.where(data[:,3]==1))
    switchingWinningProportion = np.sum(data[switchingIndices,4])/len(switchingIndices[0,:])
    stayingIndices = np.array(np.where(data[:,3]==0))
    stayingWinningProportion = np.sum(data[stayingIndices,4])/len(stayingIndices[0,:])
    
    metaData[numDoors-1,0] = numDoors
    metaData[numDoors-1,1] = switchingWinningProportion
    metaData[numDoors-1,2] = stayingWinningProportion
    
# Note: Add a counter for the cumulative number of doors Monty opens to the code to do the problem set. 

#%% 6) Plotting the outcomes of the Monty Python Carlo Hall simulation
x = metaData[minDoors-1:maxDoors,0]
y = metaData[minDoors-1:maxDoors,1]
z = metaData[minDoors-1:maxDoors,2]
plt.plot(x,y)
plt.plot(x,z)
plt.xlabel('Number of doors')
plt.ylabel('Winning proportion')
plt.title('Monty Carlo Python Hall')
plt.legend(['Switching','Staying'])

# Suggestion for exploration - change lines 351 to 353 to explore how this impacts the simulation.
