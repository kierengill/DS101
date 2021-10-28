# Python bootcamp session 4
# If-else statements, Aleatory calculations, r vs rho, the linear algebra view on correlation,
# law of large numbers, ergodicity and risk of run, blindfolded archers 
# Code by Pascal Wallisch and Stephen Spivack

#%% 1) If-else statements

# Pure if statements check only one thing. If/else statements check one
# condition *after* another. Possibility for logical mistake: If you always
# check in a certain order, the later checks will never be executed. We
# won't get there. Because the if-statement was already exited by previous
# condition being true. The if/else statement is exited as soon as *one*
# condition is true.

# Let's practice this with an example of human decision-making
# In our case, let's see how long it takes someone to change their mind
# about legalizing gambling
# Specifically, we are going to simulate both positive and negative info
# which together with "ad campaigns" will change the mind of the person

# 0. Import libraries:
import numpy as np # import numpy library
import matplotlib.pyplot as plt # import matlab plotting library

# 1. Initialize parameters:
decisionThreshold = 50 # The (arbitrary) point at which the person changes their mind
startingPosition = 0 # Our starting position
adCampaign = 1 # If we run an ad compaign, we can change their mind
maxTimeWindow = 1000 # max time (arbitrary amount) that we allow them to change their mind
mindChanged = 0 # This variable will represent whether they changed their mind
time = np.array([]) # This is where we keep track of timeCounter
state = np.array([]) # This is where we keep track of currentState 

# 2. Run simulation:
timeCounter = 0 # Start the clock at 0
currentState = startingPosition # Let's start with a reasonable starting point
plt.figure() #Open figure
while mindChanged == 0: # We want to measure time until the person changes their mind, so we run a loop that keeps going as long as the decision is unchanged
    # Each time going through the loop, we update the current state with
    # a new number. This is going to be somewhat artificial but not
    # entirely. There are lots of things going on that could change it.
    if currentState > decisionThreshold: # First, we check if the current state exceeds the decision threshold
    # If it does, we execute these commands - set mind changed state to 1,
    # which stops the loop and output that to the command line
        mindChanged = 1 # Now we change our mind!
        print('Changed my mind!')
    elif currentState > startingPosition: # If we didn't change their mind, we check if the starting position is larger than the current state
        # If so, we say that the person is trending to change their mind
        print('Trending to change my mind')
    elif currentState < startingPosition: # If they are not in a trending state, we check if it is below the starting position
        # If it is, we say that they are turned off from changing their mind
        print('Turned off')
    else: # If none of the other conditions is met
        print('My mind is unchanged') 
    if timeCounter > maxTimeWindow: # If they never change their mind
        print('My mind will not change. Are you sure you can convince me?') #Diamond hands, I guess?
        break # Terminate the while loop
    time = np.append(time,timeCounter) # Append timeCounter to time
    state = np.append(state,currentState) # Append currentState to state
    plt.plot(time[:],state[:]) # Plot the current state over time as we go
    plt.plot([0,timeCounter],[decisionThreshold,decisionThreshold],color='red',linewidth=0.25) # draw decision threshold as a line
    plt.xlabel('time') # arbitrary time units
    plt.ylabel('decision integration')  #Make sure to watch the plots evolve as this runs
    plt.show()
    plt.pause(.01) # in seconds - this is 10ms
    # Simulating ad campaign - it usually goes up, but it could go
    # down, because there are other things going on
    positiveInfo = np.random.randint(1,11,1) # This draws one random integer from 1 to 10, out of a virtual hat.
    negativeInfo = np.random.randint(1,11,1) # Same. For balance.
    integratedDecision = positiveInfo - negativeInfo + adCampaign # This allows for negative numbers which makes it more realistic. People receive both positive and negative information
    currentState = currentState + integratedDecision # Update the current state
    timeCounter = timeCounter + 1 # 1 unit of time passes

# Suggestion for exploration: Change the values for adCampaign and/or the decisionThreshold to see how the result is affected


#%% 2 Correlation simulation 1: Aleatory calculations

# 0. Import libraries
from scipy import stats # We need scipy to compute Spearman's Rho (remember: numpy and matplotlib are already imported)
    
# 1. Initialize parameters:
numReps = 1000 # Number of experiment repeats (to see the average correlation)
numGears = 100 # We are looping through 1 to 100 gears
m = 10 # Number of mutually exclusive events (gear slots)
empVsExp = np.empty([numGears+1,4]) # Initialize container to store correlations: perfect world and simulation
empVsExp[:] = np.NaN # Convert to NaN
counter = 0 # Initialize counter
  
# 2. Run simulation:  
for c in range(numGears+1): # Loop through each gear (from 0 to 100)
    
    # Simulate aleatory observations:
    ratio = c/numGears # ratio of relative to total (proportion of gears in 2nd observation that change)
    observation1 = np.random.randint(m+1,size=(numGears,numReps)) # first observation; m slots (mutually exclusive events) per gear
    observation2 = np.copy(observation1) # second observation - same as first
    observation2[0:c,:] = np.random.randint(m+1,size=(c,numReps)) # randomly change c gears in second observation
    
    # Compute Pearson R for each experimental repeat:
    temp = np.empty([numReps,1]) # initialize empty container to store each r value
    temp[:] = np.NaN # convert to NaN
    for i in range(numReps): # Loop through each experimental repeat
        r = np.corrcoef(observation1[:,i],observation2[:,i]) # compute the Pearson R
        temp[i] = r[0,1] # store coefficient in temp variable
    
    # Compute Spearman Rho for each experimental repeat:
    temp2 = np.empty([numReps,1]) # initialize empty container to store each rho value
    temp2[:] = np.NaN # convert to NaN
    for i in range(numReps): # Loop through each experimental repeat 1 to 1000
        r = stats.spearmanr(observation1[:,i],observation2[:,i]) # Compute Spearman Rho
        temp2[i] = r[0] # store coefficient in temp2 variable
    
    # Store data:
    empVsExp[counter,0] = ratio
    empVsExp[counter,1] = np.mean(temp) # take mean R for all experimental repeats
    empVsExp[counter,2] = 1 - ratio
    empVsExp[counter,3] = np.mean(temp2) # take mean rho for all experimental repeats
    counter = counter + 1 # Increment the counter
    
    # Plot data:
    plt.plot(sum(observation1),sum(observation2),'o',markersize=.75)
    plt.title('Ratio = {:.2f}'.format(empVsExp[c,0]) + ', r = {:.3f}'.format(empVsExp[c,1]) + ', rho = {:.3f}'.format(empVsExp[c,3]))
    plt.xlim(300,700)
    plt.ylim(300,700)
    plt.pause(.01) # pause (in seconds) between iterations
    
#%% Ratio of R vs. Rho as a function of increasing correlation:
ascendingMatrix = np.flipud(np.copy(empVsExp)) # copy array and flip upside down
plt.plot(ascendingMatrix[:,1]/ascendingMatrix[:,3]) # Hint: It stabilizes - 
# run 2 multiple times to confirm this. More unstable at lower correlations, as ranks will jump more
plt.title('Ratio of r vs. rho as a function of increasing correlation')
    

#%% 3) Correlation simulation 2: What is going on with the spearman correlation by using random ranks
# Instead of using scipy let's manually compute Spearman's Rho. By hand. To understand it. Just this once.


# 1. Initialize variables:
scaleFactor = 6 # Try different ones to determine that it has to be 6 to map from -1 to 1
numReps = 10000 # Number of repetitions
maxRanks = 50
data = np.empty([maxRanks-1,numReps,5]) # Initialize the 3D array (#stack,#rows,#columns) to put the data here - ranks are from 2 to 50, for a total of 49 unique ranks
data[:] = np.NaN # Convert to NaN
counter = 0

# 2. Run simulation:
for r in range(2,maxRanks+1): # Number of ranks involved - this increases by 1 for each iteration
    for i in range(numReps): # Loop through each rep of a given rank - 10000 reps per r
        temp = np.random.permutation(r) # Randomly permute the ranks (shuffes numbers from 2 to r)
        temp2 = np.random.permutation(r) # Do that again
        d = temp - temp2 # Calculate the rank differences
        dS = d**2 # Square the rank differences. If negative, this yields a sequence of odd squares
        Sd = sum(dS) # Sum the squared rank differences
        numerator = scaleFactor * Sd # Mulitply by the scale factor -> numerator
        denominator = r*(r**2-1)  # Play around with not squaring it, or go +1 instead of -1
        pos = numerator/denominator # How large is the positive part - if it is larger than 1, correlation will be negative
        rho = 1-pos
        # Store data:
        data[counter,i-1,0] = Sd
        data[counter,i-1,1] = numerator
        data[counter,i-1,2] = denominator
        data[counter,i-1,3] = pos
        data[counter,i-1,4] = rho
    counter = counter + 1 # Increment counter to keep track of the stack

# 3. Plot data (for an arbitrary number of ranks involved):
counter = 0 # initialize counter
meanAbsValue = np.empty([maxRanks-1,1]) # this is where we store abs of all rhos for a given r
meanAbsValue[:] = np.NaN # make sure to convert to NaN
for r in range(2,maxRanks+1): # Loop through each stack (2 to 50)
    tempData = data[counter,:,:] # take the stack that corresponds to r
    meanAbsValue[counter] = np.mean(abs(tempData[:,4])) # take the mean abs value of all rhos
    plt.hist(tempData[:,4],bins=51) # plot histogram and specify bin count
    plt.title('Number of ranks involved: {}'.format(r)) # add title
    plt.pause(0.1) # pause (in seconds) between iterations
    plt.xlim([-1,1]) # If you try different scale factors, you have to expand this too
    counter = counter + 1 # Increment counter to keep track of the stack


#%% Average correlation *magnitude* (regardless of sign) as a function of numbers involved
# This is the correlation *magnitude* you can expect if you just draw random
# numbers and correlate them, as a function of the numbers involved. This
# has to be taken into account when assessing any correlation. Because this
# is the effect of chance. It is easy to see why it would be 1 for 2
# numbers, and if they are ranks, because it is either 1 2 vs. 1 2 (rho = 1)
# or 1 2 vs. 2 1 (rho = -1). It's a line.
plt.plot(meanAbsValue)   
plt.title('Correlation magnitude expected by chance as a function of the number of pairs')
plt.xlabel('Number of pairs involved in the correlation')
plt.ylabel('Correlation magnitude')


#%% 4) A linear algebra view on correlation
# As you now know, the correlation between 2 variables can be interpreted as
# a relationship between elements (the x- and y- variables can be
# interpreted as coordinates in 2D):

mu = 0
sigma = 1
X = np.random.normal(mu,sigma,100)
Y = X + np.random.normal(mu,sigma,100)
temp = np.corrcoef(X,Y)
r = temp[0,1]
plt.plot(X,Y,'o',markersize=.75)
plt.title('r = {:.3f}'.format(r))

# As usual, linear algebra provides an alternative view that is completely
# consistent with the classical view, but can provide enlightening to some.
# In this view, we are looking at the relationship between 2 (two)
# 100-dimensional vectors. For those, the correlation between them is given
# as their dot product. If they are unit vectors. So lets' reduce them to
# unit vectors first. Then take the dot product.

xUnit = X/np.linalg.norm(X) # Convert x to its unit vector
yUnit = Y/np.linalg.norm(Y) # Convert y to its unit vector
rVec = np.dot(xUnit,yUnit) # Take the dot product
print(abs(r-rVec)) # Close enough - within numerical precision


#%% 5) The law of large numbers

# a) Setup:
trueValue = 7 # This is the true value
noiseLevel = 1 # This is the noiselevel
numSamples = 1000 # This is how often we measure
measuredValues = trueValue + np.random.normal(0,1,[numSamples,1]) * noiseLevel # The measurements, contaminated by noise

# b) Running the simulation:
meanSamples = np.empty([numSamples,1]) # Initialize mean samples
meanSamples[:] = np.NaN # convert to NaN
stdSamples = np.empty([numSamples,1]) # Initialize std samples
stdSamples[:] = np.NaN # convert to NaN

for ii in range(numSamples): # Integrate samples from 1 to however many there are
    randomIndices = np.random.randint(0,numSamples-1,[ii+1,1]) # Determine which values we will take the average over
    meanSamples[ii,0] = np.mean(measuredValues[randomIndices]) # Take the mean over the measured values
    stdSamples[ii,0] = np.std(measuredValues[randomIndices]) # Take the std over the measured values

# c) Plotting it
plt.subplot(1,2,1)
plt.plot(meanSamples)
plt.xlabel('Number of samples')
plt.ylabel('Mean value of measurements')
plt.title('Mean')
plt.subplot(1,2,2)
plt.plot(stdSamples)
plt.xlabel('Number of samples')
plt.ylabel('Standard deviation of measurements')
plt.title('Standard deviation')

# Note how the value of the mean quickly approaches the true value, regardless of noise. 

# Suggestion for exploration: Dial up the noise level and note how it still
# converges, but more slowly. Experiment with different noise levels and
# increase the number of samples to integrate over, in order to compensate.


#%% 6) Ergodicity (or the lack thereof) and the risk of ruin

# Here, we will explore the statistical basis of virtue. 
# Many real life processes are not ergodic. However, short term incentives
# might lead organisms to adopt strategies that are not adaptive in
# the long run.

# This could model all kinds of behaviors. Lying, cheating, stealing, taking risks
# that could lead to physical injury or financial ruin, etc.

# Setup:
payoffWithoutRecklessAction = 50 # Expected payoff per attempt without being reckless
payoffWithRecklessAction = 100 # Expected payoff per attempt with being reckless (e.g. cheating)
attempts = 200 # How many times is the action performed (e.g. cheating)
    
# Expected values with ergodicity (no risk of ruin):
expectedValueVirtue = attempts * payoffWithoutRecklessAction # In utility points
expectedValueReckless = attempts * payoffWithRecklessAction # This could be anything. Pleasure, money, etc.

#Looks like being reckless pays - in ergodic systems
print(expectedValueReckless > expectedValueVirtue) # Oh no

# But - and this is an important but - most real life systems are not
# ergodic. Meaning: You can get caught. Getting caught doesn't only lead -
# usually - to punishment (being shamed, imprisoned, etc.), but also
# nullifies the ability to keep on playing. For instance, being seriously injured
# midway during an attempt to win a competition prevents the player from
# competing. We are also not considering the damage reckless behavior
# inflicts on others. So the real life outcomes are actually considerably worse.
# Here, we only consider the direct expected outcomes when continuing to play the game
# (whatever the game is). Note that in real life, "ruin" could also nullify
# enjoyment of prior gains (e.g. in cases where ruin = death). 


#%% Let's simulate this:
    
riskOfRuin = 0.05 # How likely is one to get caught performing the reckless action (e.g. cheating) per attempt?
runs = 1000 # How often are we running the simulation

# Note that we have to only simulate the reckless behavior, as the virtuous
# organism just gets the expected value (no risk of ruin). Which we already
# calculated above

sata = np.empty([runs,attempts]) # Setup sata matrix
sata[:] = np.NaN # Convert to NaN

for ii in range(runs): # iterate over each run
    ruinedYet = 0 # Have I been ruined yet? This needs to be initialized here, at the outer loop, before the inner one
    for jj in range(attempts): # iterate over each attempt
        ruinedThisAttempt = np.random.uniform(0,1,1) # Drawing a number from a uniform distribution between 0 and 1
        if ruinedThisAttempt < riskOfRuin: # If this number is smaller than the cutoff, we were ruined (e.g. caught) in this attempt
            ruinedYet = 1 # Now we have been ruined
        if ruinedYet == 1: # If we are ruined
            sata[ii,jj] = 0 # Payoff this attempt - we can no longer play if we are ruined. 
            # Trust is hard to regain once lost. We might not get the chance. 
            # Note that in real life, this value could be negative, if there are punishments. 
        else:
            sata[ii,jj] = payoffWithRecklessAction # We can get the - usually higher - payoff of being reckless 
            
#%% Plotting the expected outcomes of being reckless

plt.plot(np.mean(sata,axis=0),color='red',linewidth=1)
plt.plot([1,attempts],[payoffWithoutRecklessAction,payoffWithoutRecklessAction],color='blue',linewidth=0.5)
plt.xlabel('Attempts')
plt.ylabel('Expected payout per attempt')
plt.legend(['Reckless','Ethical'])
# The area under the curve is the expected long-term outcome of either strategy (ethical vs. reckless)      
            
# Which strategy is superior now?
actualExpectedValueReckless = sum(np.mean(sata,axis=0)) # Expected value of being reckless under non-ergodic conditions        
print(expectedValueVirtue > actualExpectedValueReckless) # Now, being reckless underperforms 

# Enlightened self-interest: Understanding that ethical values can protect
# your *own* long-term self-interests (being led down the primrose path by
# your (intrinsic or extrinsic) reward system). 

# Suggestion for exploration: Play around with different value of risk of
# ruin to titrate risk. The reckless strategy *can* make sense if there are
# only *very* few attempts, under special situations.
# There are historical examples where this could makes sense. 
# If you are interested, read up on - for instance - the "Fornlorn Hope".
# Or on Doppelsöldner.


#%% 7) Able Archer 83

# A blindfolded archer stands on a rotating platform
# The platform smoothly rotates from -90 degrees to 90 degrees
# The archer fires - in random intervals, but at a given firing rate - at a target
# straight ahead. 

# The target is placed on a screen that is 18 m (the standard competition
# distance of target archery) away and infinitely wide (apparently, this is
# taking place on a flat earth too, so we can disregard the curvature of the
# earth). In fact, this seems to be taking place in space, as we would like
# to disregard the effects of gravity and friction from air as well (as you
# can see, the crux of simulations is that their results often rely on 
# many simplifying assumptions which might not reflect reality). 
# The figure of merit is the distance of where the arrow hits the screen
# relative to the target, in the horizontal direction. 

targetDistance = 18 # Distance from target in m
numCycles = int(1e3) # How many cycles of platform rotation are we modeling? 
firingRate = 1 # How often - per rotation cycle - does the archer fire?
granularity = 1e-3 # What granularity of platform rotations (in degrees) are we modeling?
startAngle = -90 + granularity
stopAngle = 90 - granularity
numSteps = int((abs(startAngle) + stopAngle)/granularity + 1)
oneCycle = np.linspace(startAngle,stopAngle,numSteps) # spatial base of one cycle
# Note - we don't let it go all the way to -90 or 90, as the archer is then
# firing parallel to the screen, which means it would never hit it

# Determining when in the cycle the archer fires - note that we would
# ideally want to model firing intervals as a poisson distribution, as the
# *rate* is constant, and the events are rare. But that's not the primary point of this segment, 
# so for the sake of brevity, we just have the archer fire once per cycle.
# Which is more regular than one would get from Poisson. 
# Suggestion for exploration: Replace this part with a poisson distribution
totalShots = numCycles*firingRate # How many total shots at target?
whereInCycle = np.random.randint(0,numSteps,totalShots) # Where in the cycle do all shots happen?
conversionCycleTimingToDegrees = oneCycle[whereInCycle] # Use those as indices to the cycle

# Now that we have the shot angles, we have to think a bit. Conjuring up
# our trig knowledge from middle school (who knew it would ever come in
# handy to help simulate blindfolded archers firing randomly in space?):
# You can either make a drawing or trust me that the distance from the
# target can be gotten as distanceFromTarget = tan(shotAngle)*targetDistance
    
# Note: We have to convert radians to degrees first. Because degrees are
# arbitary (which mathematicians detest) and radians are not, as every
# circle has - by definition - a radius and a circumference.
# 1 radian = a radius length put along the circumference of the circle

shotAnglesInRadians = np.radians(conversionCycleTimingToDegrees) # Conversion from degrees to radians
distanceFromTarget = np.tan(shotAnglesInRadians)*targetDistance # These are all the distances
numBins = 10000
plt.hist(distanceFromTarget,numBins) # Plotting this
# Oh no. This looks like a cauchy distribution

# Are measures of central tendency and dispersion converging for this process? 
sumStats = np.empty([totalShots,3]) # initialize a 1000x3 empty container
sumStats[:] = np.NaN # Convert to NaN
for ii in range(totalShots):
    sumStats[ii,0] = ii + 1
    sumStats[ii,1] = np.mean(distanceFromTarget[:ii+1])
    sumStats[ii,2] = np.std(distanceFromTarget[:ii+1])

# Plotting that - note the wild jumps, not that they are subtle. Looks like
# eye movement traces, tbh
# Suggestion for exploration: If you are unconvinced, increase cycles,
# firing rate and/or reduce granularity
plt.subplot(1,2,1)
plt.plot(sumStats[:,0],sumStats[:,1])
plt.title('Mean')
plt.xlabel('Integrating over n samples')
plt.subplot(1,2,2)
plt.plot(sumStats[:,0],sumStats[:,2])
plt.title('Standard Devitation')
plt.xlabel('Integrating over n samples')

# Suggestion for exploration: Look up what "Able Archer 83" really is/was. 
# Then thank your lucky stars (or rather, Stanislav Petrov, that you're still here)
