# Lab session 1
# Objective: Introducing numpy and arrays
# Code by Pascal Wallisch and Stephen Spivack
# Date: 2-08-2021

# Everything after the pound/hashtag sign (#) is a "comment".
# Think of it as a "note to (your future) self". Python ignores everything after the # sign.
# Comments are rendered in gray by Spyder by default.

# This is the Spyder editor environment. Here is where you write programs.
# Programs are a collection of statements/commands that are interpreted and then executed by the computer.

# Python reads them from left to right, then from top to bottom.
# And then executes each command immediately. 

# This has 3 important implications:
# 1) Python is an "interpreted" language - different from a compiled language. More 
# commands to to read will take longer to execute. This won't matter today, but it does 
#matter if you have a lot of data to analyze.

# 2) Order of operations really matters - the machine does not glean your intent. It 
# executes exactly what you wrote, so you need to be careful about your commands. 
# It takes all commands 100% literal. 

# 3) Errors: This is a corrolary of 2) - if you don't fully appreciate this, it can lead to
# both logical and syntactical errors. 

# a) Syntax error: Make sure to first declare a variable, before you use it, like so:
A = 5
print(A) #'print' is a function. Functions take arguments in parantheses. Print outputs the contents of the argument to the console
#If you executed line 30 before line 29, Python would throw a syntax error - specifically it would complain that "name A is not defined" 
#Suggestion for exploration: Try this by deleting the variable "A" in the variable explorer to your right, then execute line 30 to experience the error

# b) Logical error: Sometimes the order of operations matters in the *algorithm* you are trying to implement. 
# In other words, Python will do exactly what you asked for - it will not throw an error 
# This is problematic because it will not be doing what you think it is doing. 
# Oh no. 
# Issues pertaining to unintended order of operations is one of the most common sources of logical errors. 
# For instance, some algorithms we'll encounter later presume that you sorted the values first, before determining some percentiles. 
# Reversing this order would yield the wrong answer. This sounds trivial, but we have seen people do just that in the past. 

#%% Double percentage sign after the hashtag opens a new "code section"
# Code sections are used to delineate logically different operations, e.g.
# loading of data, vs. analyzing it vs. plotting it.
# We strongly recommend to organize your code into logically self-contained sections or segments. 
# This will make your code much easier to maintain and modify ("code surgery").
# At a maximum, a segment should take up the entire screen. If it is longer than that, break it up into smaller segments. 
# You want to avoid scrolling within a segment, if you can help it. You'll see later (when we write larger and more complex programs, why) 
# It is a good idea to number the code segments. As this is the first time we are introducing the idea of a code segment, this one (that starts in line 42) has the implied number 0
# Code sections also help with running the code. You can run a whole section with shift-enter (on a mac)

# One more thing about comments: You can use them strategically, as Python
# will ignore them. So we would advise to never delete code, just to "comment it out" (in case you want to re-use a segment).
# If there is a large of portion of unused code, we suggest to collate that into a segment of commented out code at the bottom of your program. 
# We call this segment the "quarry" because you can mine it for code nuggets later.  

# How to use the console vs. the editor:
# This is just a recommendation, but here is what we do:
# The console (to the right in the default Spyder layout) executes ONE (1) line at a time. 
# So it is good for prototyping, i.e. to figure out what a given line of code is doing. 
# If we are unsure about the syntax of any given command, we use the console to debug the command until it works before transferring it to the editor.
# The editor (here), where you are reading this, is all about a ledger of commands, executed in line order (to the far left)
# So commands are building on themselves. 

# Finally, whereas we organized this code into sections, we strongly recommend to execute this code one line at a time, not the whole section at once, if you are going through this for the first time 
 
#%% 1 After explaining commenting and its recommended usage, let's continue where we left off with variables and commands:
A = 3 #Declare the variable "A" and assign the number "3" as its value 
#Interlude: In mathematics, the equal sign (=) represents equality (of the left and the right side). The idea is that 
#nothing can be more equal than two parallel lines (arguably - while well intentioned - they are doing this wrong, as one line is literally on top of
#the other, so || would be an even more equal equal sign), and this idea is central to the logic of mathematics (the notion of an "equation")
#However, in Python (and Matlab, for that matter), the equal sign does NOT represent equality. It represents the assignment operator. 
#In other words - and counterintuitively for western scripts - this reads as "take whatever is on the right side and assign it to the left side".
#Whatever was on the left side before is replaced (and lost) by what is coming in from the right side. 
#That is another reason why order of operations matters, once you assigned a value to a variable, its value might have changed, as follows:

B = 2 #Declare the variable B and assign the number to 2.
C = A + B #Create variable C as a sum of the values of A and B

#Note that all of these variables have been created with their respective values (confirm by looking at the "variable explorer" tab)
#If you want to output the value their value to the console, you have to explicitly ask for this, by using the "print" function, using the variable name as an argument:
print(C)    

#At this point, you might wonder what *is* the equal sign in Python, if it is not the = sign. 
#That is a valid concern. It is the double equal (==) sign in Python (and for that matter, in Matlab). 
print(A == B) #Testing for equality of A and B, and outputting the result to the console.
#The console should return a "Boolean" in response to line 84. 
#A boolean is a single binary digit (or bit), and can only take two values (0 or false) and (1 or true)
#Here, Python got it right - A currently holds the value of 3 and B holds the value of 2, and 3 is not equal to 2, so the test returns "false" - they are not equal 

#Let's update the value of variable B for the test to come true (for A and B to tbe truly equal)
B = B + 1 # This line also nicely illustrates the difference between equality (as in math) and an assignment operator. 
#If this was a mathematical equation, it would be necessarily unequal - B (on the left side) would always be smaller than B+1 on the right side (unless B is infinite, which is not a possible value in this reality - it is in math)
#However, this represents an assignment, so this reads as "take the current value of B (2), add 1 to it to yield 3, and assign this result to the left side, updating B.
print(A == B) #Now, A and B are the same value (3) and the test of equality reflects that - returning "true" to the console 

#Right now, lines 84 and 93 return only the most sparse output to the console, the result of the equality test itself. 
#We recommend to be more verbose than that - code amnesia (you quickly forgetting what a particular line of code is doing is real), like so:
print('Do variables A and B have the same value?',A==B) #Print also returns strings (the green words between single square quotes) to the console, and contextualizes the answer semantically. You can string them together with commas

#The assignment operator logic can also be used to implement recursive algorithms. We will do so later, but note that we can use the same command to further add to ('increment') B:
B = B + 1 #Now, B will be 4    
   

#%% 2 This is another code section - on naming variables
#In the previous section, we introduced the notion of variables and just used "A", "B" and "C" as names. 
#We actually discourage that when writing your own programs. You can name variables anything you want. 
#So make it something good and descriptive:
    
lascap = A + B + C #Illustrating that you can name variables anything you want. Here, we are accessing the global namespace. 
sampleSize = 69 #This is better - because the variable name reflects what the variable represents. Generally speaking, we recommend that, as it can save a comment. 
#To elaborate on this briefly, line 106 is better than 108:
size = 69 #We strongly discourage the use of simple variable names like that. Python will do it, but confusing variable names is another key source of logical errors (you think the variable refers to something it does not, and size could literally be the size of anything)
#One thing variable names cannot have is a space in between them:
sample size = 69 #Executing this line will throw a syntax error ('invalid syntax') in the console. Do not do this.
#Something else that is dangerous - and strongly discouraged is to use variable names that are already names of functions:
print = 69 #Say you want to make 69 prints of something and represent that with a variable. Python will let you do this. 
#But it is unwise. Because you can now no longer use the print function:
print(size) #Will give a somewhat cryptic error message. You have to delete the variable print in the variable explorer first, to make line 114 work again (shift-click on the variable on a mac, then "remove")

#One more thing regarding variable names. Python convention is to use the underscore (_) to create complex and descriptive variable names, like so:
sample_size_training_data = 100 #This is fine, but we consider that aesthetically unpleasant. You can do this. We recommend "camel case" instead (and use that in our own work):
sampleSizeTrainingData = 100 #Do whatever works for you, but this is what we do. It's called "camel case" in the art.     

#%% 3 Functions
# To borrow a metaphor from physics, if variables are your "matter", functions are your "forces"
# Functions operate on their inputs and yield an output. You can write your own functions, and we will teach you how to do so shortly, but for today, we will use pre-existing functions
# Python does not come with that many built-in functions, but there are a large number of libraries (libraries are collections of functions) that you can import and then use
# Importantly, you have to import the library first, before you can use the functions. This is another example of order of operations being critically important:
import math #This line imports the "math library" (a collection of standard mathematical functions)
# To call a function from this library, we need to first indicate the library name, then a period (.), then the function from that library.
# This reaches into a particular library before looking for the specific function name there. Later, we will introduce the concept of aliases to save on typing. 
math.sqrt(4) #Takes and outputs the square root of the input argument (4). Executing line 127 before 126 yields a syntax error. Try it?
math.sqrt(C) #Arguments don't have to be explicit numbers, here we pass a variable, and we don't have to freshly declare C, as C is still in the namespace, from before. 

#Generally speaking, there are many ways to achieve a given outcome in coding - which is why it is such a great way of creative problem solving - here we would like to highlight
#that the same outcomes can be achieved by mathematical operators as well as functions:
D = B**0.5 #Take the square root of B with an operator (Python uses the double asterisk ** to denote powers) and assign it to the variable "D" 
E = math.sqrt(B) #The same operation, but implemented with a function instead, and assigned to the new variable "E"
#Testing that this worked:
F = D==E #Do a test of equality for D and E and assign the outcome to the variable F, which will contain a boolean.     

#Here is another example - the absolute value function, which is "built-in" - already comes with Python - so we need don't to denote a library:
abs(2) #Taking the absolute value of 2, which is 2
abs(-2) #Taking the absolute value of -2, which is 2

#As we already discussed, and to emphasize, functions operate on and modify inputs to create outputs. 
#The arguments to functions are indicated by regular parantheses ( ).

# If you want to know how any given function works, call for help - another built-in function:
help(abs) #Help on the built-in absolute value function 
help(help) #Help on the built-in help function

help(math.sqrt) #Help on the square root function of the math library
#Of course - this presumes that you know what functions are even in the math library. How would you know that?
help(math) #By calling the help function on the library itself. This explains the library, as well as what functions and constants it contains
#Note that you have to import the library first, before you can call help on it

# Importantly, functions can be chained together, and if you chain them together, the
# "innermost" function - that is surrounded by most parantheses - will be executed first
math.sqrt(-2) # This will yield a complex number which cannot be computed by the math square root function - try it
math.sqrt(abs(-2)) # Keeping it real

# This is called "nesting" of functions. 
# Most statements you will write consist of combinations of variables and functions. 

#Note: Python also has a few in-built functions that are special insofar as they are reserved keywords that cannot be used as a variable name.
#They are by default rendered in purple in Spyder and usually affect the way Python interprets the next command, not just an argument. 
#We already encountered one example of this - the import function that does not need parentheses for the name of the library to import. 
#Another good example is the del keyword. del deletes a variable from the namespace. We could have used del in line 118 instead to delete the variable programmatically (without clicking), e.g.
del B #Deletes variable B from the namespace
   

#%% 4 Vectors, indexing and slicing
# So far, all our variables contained scalars (single numbers) only. Scalars are technically 0D tensors and are called that because if you multiply a vector by a scalar, you only change ("scale") magnitude, not direction
#That's fine to represent constants and the like, but to represent data, it is usually more efficient to do so with vectors and matrices.

# Vectors: There are multiple interpretations of this concept. In physics, they are usually used to represent forces (magnitude
# plus direction), in CS it is typically "just" an ordered list of numbers. In math, a vector is a stack of numbers (a 1D tensor).
# Importantly, a vector is a single stack, so the dimensionality of a vector is either rx1 or 1xc (r = number of rows), c = number of
# columns). What we are going to do in this class is compatible with all 3 interpretations. 

import numpy as np # import the numpy library, and call it the alias "np", to save typing.
#In the lab, we'll give a presentation on numpy, but it enables the kind of vector and matrix data types that basically allow you to do Matlab-like computations in Python  
firstVector = np.array([1,2,3,4,5]) #Both vectors and matrices are created with the numpy function "array". Here, we create a vector as a 1D array with explicit elements.
#There are other ways of doing this, but here, we do so by square brackets and elements separated by commas. 
# Square brackets denote an ordered set of numbers. We pass this set to the function as an argument within regular parantheses. 
#After running line 177, note via the variable explorer that an array of size "5," - denoting a single row with 5 columns with values [1 2 3 4 5] has been created.

#Suggestion for exploration: Create another vector with 3 or 7 elements, and elements that are not consecutive numbers.

#Indexing:
#In lab, we will give a presentation on how indexing works in Python conceptually, but meanwhile, we're going to do it in code first.
#Importantly - for historical reasons - Python *indexes from 0* (like most other programming languages, and unlike most scientific computing environments)
#If you are not a coder, it will probably take a while to get used to the fact that the first element in a vector is the 0th element:
accessingTheFirstElement = firstVector[0] #Reach into the vector and retrieve the first element, assigned to the variable on the left
print(accessingTheFirstElement) #Output what that element is to the console - this value will be 1.

#More about indexing in the lecture - for now, note that each element of a vector has both an index and a value. 
#To retrieve the value at a given position in the vector, you have to use its index. 
#You can also use indexing to assign a new value to an element
firstVector[0] = 17 #Changing the first element of our vector to the number 17

#Note, even though we changed the first element of the vector if you ran line 194, the contents of accessingTheFirstElement has not updated - this is not excel
#If you want to retrieve the updated value, you have to reach into the vector at this location again:
updatedValue = firstVector[0]

#Another common source of logical errors is to change some source array, but retrieve the value of the elements before the change was made. 

#Suggestion for exploration: Try to retrieve different elements and assign different values to these elements     
    

#Slicing: Slicing is a more generalized form of indexing, namely if you want to retrieve more than one element at a time.
firstSlice = firstVector[0:2] #Take a slice of the vector, from the 0th (inclusive) to the 2nd (exclusive) element, yielding a slice of size 2.
#Again, indexing and slicing is a bit wonky relative to scientific languages like R, Matlab or Julia. But we will have visuals that illustrate this nicely in the lab lecture
#Nevertheless, please practice this a lot and make sure to check that the output is what you think it is. Off-by-one errors (due to the "fencepost problem") is one of the most common logical errors in Python
secondSlice = firstVector[:2] #This will be the same slice as the first one, if you index from the left edge, you can leave that off

print(firstSlice == secondSlice) #Check that they are indeed the same, element-wise

thirdSlice = firstVector[1:4] #Take the middle 3 elements of the vector and put them in the thirdSlice varaiable
firstVector[1:4] = 7 #Change the middle 3 elements of the 5-element vector to the number 7.
#Note that executing this statement - line 220 - *did* auto-update the slices that rely on it (!), like in Excel (and unlike in Matlab)

#Suggestion for exploration: Change some other range of elements to other numbers and check how the slices that depend on it update automatically     

#Note that so far, we have created vectors explicitly, by specifying the specific values of their elements.
#In practice - because this is rather tedious if there are more than just a few numbers that make up the vector - you will rarely create vectors like that.  
#More often, you'll use enumeration functions:
numUsers = 42; # How many users?
userList = np.linspace(1,numUsers,numUsers) # This creates a vector of numbers from 1 to numUsers (inclusive), in numUsers steps

#Suggestion for exploration: Create a longer or shorter vector by changing the numUsers variable. Make a really long one and output it to the console with print

#linspace is a numpy function that creates n evenly spaced numbers from startnumber to endnumber, like the corresponding Matlab function linspace. 
#Another numpy function to create a vector of evenly spaced numbers is arange (not "arrange"). 
#For instance:
stepSize = 1
userList2 = np.arange(1,numUsers+1,stepSize) #start number, endnumber (+1 or it won't include the endpoint due to the fencepost problem), interval    
#So what is the difference? linspace created floating point numbers, whereas arange created integers (you can check this in "type" of the variable explorer):
A = userList[5] #user number 6 as a float
B = userList2[5] #user number 6 as an int
print(A==B) #But test for equality compares values, not variable types, so this will return "true"

#We can use this to create useful lists of numbers, for instance lists of odd and even numbers from 1 to n
n = 100 #What range are we looking for?
oddNumbers = np.arange(1,n+1,2) # wonderful
evenNumbers = np.arange(2,n+1,2) #exactly
#There are 50 odd and 50 even numbers from 1 to 100. Make sure to add the +1, or you'll miss the last even number.

#What is the 27th odd number in this range? 
oddNumbers[26] # What is the 27th odd number? (Remember that we index from 0!)
evenNumbers[9] # What is the 10th even number?

#Suggestion for exploration: Try other indices to retrieve specific odd or even numbers


#%% 5 Matrices and matrix operations
# If vectors are stacks of numbers, what are matrices?
# Matrices are stacks of vectors or 2D tensors.
# Importantly, both vectors and matrices are created with the numpy "array" function. 
# You can get the dimensionality of the tensor with the numpy function "ndim":
print(np.ndim(A)) #Dimensionality of variable A = 0
print(np.ndim(oddNumbers)) #Dimensionality of variable oddNumbers = 1

M = np.array([[1,2,3],[4,5,6]])  #Create matrix M with 2 rows and 3 columns
print(np.ndim(M)) #Dimensionality of variable M    

# The command in line 266 first created the two horizontal vectors and then stacked them vertically.
# We will use matrices mostly to represent data. 
# Convention is: Rows represent "replicates" (users, participants, mice, units of analysis)
# Columns represent "attributes" (reaction time, age, heart rate, accuracy)...
# To retrieve the number of rows and columns, we introduce a new concept. In python, all variables are "objects". 
# You can retrieve the "methods" of an object by typing its name, then a period in the console. Then tab will bring up a list of methods of that object. 
print(M.shape) #Outputs the shape (rows, columns) of matrix M into the console
M = M.reshape(3,2) #Turn M into a 3,2 (rows, columns) shaped matrix with the same elements
print(M.shape) #Outputs the shape (rows, columns) of matrix M into the console

#You can turn a matrix intro a vector by the "flatten" method:
MVec = M.flatten()    

#And turn it back into a matrix of the original shape:
vecM = MVec.reshape(3,2)

#Suggestion for exploration: Create another matrix M2 with different number of rows and columns and different element values then reshape it

#In terms of matrix oerations, pretty much all operations from linear algebra can be applied to these matrices:
#For instance instead of the reshape function, we could have used the transpose function to achieve the same outcome

vecM = vecM.transpose() #This yields the transpose of vecM by invoking the transpose method of this array
vecM = np.transpose(vecM) #Transposing it back by invoking the numpy transpose function. We do this to illustrate that there are many ways to achieve the same outcome. 
#This is generally true when coding - divergent implementations often yield convergent results. 

#Matrix addition works if (and only if) the matrices added have the same shape
Q = M + M #Adding two (3,2) matrices. Note that the elements of Q result from element-wise addition of the two matrices M that were added together
MT = M.transpose() #Create the matrix MT (Mtranspose) of shape (2,3)
Q2 = M + MT #This will error out because the matrix dimensions don't agree. 

#Dimensions have to match exactly:
MVec2 = np.copy(MVec) #Create a copy of the flattened Matrix M (now a vector), with the numpy "copy" function    
MVec2 = np.append(MVec2,7) #Add ("append") the value "7" to the end of our copy of MVec - so it is now one element longer
addedVectors = MVec + MVec2 #This will throw an error because the shapes don't match. If you were to add two vectors, one with a million elements and one with a million and one elements, this would not work. Shapes have to match exactly.

#Element-wise matrix multiplication also presumes exact matching of shapes
N = np.multiply(M,M) # This is element-wise multiplication of matrix M with itself, squaring all entries
#Note that this is element-wise multiplication of matrices, NOT the linear algebra operation of matrix multiplication.
#Matrix multiplication is extremely useful, but we will cover that next week, *after* introducing linear algebra in lecture

#Suggestion for exploration: Create new matrices of various shapes and see if you can add them or not. Do the same for element-wise multiplication and check the results 


#%% 6 Accessing matrix elements and representing missing data

M[0][1] #This accesses - and returns to the console - the value of the matrix entry in the first row and the second column (remember, Python indexes from 0, so everything is off by one)
M[0,1] #This does the same as line 314, but with one square bracket 
M[1,0] = 9 #Change the value in the 2nd row and 1st column of M to 9
M[1,:] #Reach into the 2nd row and return the contents of the entire 2nd row to the console
M[0][:] = 99 #Change the entries in the entire 1st row of M to 99
M[0:2,1] #Output the entries in the first two rows and 2nd column of M

#STRONG suggestion for exploration: Practice, practice and practice this indexing if you are not already familiar with it. Otherwise, you will make many logical errors by being off by one all the time.
#A fleeting glance at line 319 might suggest that this command will return the first column (and that's what it would do in Matlab or Julia), but that's not how indexing works in Python - so be careful. And/or practice.
 
#The trouble with holes:
spendingUserA = np.array([1,2,3]) #Create vector with elements 1 to 3 to represent spending of a user on 3 days 
spendingUserB = np.array([1,2]) #Create vector with elements 1 to 2 to represent spending of another user on 2 out of the 3 days
overallSpending = np.stack((spendingUserA,spendingUserB)) #This command intends to stack the vectors on top of each other to create a matrix that represents the overall spending on both days. However, this command will throw an error.
# Matrixes cannot have holes. Much like nature abhors a vacuum, matrices cannot abide holes because it wouldn't know how to align the stacks (left-aligned or right-aligned) even if there is a single hole. This issue is worse if there are multiple holes. 
# In other words, all rows of a matrix have to have the same number of columns and all columns have to have the save number of rows.
# This is problematic because in data analysis, we have to - more often than not - handle missing data. 
# Here, we might not know whether the 2nd user spent anything on one of the days - we only have data from 2 out of the 3 days (maybe the tracking device malfunctioned on one of the days)
# So we need to find a way to represent the "hole" (which will be useful to represent missing data) explicitly. 
# This is done with a special element, a "nan" - this stands for "not a number", but one that is represented like a number (not a string), so you can do matrix operations on it, much like on any other number (and unlike operations on characters)
# Nans are very valuable, as they are quite useful:
    
spendingUserB = np.array([1,2, np.nan]) #Create vector with elements 1 to 2 to represent spending of user B on the first 2 days. Say that the data from day 3 is missing, and we input a nan explicitly. 
overallSpending = np.stack((spendingUserA,spendingUserB)) #Now it works
overallSpending.shape #Confirming that the (2,3) matrix was created successfully, by staking the two (1,3) vectors, the 2nd one with the explicit hole

#Suggestion for exploration: Create other vectors with "holes" in different locations (first, then second element), then stack them all together to create a (4,3) matrix

#At this point, you might wonder where these nans come from. How are nans made? 
#Briefly, they are Pythons (and Matlabs, for that matter) answer to an age old philosophical question: What is the result of zero times infinity?
#Is it zero because any number multiplied by zero is zero, or infinite, because that's a lot of zeros (an infinite number of them)? Which way does it go?
#The answer is: 
0 * np.inf #How nans are made, forged in the very essence of mathematics. You can assign the output of this command to a variable to catch the nan, if you want.
                          

#%% 7 Some probability calculations with Python

#Here, we will simulate some experiments in probability with numpy (which we already imported)
#If it is unclear what is being simulated here: This code section will make a lot more sense if you run it *after* the lecture 
#This is particularly true if you don't already have a lot of experiene with probability and probability distributions

# That said, here is the situation:
# Imagine spinning an n-sided gear 100,000 times
# What does the resulting probability distribution look like?
# Is this distribution affected by the type of gear and how it is spun?
# Specifically, we consider the following 3 conditions:
# 1. the gear has 10 sides (numbered 1 to 10) and we read off its position once per spin 
# 2. the gear has 5 sides (1 to 5) and we read off its position twice per spin, adding up the two readings
# 3. the gear has 5 sides (1 to 5) and we read off its position 10x per spin, adding up all 10 readings


# 1. Spin a gear with sides numbered from 1 to 10, 100,000 times: 
#a) Initializing the parameters that describe the gear    
numSpins = 100000 # how many times are we spinning this gear?
numSides = 10 # how many sides does the gear have?

#b) Determining the outcomes of the spins by using numpys "random" function. Specifically, we're asking for random integers, as the outcomes of the spins are discrete whole numbers.
sata = np.random.randint(1,numSides+1,numSpins) # spinning the gear - # arguments: (low end of the outcome range, high end of the outcome range, how often are we doing this?)
#Note that this is a simulation, so it yields sata, *not* data. Data come from measurements. These are not measurements. "Simulated data" is a contradiction in terms, like the nan in line 346 of this code. See Nylen & Wallisch for a more in-depth discussion of sata.

#c) Plotting the distribution. Here, we will use pyplot from the matplotlib library. We will exlore matplotlib in detail soon. For now, just know that this collection of functions was reverse engineered to allow Matlab-like data visualizations in Python.
import matplotlib.pyplot as plt # import the function collection "pyplot" from the matplotlib library and call it "plt"

numBins = len(np.unique(sata)) # Determine how many bins we need in our histogram by counting how many unique outcomes are in the variable "sata" - given that we have a 10-sided gear, 10 is a safe bet, but in coding it pays to presume nothing 
plt.hist(sata,bins=numBins) # plot frequency distribution of outcomes in sata as a histogram with numBins bins. We expect a uniform distribution. Check this in the "Plots" tab of the Spyder environment, between the "Help" and "Files" tab.


# 2. Spin again - but read the outcomes off twice per spin - this time with sides from  1 to 5, then look at how the sums of these readings distribute:
numSides = 5 #New gear. This one has only 5 sides.
readOne = np.random.randint(1,numSides+1,numSpins) # first read
readTwo = np.random.randint(1,numSides+1,numSpins) # second read
temp = np.stack([readOne,readTwo],axis=1) # stack them into a single matrix, along the 2nd axis. So this yields a (100000,2) matrix. Setting axis to 0 would yield a (2,100000) matrix 
sata = np.sum(temp,axis=1) # add them up, element-wise, along axis 1, yielding 100000 sums. Summing along axis 0 would yield 2 sums.
numBins = len(np.unique(sata)) # how many bins for the histogram, taking nothing for granted?
plt.hist(sata,bins=numBins) # plot frequency distribution 


# 3. Using the same gear, but now read off ten times per trial - then look at how the resulting sums distribute:
firstRead = np.random.randint(1,numSides+1,numSpins) # first spin
secondRead = np.random.randint(1,numSides+1,numSpins) # second spin
thirdRead = np.random.randint(1,numSides+1,numSpins) # third spin
fourthRead = np.random.randint(1,numSides+1,numSpins) # fourth spin
fifthRead = np.random.randint(1,numSides+1,numSpins) # fifth spin
sixthRead = np.random.randint(1,numSides+1,numSpins) # sixth spin
seventhRead = np.random.randint(1,numSides+1,numSpins) # seventh spin
eighthRead = np.random.randint(1,numSides+1,numSpins) # eighth spin
ninthRead = np.random.randint(1,numSides+1,numSpins) # ninth spin
tenthRead = np.random.randint(1,numSides+1,numSpins) # tenth spin
temp = np.stack([firstRead, secondRead, # stack all 10 into a single array
                 thirdRead, fourthRead,
                 fifthRead, sixthRead,
                 seventhRead, eighthRead,
                 ninthRead, tenthRead],axis=1) 
sata = np.sum(temp,axis=1)/5 # add them up - element-wise - then divide by 5 to normalize the range
numBins = len(np.unique(sata)) # how many bins do we need now?
plt.hist(sata,bins=numBins) # plot frequency distribution (normal)

#How does the distribution change as more and more reads are taken per spin?

#Suggestion for exploration: Try modifying the gears in lines 368 and 382 to see how this affects the resulting distribution.
#Do the same with the number of spins in line 367 - how does this affect the resulting distribution?
#Finally, add or remove reads from the spins in the example from line 391 onwards (make sure to change the normalization factor accordingly) - how does this change things?

#This is a sufficient amount of code and concepts for a first forray into Python, probability and simulation. We will continue here next time. 
#As a heads up, we will introduce a more efficient control of program flow next time. Note how awkward lines 392-401 are, spelling everything out manually. 
#If only there was a way to do this in one line. That would be so much more efficient. Spoiler: There is. We'll introduce that next time. So you never have to type all of that.
