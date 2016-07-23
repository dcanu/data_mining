import csv
import math
import sys
import operator
from time import time

#Classify the test data point
def KnnNeighbors(trainingData, testingRow, distMetric, k):
	metricDist = []
	length = len(testingRow)-1
	trainLength = len(trainingData)
	
	#Calculate the distance between the testing point and the training dataset points
	#according to the appropriate distance metric
	for x in range(trainLength):
		dist = 0
		if distMetric == 0 :
			dist = euclideanDistance(testingRow, trainingData[x], length)
		elif distMetric == 1 :
			dist = manhattanDistance(testingRow, trainingData[x], length)
		elif distMetric == 2 :
			dist = cosineDistance(testingRow, trainingData[x], length)
		metricDist.append((dist, trainingData[x]))
	#ascending sort the array to get the K nearest points i.e the smallest distance first
	metricDist = sorted(metricDist)
	#classify it according to a majority vote for the K nearest points
	majorityClass = {}
	for x in range(k):
		resp = metricDist[x][1][-1]
		if resp in majorityClass:
			majorityClass[resp] += 1
		else:
			majorityClass[resp] = 1
	# Descending sort the array to get the largest voted value
	sortedClass = sorted(majorityClass.items(), key=operator.itemgetter(1), reverse=True)
	return sortedClass[0][0]

	
	
#Read from the test and training datasets
#Convert the values from string to float
def loadFiles(trainFileName,testFileName, trainingData , testingData):
	#open the file
	csvfileTrain = open(trainFileName, 'r')
	csvfileTest = open(testFileName, 'r')
	#create file reader
	myFileDataTrain = csv.reader(csvfileTrain)
	myFileDataTest = csv.reader(csvfileTest)
	trncol=len(next(myFileDataTrain))
	tsncol=len(next(myFileDataTest))
	traindataset = list(myFileDataTrain)
	testdataset = list(myFileDataTest)
	#convert to float
	for x in range(len(traindataset)):
		for y in range(trncol):
			traindataset[x][y] = float(traindataset[x][y])
		trainingData.append(traindataset[x])
	
	for x in range(len(testdataset)):
		for y in range(tsncol):
			testdataset[x][y] = float(testdataset[x][y])	
		testingData.append(testdataset[x])
	
	return 0	
	
# ------------Euclidean Distance-------------
# calculate sum of (a-b)^2 for all values of a and b
# take square root of result
def euclideanDistance(setA, setB, len):
	dist = 0.0
	for x in range(len):
		dist += pow((setA[x] - setB[x]), 2)
	return math.sqrt(dist)

# ------------Manhattan Distance-------------
# calculate sum of |a-b| for all values of a and b
def manhattanDistance(setA, setB, len):
	dist = 0.0
	for x in range(len):
		dist += abs((setA[x] - setB[x]))
	return dist

# ------------Cosine Distance-------------
# calculate a*b, a^2, b^2 for all values of a and b
# calculate 1 - (sum(a*b)/sum(a^2)^0.5 x sum(b^2)^0.5)	
def	cosineDistance(setA, setB, len):
	numer = 0.0
	denomA = 0.0
	denomB = 0.0
	
	for x in range(len):
		numer += setA[x]*setB[x]
		denomA += pow(setA[x],2)
		denomB += pow(setB[x],2)

	return 1 - (numer / (math.sqrt(denomA)*math.sqrt(denomB)))
	
# ----------------------------------Main-------------------------------------------
trainingData=[]
testingData=[]
d = [0,1,2]
n = [1,3,5,9]
dmetric = ["euclidean", "manhattan", "cosine"]
#Read Arguments
myArgs = len(sys.argv)
if myArgs != 2:
	print("Usage -> knn.py dataset-name")
	exit()
#Load Data from files
else:
	if sys.argv[1] == "forestfires":
		loadFiles('forestfiresmod.csv','forestfiresmod.csv', trainingData, testingData)
	elif sys.argv[1] == "iris":
		loadFiles('iris-train.csv','iris-test.csv', trainingData, testingData)
#iterate once for each distance metric (total 3 times)
for dm in d:
	print("\n------Distance Metric : " +  dmetric[dm]+"------")
	print("| K | Correct | Incorrect |    Time    |") 
	print("----------------------------------------")
	#iterate for k value of 1 ,3 ,5 ,9 
	for k in n:
		#log execution time
		t0 = time()
		numCorr = 0
		numWron = 0
		#call the Knn function for each row in test set
		for x in range(len(testingData)):
			result = KnnNeighbors(trainingData, testingData[x],dm, k)	
			#count the correct and incorrect values
			if result == testingData[x][-1]:
				numCorr += 1
			else:
				numWron +=1
		t1 = time()
		print("| " + repr(k) + " |    " + repr(numCorr) + "    |     " + repr(numWron) + "     | " + repr(round(t1-t0,8)) + " |") 
#----------------------------------END-----------------------------------------------

