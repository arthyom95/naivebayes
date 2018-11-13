#opening training, vocabulary and test files
TRAINING_FILE = open("sampleTrain.txt","r")
VOCAB_FILE = open("sampleTrain.vocab.txt","r")
TEST_FILE = open("sampleTest.txt","r")

#read the training file and save the data in a dictionary
#the dictionary contains two entries, one for each class, with the class being the key
#each "sentence" belonging to a class is added to a list to the value of the dictionary entry with the corresponding class
def readTrainingFile(trainingFile):
	fileContents = {}
	
	for line in trainingFile:		
		line.strip()
		values = line.split("\t")
		if not values[1] in fileContents:
			#rstrip() removes numerous unnecessary special characters, such as the end of line character
			fileContents[values[1]] = [values[2].rstrip()]
		else:
			fileContents[values[1]].append(values[2].rstrip())					
			
	return fileContents

#read the test file
#because the document's id is used when displaying the results, the document's id is used as the key in the first dictionary
#with the values of the dictionary being another dictionary, which has the class as a key and the "sentence" as the value
def readTestFile(testFile):
	fileContents = {}
	
	for line in testFile:		
		lineDict = {}
		line.strip()
		values = line.split("\t")
		lineDict[values[1]] = values[2].rstrip()
		fileContents[values[0]] = lineDict	
			
	return fileContents

#saved as a list
def readVocabularyFile(vocabularyFile):
	vocabularyList = []
	for line in vocabularyFile:					
		vocabularyList.append(line.rstrip())
	return vocabularyList

#while breaking the camel-case typing convention for variables, using a capital c for class seemed
#like the most sensible solution, given that "class" is a reserved word in python and cannot be used as function parameter
#the function gets the prior probability for a single class
def getPriorProbabilityForClass(trainingData, Class):	
	noOfDocuments = 0
	noOfItemsInClass = 0
	
	for key, value in trainingData.items():			
		noOfDocuments = noOfDocuments + len(value)
		if key == Class:
			noOfItemsInClass = len(value)
		
	if(noOfItemsInClass > 0):
		return noOfItemsInClass / noOfDocuments
	else:
		return 0

#the above function applied for all the classes
#returns a dictionary with the key being the class and the value being its prior probability
def getPriorProbabilities(trainingData):
	priorProbabilities = {}
	
	#range is not inclusive
	for Class in range(0,2):
		priorProbabilities[Class] = getPriorProbabilityForClass(trainingData, str(Class))
		
	return priorProbabilities

#counts how many occurrences of a feature exist in all the documents for a specified class
def getFeatureCountForClass(trainingData, Class, feature):
	featureCount = 0
	for key, value in trainingData.items():
		if key == Class:
			for sentence in value:				
				if feature in sentence:
					featureCount = featureCount + sentence.lower().split().count(feature)
	return featureCount
	
#gets the counts for all features in the vocabulary for a given class
def getVocabCountForClass(trainingData, Class, vocabList):
	vocabCount = {}
	for word in vocabList:
		vocabCount[word] = getFeatureCountForClass(trainingData, Class, word)
	return vocabCount

#gets the counts for all features in the vocabulary for all classes
def getVocabCountsForClasses(trainingData, vocabList):
	vocabCounts = {}
	
	#range is not inclusive
	for Class in range(0,2):
		vocabCounts[Class] = getVocabCountForClass(trainingData, str(Class), vocabList)
		
	return vocabCounts

#counts how many words there are in total for each class
def getWordsPerClass(trainingData):
	wordsPerClass = {}	
	
	for Class in range (0,2):	
		wordCount = 0
		#we only need to count all the words in the training data for a specific class
		#given that there are no words outside of the vocabulary present in the training data
		classItems = trainingData.get(str(Class))
		
		for sentence in classItems:
			words = sentence.split(" ")
			wordCount = wordCount + len(words)
			
		wordsPerClass[Class] = wordCount
		
	return wordsPerClass

#calculate the feature likelihood
#results are returned as a dictionary with the class as the key and with the value
#being another dictionary containing each word as a key, with each corresponding value being the likelihood
def getFeatureLikelihood(trainingData, vocabList):
	featureLikelihood = {}
	vocabCounts = getVocabCountsForClasses(trainingData, vocabList)
	wordsPerClass = getWordsPerClass(trainingData)
	
	for Class in range(0,2):
		featureCountsForClass = vocabCounts.get(Class)
		wordsPerClassForClass = wordsPerClass.get(Class)		
		featureLikelihoodForClass = {}
		for word in vocabList:			
			#applying laplace smoothing			
			featureLikelihoodForClass[word] = (featureCountsForClass.get(word) + 1) / (wordsPerClassForClass + len(vocabList))
						
		featureLikelihood[Class] = featureLikelihoodForClass
			
	return featureLikelihood

#determine the class a sentence should belong to
def getClassForSentence(sentence, featureLikelihood, priorProbabilities):
	probabilityFirstClass = 1.0 * priorProbabilities.get(0)
	probabilitySecondClass = 1.0 * priorProbabilities.get(1)
	
	firstClassProbabilities = featureLikelihood.get(0)
	secondClassProbabilities = featureLikelihood.get(1)		
	
	sentence = sentence.rstrip()
	wordList = sentence.split(" ")
	
	for word in wordList:
		probabilityFirstClass = probabilityFirstClass * float(firstClassProbabilities.get(word))
		probabilitySecondClass = probabilitySecondClass * float(secondClassProbabilities.get(word))
	
	if probabilityFirstClass > probabilitySecondClass:
		return 0
	else:
		return 1

#apply the above function for all the test data
def getClassesForTestData(testData, featureLikelihood, priorProbabilities):
	testDataClasses = {}

	for key, value in testData.items():
		for itemsKey, itemsValue in value.items():
			testDataClasses[key] = getClassForSentence(itemsValue, featureLikelihood, priorProbabilities)
				
	return testDataClasses	

#calculate the accuracy of the test data by using the actual result and the provided true class
def getTestDataAccuracy(testData, predictedClasses):
	totalNoOfTests = 0
	noOfCorrectPredictions = 0
	
	for key, value in testData.items():
		totalNoOfTests = totalNoOfTests + 1
		for itemsKey, itemsValue in value.items():
			predictedClass = predictedClasses.get(key)
						
			if str(predictedClass) == itemsKey:
				noOfCorrectPredictions = noOfCorrectPredictions + 1	
	
	return (noOfCorrectPredictions/totalNoOfTests)*100

#function for printing the feature likelihood
def prettyPrintFeatureLikelihood(featureLikelihood):
	firstClassFeatureLikelihood = featureLikelihood.get(0)
	secondClassFeatureLikelihood = featureLikelihood.get(1)	
	
	print("class 0")
	for key,value in firstClassFeatureLikelihood.items():
		print(str(key) + "  " + str(value))
		
	print("")
	
	print("class 1")
	for key,value in secondClassFeatureLikelihood.items():
		print(str(key) + "  " + str(value))	
	
#function for printing the results
def prettyPrintResults(priorProbabilities, featureLikelihood, predictedClasses, testDataAccuracy):
	print("")
	print("Prior Probabilities: ")
	print("class 0 = " + str(priorProbabilities.get(0)))
	print("class 1 = " + str(priorProbabilities.get(1)))
	
	print("")
	print("")
	print("Feature likelihoods")
	prettyPrintFeatureLikelihood(featureLikelihood)
	
	print("")
	print("")
	print("Predictions on test data")
	print("d5 = " + str(predictedClasses.get("d5")))
	print("d6 = " + str(predictedClasses.get("d6")))
	print("d7 = " + str(predictedClasses.get("d7")))
	print("d8 = " + str(predictedClasses.get("d8")))
	print("d9 = " + str(predictedClasses.get("d9")))
	print("d10 = " + str(predictedClasses.get("d10")))
	
	print("")
	print("")
	print("Accuracy on test data = " + str(testDataAccuracy))
	
					
trainingData = readTrainingFile(TRAINING_FILE)
testData = readTestFile(TEST_FILE)
vocabularyList = readVocabularyFile(VOCAB_FILE)

priorProbabilities = getPriorProbabilities(trainingData)
featureLikelihood = getFeatureLikelihood(trainingData,vocabularyList)
predictedClasses = getClassesForTestData(testData, featureLikelihood, priorProbabilities)
testDataAccuracy = getTestDataAccuracy(testData, predictedClasses)

prettyPrintResults(priorProbabilities, featureLikelihood, predictedClasses, testDataAccuracy)