# AIAS_BackProp

Please follow the steps to run this script
1. Please run BackPropAlgorithm.m , This file will automatically read the dataset and performs the following
   a. Normalize the data according to the MIN, MAX values
   b. Train the Neural Network
   c. Save the weights into Weights.mat file
   d. Plot the Error plot during Training 

2. Please run TestingBackProp.m file to test this algorithm, this does the following,
	a. Load the saved Weights from Weights.mat file
	b. Run the Neural Network and give out probabilities
	c. Plots the bar graph to show the outputs of prediction