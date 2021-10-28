/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <limits>
#include <math.h>


using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Obtain an integer random number in the range [Low,High]
int randomInt(int Low, int High)
{
	return rand()%High + Low;
}

// ------------------------------
// Obtain a real random number in the range [Low,High]
double randomDouble(double Low, double High)
{
	return Low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(High-Low)));
}

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	eta = 0.7;
	mu = 1;
	validationRatio = 0.0;
	decrementFactor = 1.0;
	online = false;
	outputFunction = 0;
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, std::vector<int> npl) {
	nOfLayers = nl;
	layers.resize(nOfLayers);

	for (size_t j = 0; j < nOfLayers; j++)
	{
		layers[j].nOfNeurons = npl[j];
		layers[j].neurons.resize(npl[j]);

		//bias initialization
		if(j > 0)
		for (size_t i = 0; i < layers[j].nOfNeurons; i++){
			layers[j].neurons[i].bias = 1;
			layers[j].neurons[i].lastBias = 0;
		}
	}

	return 1;
}


// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron() {
	freeMemory();
}


// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory() {
	layers.clear();
}
//

//Return decremented eta by layer
double MultilayerPerceptron::getDecrementedEta(int layer){
	double eta_decremented = pow(decrementFactor,-(nOfLayers-layer)) * eta;
	return eta_decremented;
}

//checkClass
void MultilayerPerceptron::checkClass(int &class_desired,int &class_obtained,std::vector<double> prediction,std::vector<double> pattern_outputs){
	
	double max = prediction[0];
	class_obtained = 0;
		for (size_t j = 1; j < prediction.size(); j++)
		if( prediction[j] > max){
			max = prediction[j];
			class_obtained = j;
		}
	//max class index pattern
	max = pattern_outputs[0];
	class_desired = 0;
	for (size_t j = 1; j < pattern_outputs.size(); j++)
		if( pattern_outputs[j] > max){
			max = pattern_outputs[j];
			class_desired = pattern_outputs[j];
		}
}
// ------------------------------
// Fill all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() {
	for (size_t j = 1; j < nOfLayers; j++) //we skip input layer 
	{
		for (size_t i = 0; i < layers[j].nOfNeurons; i++)
		{
			layers[j].neurons[i].w.resize(layers[j - 1].nOfNeurons); //reserve memory for w vector
			layers[j].neurons[i].deltaW.resize(layers[j - 1].nOfNeurons); //reserve memory for deltaW vector
			layers[j].neurons[i].lastDeltaW.resize(layers[j - 1].nOfNeurons); //reserve memory for lastdeltaW vector

			for (size_t k = 0; k < layers[j].neurons[i].w.size(); k++)
			{
				float random_weight = ((float)rand() / (float)(RAND_MAX)) * 2 - 1;
				layers[j].neurons[i].w[k] = random_weight; //random weight
			}
		}
	}
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(std::vector<double> input) {
	
	for (size_t i = 0; i < layers[0].nOfNeurons; i++)
		layers[0].neurons[i].out = input[i];
}

// ------------------------------
// Get the outputs predicted by the network (out vector of the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(std::vector<double> &output)
{
	for (size_t i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
		output[i] = (layers[nOfLayers - 1].neurons[i].out);
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() {

	for (size_t j = 0; j < nOfLayers; j++)
		for (size_t i = 0; i < layers[j].nOfNeurons; i++)
			layers[j].neurons[i].wCopy = layers[j].neurons[i].w;
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() {

	for (size_t j = 0; j < nOfLayers; j++)
		for (size_t i = 0; i < layers[j].nOfNeurons; i++)
			layers[j].neurons[i].w = layers[j].neurons[i].wCopy;
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() {

	for (size_t j = 1; j < nOfLayers; j++) //skip input layer
	for (size_t i = 0; i < layers[j].nOfNeurons; i++)
	{
		double net = 0.0;
		//HIDDEN LAYERS AND OUTPUT LAYER
		for (size_t k = 0; k < layers[j].neurons[i].w.size(); k++)
			net += layers[j].neurons[i].w[k] * layers[j - 1].neurons[k].out;
		
		net += layers[j].neurons[i].bias;	//bias
		//h(net)
		if( outputFunction == 1 && j == nOfLayers -1)		//prepare net for softmax
			layers[j].neurons[i].out = net;
		else										//sigmoid
			layers[j].neurons[i].out = 1 / (1 + exp(-net)); 
			
	}
	//apply softmax if necessary
	if(outputFunction == 1){
		double sum = 0.0;
		for (size_t i = 0; i < layers[nOfLayers-1].nOfNeurons; i++)
			sum += exp( layers[nOfLayers-1].neurons[i].out );

		for (size_t i = 0; i < layers[nOfLayers-1].nOfNeurons; i++)
			layers[nOfLayers-1].neurons[i].out = (exp(layers[nOfLayers-1].neurons[i].out)) / sum;
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::obtainError(std::vector<double> target, int errorFunction) {

	if(errorFunction == 0){	//MSE

		double error = 0.0;
		for (size_t i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
		{
			error += pow((target[i] - layers[nOfLayers - 1].neurons[i].out), 2);
		}
		error /= layers[nOfLayers - 1].nOfNeurons;
		return error;
	}
	else{	//Cross Entropy

		double error = 0.0;
		for (size_t i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
		{
			int d,class_desired,class_obtained;
			std::vector<double> prediction;
			prediction.resize(target.size());
			getOutputs(prediction);
			
			checkClass(class_desired,class_obtained,prediction,target);

			if( class_desired == i )
				d = 1;
			else
				d = 0;

			error += log(layers[nOfLayers - 1].neurons[i].out) * d;
		}
		error /= layers[nOfLayers - 1].nOfNeurons;
		return error;
	}
	
}

// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::backpropagateError(std::vector<double> target, int errorFunction) {

	int j = nOfLayers - 1; //start at output layer
	while (j > 0) //we skip input layer
	{

		for (size_t i = 0; i < layers[j].nOfNeurons; i++)
		{
			//output layers
			if (j == nOfLayers - 1)
			{
				//sigmoid and MSE
				if (outputFunction == 0 && errorFunction == 0)
					layers[j].neurons[i].delta = -(target[i] - layers[j].neurons[i].out ) * layers[j].neurons[i].out * (1 - layers[j].neurons[i].out);
				//sigmoid and Entropy	
				else if (outputFunction == 0 && errorFunction == 1)
					layers[j].neurons[i].delta = -(target[i] / layers[j].neurons[i].out ) * layers[j].neurons[i].out * (1 - layers[j].neurons[i].out);
				//softmax and MSE
				else if(outputFunction == 1 && errorFunction == 0){
					layers[j].neurons[i].delta = 0;
					for (size_t k = 0; k < layers[j].nOfNeurons; k++)
					{
						if (k == i)
							layers[j].neurons[i].delta += (target[k] - layers[j].neurons[k].out ) * layers[j].neurons[i].out * (1-layers[j].neurons[k].out);
						else
							layers[j].neurons[i].delta += (target[k] - layers[j].neurons[k].out ) * layers[j].neurons[i].out * (-layers[j].neurons[k].out);
					}
					layers[j].neurons[i].delta = - layers[j].neurons[i].delta;
				}
				//softmax and Entropy
				else if(outputFunction == 1 && errorFunction == 1){
					layers[j].neurons[i].delta = 0;
					for (size_t k = 0; k < layers[j].nOfNeurons; k++)
					{
						if (k == i)
							layers[j].neurons[i].delta += (target[k] / layers[j].neurons[k].out ) * layers[j].neurons[i].out * (1-layers[j].neurons[k].out);
						else
							layers[j].neurons[i].delta += (target[k] / layers[j].neurons[k].out ) * layers[j].neurons[i].out * (-layers[j].neurons[k].out);
					}
					layers[j].neurons[i].delta = - layers[j].neurons[i].delta;
				}
			}
			//hidden layers 
			else
			{
				double d = 0.0;
				for (size_t k = 0; k < layers[j + 1].nOfNeurons; k++)
					d += layers[j + 1].neurons[k].delta * layers[j].neurons[i].w[k];

				d = d * layers[j].neurons[i].out * (1 - layers[j].neurons[i].out);

				layers[j].neurons[i].delta = d;
			}
		}

		j--;
	}
}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {

	for (size_t j = 1; j < nOfLayers;  j++)
	{
		for (size_t i = 0; i < layers[j].nOfNeurons; i++)
		{
			for (size_t k = 0; k < layers[j-1].nOfNeurons; k++)
			{
				layers[j].neurons[i].deltaW[k] = layers[j].neurons[i].delta * layers[j-1].neurons[k].out;
			}

			layers[j].neurons[i].deltaBias =  layers[j].neurons[i].delta * layers[j].neurons[i].bias; //bias
			
		}
		
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() {

	for (size_t j = 1; j < nOfLayers;  j++)
	{
		for (size_t i = 0; i < layers[j].nOfNeurons; i++)
		{
			for (size_t k = 0; k < layers[j-1].nOfNeurons; k++)//update weights
			{
				layers[j].neurons[i].w[k] = layers[j].neurons[i].w[k]-
					getDecrementedEta(j) * layers[j].neurons[i].deltaW[k] - mu*(getDecrementedEta(j)* layers[j].neurons[i].lastDeltaW[k]);

			}
			layers[j].neurons[i].lastDeltaW = layers[j].neurons[i].deltaW; //update last delta
			//update bias
			layers[j].neurons[i].bias =  layers[j].neurons[i].bias - 
			 	getDecrementedEta(j) * layers[j].neurons[i].deltaBias - mu*(getDecrementedEta(j)* layers[j].neurons[i].lastBias);
			
			layers[j].neurons[i].lastBias = layers[j].neurons[i].deltaBias; //update last bias
			
		}
		
	}
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() {

	for (size_t j = 1; j < nOfLayers; j++){
		std::cout << "LAYER[" << j << "]"<<std::endl;

		for (size_t i = 0; i < layers[j].nOfNeurons; i++){
			std::cout << "	NEURON[" << i << "]"<<std::endl;
			std::cout << "	BIAS : " << layers[j].neurons[i].bias << std::endl;

			for (size_t k = 0; k < layers[j].neurons[i].w.size(); k++)
				std::cout << "	W[" << i << "][" << k << "]: " << layers[j].neurons[i].w[k];
			
			std::cout << std::endl;
		}
	}
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
// The step of adjusting the weights must be performed only in the online case
// If the algorithm is offline, the weightAdjustment must be performed in the "train" function
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::performEpoch(std::vector<double> input, std::vector<double> target, int errorFunction) {

	feedInputs(input);
	
	forwardPropagate();
	
	backpropagateError(target,errorFunction);

	accumulateChange();

	if( online )
	weightAdjustment();
}

// ------------------------------
// Read a dataset from a file name and return it
Dataset* MultilayerPerceptron::readData(const char *fileName) {

	Dataset *dataset = new Dataset;

	std::ifstream f;
	std::string value = "";

	f.open(fileName, std::ifstream::in);

	if (f.is_open())
	{

		f >> dataset->nOfInputs >> dataset->nOfOutputs >> dataset->nOfPatterns;

		dataset->inputs.resize(dataset->nOfPatterns);
		for (int i = 0; i < dataset->nOfPatterns; ++i)
			dataset->inputs[i].resize(dataset->nOfInputs);

		dataset->outputs.resize(dataset->nOfPatterns);
		for (int i = 0; i < dataset->nOfPatterns; ++i)
			dataset->outputs[i].resize(dataset->nOfOutputs);

		for (size_t i = 0; i < dataset->nOfPatterns; i++)
		{
			for (size_t j = 0; j < dataset->nOfInputs; j++)
			{
				f >> value;
				dataset->inputs[i][j] = stod(value);
			}
			for (size_t k = 0; k < dataset->nOfOutputs; k++)
			{
				f >> value;
				dataset->outputs[i][k] = stod(value);
			}
		}
	}

	return dataset;
}


// ------------------------------
// Train the network for a dataset (one iteration of the external loop)
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::train(Dataset* trainDataset, int errorFunction) {

	int i;
	for (i = 0; i < trainDataset->nOfPatterns; i++)
	{
		performEpoch(trainDataset->inputs[i], trainDataset->outputs[i],errorFunction);
		if (!online) 
		weightAdjustment();
	}
	
}

// ------------------------------
// Test the network with a dataset and return the error
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::test(Dataset* dataset, int errorFunction) {

	int i;
	double error = 0.0;
	for (i = 0; i < dataset->nOfPatterns; i++)
	{
		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		error += obtainError(dataset->outputs[i],errorFunction);
	}

	if(errorFunction == 0)	//MSE
		error = error / dataset->nOfPatterns;
	else	//Entropy
		error = -error / dataset->nOfPatterns;

	return error;
}


// ------------------------------
// Test the network with a dataset and return the CCR
double MultilayerPerceptron::testClassification(Dataset* dataset) {

	double sum = 0;

	for(int i=0; i<dataset->nOfPatterns; i++){
		std::vector<double> prediction;
		prediction.resize( dataset->nOfOutputs );

		// Feed the inputs and propagate the values
		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);

		//max predicted
		int class_desired,class_obtained;
		checkClass(class_desired,class_obtained,prediction,dataset->outputs[i]);

		if (class_desired == class_obtained)
			sum++;
				
		prediction.clear();

	}

	return 100 * sum/dataset->nOfPatterns;
}


// ------------------------------
// Optional Kaggle: Obtain the predicted outputs for a dataset
void MultilayerPerceptron::predict(Dataset* dataset)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers-1].nOfNeurons;
	std::vector<double> salidas;
	salidas.resize(numSalidas);
	
	std::cout << "Id,Category" << endl;
	
	for (i=0; i<dataset->nOfPatterns; i++){

		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		getOutputs(salidas);

		int maxIndex = 0;
		for (j = 0; j < numSalidas; j++)
			if (salidas[j] >= salidas[maxIndex])
				maxIndex = j;
		
		std::cout << i << "," << maxIndex << endl;

	}
}


// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
// Both training and test CCRs should be obtained and stored in ccrTrain and ccrTest
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::runBackPropagation(Dataset * trainDataset, Dataset * testDataset, int maxiter, double *errorTrain, double *errorTest, double *ccrTrain, double *ccrTest, int errorFunction)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving = 0;
	nOfTrainingPatterns = trainDataset->nOfPatterns;

	Dataset * validationDataset = NULL;
	double validationError = 0, previousValidationError = 0;
	int iterWithoutImprovingValidation = 0;

	// Generate validation data
	if (validationRatio > 0 && validationRatio < 1)
	{
		validationDataset = new Dataset;
		validationDataset->nOfPatterns = trainDataset->nOfPatterns * validationRatio;
		validationDataset->nOfInputs = trainDataset->nOfInputs;
		validationDataset->nOfOutputs = trainDataset->nOfOutputs;

		std::vector<int> patterns_to_select = integerRandomVectoWithoutRepeating(0,trainDataset->nOfPatterns-1,
																			validationDataset->nOfPatterns);
		//add patterns to validation dataset
		for (size_t i = 0; i < validationDataset->nOfPatterns; i++)
		{
			validationDataset->inputs.push_back(trainDataset->inputs[patterns_to_select[i]]);
			validationDataset->outputs.push_back(trainDataset->outputs[patterns_to_select[i]]);
		}
		//erase patterns from train dataset
		for (size_t i = 0; i < patterns_to_select.size(); i++)
		{
			trainDataset->inputs.erase( trainDataset->inputs.begin()+ patterns_to_select[i] );
			trainDataset->outputs.erase( trainDataset->outputs.begin()+ patterns_to_select[i] );
			trainDataset->nOfPatterns--;
		}
	}

	// Learning
	do {
		
		train(trainDataset,errorFunction);
		
		double trainError = test(trainDataset,errorFunction);
		if(countTrain==0 || trainError < minTrainError){
			minTrainError = trainError;
			copyWeights();
			iterWithoutImproving = 0;
		}
		else if( (trainError-minTrainError) < 0.00001)
			iterWithoutImproving = 0;
		else
			iterWithoutImproving++;

		if(iterWithoutImproving==50){
			std::cout << "We exit because the training is not improving!!"<< endl;
			restoreWeights();
			countTrain = maxiter;
		}

		countTrain++;

		if(validationDataset!=NULL){
			if(previousValidationError==0)
				previousValidationError = 999999999.9999999999;
			else
				previousValidationError = validationError;
			validationError = test(validationDataset,errorFunction);
			if(validationError < previousValidationError)
				iterWithoutImprovingValidation = 0;
			else if((validationError-previousValidationError) < 0.00001)
				iterWithoutImprovingValidation = 0;
			else
				iterWithoutImprovingValidation++;
			if(iterWithoutImprovingValidation==50){
				std::cout << "We exit because validation is not improving!!"<< endl;
				restoreWeights();
				countTrain = maxiter;
			}
		}

		std::cout << "Iteration " << countTrain << "\t Training error: " << trainError << "\t Validation error: " << validationError << endl;

	} while ( countTrain<maxiter );

	if ( (iterWithoutImprovingValidation!=50) && (iterWithoutImproving!=50))
		restoreWeights();

	std::cout << "NETWORK WEIGHTS" << endl;
	std::cout << "===============" << endl;
	printNetwork();

	std::cout << "Desired output Vs Obtained output (test)" << endl;
	std::cout << "=========================================" << endl;
	for(int i=0; i<testDataset->nOfPatterns; i++){
		std::vector<double> prediction;
		prediction.resize( testDataset->nOfOutputs );

		// Feed the inputs and propagate the values
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for(int j=0; j<testDataset->nOfOutputs; j++)
			std::cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
		std::cout << endl;
		prediction.clear();

	}

	*errorTest=test(testDataset,errorFunction);;
	*errorTrain=minTrainError;
	*ccrTest = testClassification(testDataset);
	*ccrTrain = testClassification(trainDataset);

}

// -------------------------
// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char * fileName)
{
	// Object for writing the file
	ofstream f(fileName);

	if(!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for(int i = 0; i < nOfLayers; i++)
	{
		f << " " << layers[i].nOfNeurons;
	}
	f << " " << outputFunction;
	f << endl;

	// Write the weight matrix of every layer
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(layers[i].neurons[j].w.size() > 0)
				    f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;

}


// -----------------------
// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char * fileName)
{
	// Object for reading a file
	ifstream f(fileName);

	if(!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	std::vector<int> npl;

	// Read number of layers
	f >> nl;
	
	npl.resize(nl);

	// Read number of neurons in every layer
	for(int i = 0; i < nl; i++)
	{
		f >> npl[i];
	}
	f >> outputFunction;

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(!(outputFunction==1 && (i==(nOfLayers-1)) && (k==(layers[i].nOfNeurons-1))))
					f >> layers[i].neurons[j].w[k];

	f.close();
	npl.clear();

	return true;
}
