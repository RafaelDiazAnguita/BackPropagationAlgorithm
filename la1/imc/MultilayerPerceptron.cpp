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
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <limits>
#include <math.h>

using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	eta = 0.1;
	mu = 0.9;
	validationRatio = 0.0;
	decrementFactor = 1.0;
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, std::vector<int> npl)
{
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
MultilayerPerceptron::~MultilayerPerceptron()
{
	freeMemory();
}

// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory()
{
	layers.clear();
}
//
//Return decremented eta by layer
double MultilayerPerceptron::getDecrementedEta(int layer){
	double eta_decremented = pow(decrementFactor,-(nOfLayers-layer)) * eta;
	return eta_decremented;
}

// ------------------------------
// Feel all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights()
{
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
void MultilayerPerceptron::feedInputs(std::vector<double> input)
{
	for (size_t i = 0; i < layers[0].nOfNeurons; i++)
	{
		layers[0].neurons[i].out = input[i];
	}
}

// ------------------------------
// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(std::vector<double> &output)
{
	for (size_t i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
		output[i] = (layers[nOfLayers - 1].neurons[i].out);
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights()
{
	for (size_t j = 0; j < nOfLayers; j++)
		for (size_t i = 0; i < layers[j].nOfNeurons; i++)
			layers[j].neurons[i].wCopy = layers[j].neurons[i].w;
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights()
{
	for (size_t j = 0; j < nOfLayers; j++)
		for (size_t i = 0; i < layers[j].nOfNeurons; i++)
			layers[j].neurons[i].w = layers[j].neurons[i].wCopy;
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate()
{
	for (size_t j = 1; j < nOfLayers; j++) //skip input layer
	{
		for (size_t i = 0; i < layers[j].nOfNeurons; i++)
		{
			double net = 0.0;
			//HIDDEN LAYERS AND OUTPUT LAYER
			for (size_t k = 0; k < layers[j].neurons[i].w.size(); k++)
			{
				net += layers[j].neurons[i].w[k] * layers[j - 1].neurons[k].out;
			}
			net += layers[j].neurons[i].bias;	//bias

			layers[j].neurons[i].out = 1 / (1 + exp(-net)); //h(net)
		}
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MultilayerPerceptron::obtainError(std::vector<double> target)
{
	double error = 0.0;
	for (size_t i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
	{
		error += pow((target[i] - layers[nOfLayers - 1].neurons[i].out), 2);
	}
	error /= layers[nOfLayers - 1].nOfNeurons;
	return error;
}

// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MultilayerPerceptron::backpropagateError(std::vector<double> target)
{

	int j = nOfLayers - 1; //start at output layer
	while (j > 0) //we skip input layer
	{

		for (size_t i = 0; i < layers[j].nOfNeurons; i++)
		{
			//output layers
			if (j == nOfLayers - 1)
			{
				layers[j].neurons[i].delta = -2*(target[i] - layers[j].neurons[i].out ) * layers[j].neurons[i].out * (1 - layers[j].neurons[i].out);
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
void MultilayerPerceptron::accumulateChange()
{			
	
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
void MultilayerPerceptron::weightAdjustment()
{
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
void MultilayerPerceptron::printNetwork()
{
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
void MultilayerPerceptron::performEpochOnline(std::vector<double> input, std::vector<double> target)
{

	feedInputs(input);
	
	forwardPropagate();

	backpropagateError(target);

	accumulateChange();

	weightAdjustment();

}

// ------------------------------
// Read a dataset from a file name and return it
Dataset *MultilayerPerceptron::readData(const char *fileName)
{
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
// Perform an online training for a specific trainDataset
void MultilayerPerceptron::trainOnline(Dataset *trainDataset)
{
	int i;
	for (i = 0; i < trainDataset->nOfPatterns; i++)
	{
		performEpochOnline(trainDataset->inputs[i], trainDataset->outputs[i]);
	}
}

// ------------------------------
// Test the network with a dataset and return the MSE
double MultilayerPerceptron::test(Dataset *testDataset)
{
	int i;
	double error = 0.0;
	for (i = 0; i < testDataset->nOfPatterns; i++)
	{
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		error += obtainError(testDataset->outputs[i]);
	}
	error = error / testDataset->nOfPatterns;
	return error;
}

// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(Dataset *pDatosTest)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers - 1].nOfNeurons;
	std::vector<double> obtained;
	obtained.resize(numSalidas);

	cout << "Id,Predicted" << endl;

	for (i = 0; i < pDatosTest->nOfPatterns; i++)
	{

		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(obtained);

		cout << i;

		for (j = 0; j < numSalidas; j++)
			cout << "," << obtained[j];
		cout << endl;
	}
}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MultilayerPerceptron::runOnlineBackPropagation(Dataset *trainDataset, Dataset *pDatosTest, int maxiter, double *errorTrain, double *errorTest,
													int &totalIterations)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();


	double minTrainError = 0;
	int iterWithoutImproving,valWithoutImproving;
	double testError = 0;

	double validationError = 1;

	// Generate validation data
	Dataset * validationDataset = new Dataset;
	if (validationRatio > 0 && validationRatio < 1)
	{
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
	do
	{

		trainOnline(trainDataset);
		double trainError = test(trainDataset);
		if (countTrain == 0 || trainError < minTrainError)
		{
			minTrainError = trainError;
			copyWeights();
			iterWithoutImproving = 0;
		}
		else if ((trainError - minTrainError) < 0.00001)
			iterWithoutImproving = 0;
		else
			iterWithoutImproving++;

		if (iterWithoutImproving == 50)
		{
			cout << "We exit because the training is not improving!!" << endl;
			restoreWeights();
			totalIterations += countTrain;
			break;
		}

		countTrain++;

		// Check validation stopping condition and force it
		// BE CAREFUL: in this case, we have to save the last validation error, not the minimum one
		// Apart from this, the way the stopping condition is checked is the same than that
		// applied for the training set
		if (validationRatio > 0 && validationRatio < 1){
			double lastValidationError = test(validationDataset);
			if (countTrain == 0 || lastValidationError < validationError)
			{
				validationError = lastValidationError;
				copyWeights();
				valWithoutImproving = 0;
			}
			else if ((trainError - minTrainError) < 0.00001)
				valWithoutImproving = 0;
			else
				valWithoutImproving++;

			if (valWithoutImproving == 50)
			{
				cout << "Validation has forced early stopping!!" << endl;
				restoreWeights();
				totalIterations += countTrain;
				break;
			}
		}

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << "\t Validation error: " << validationError << endl;

	} while (countTrain < maxiter);

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for (int i = 0; i < pDatosTest->nOfPatterns; i++)
	{
		std::vector<double> prediction;
		prediction.resize(pDatosTest->nOfOutputs);

		// Feed the inputs and propagate the values
		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for (int j = 0; j < pDatosTest->nOfOutputs; j++)
			cout << pDatosTest->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		prediction.clear();
	}

	testError = test(pDatosTest);
	*errorTest = testError;
	*errorTrain = minTrainError;
}

// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char *archivo)
{
	// Object for writing the file
	ofstream f(archivo);

	if (!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for (int i = 0; i < nOfLayers; i++)
		f << " " << layers[i].nOfNeurons;
	f << endl;

	// Write the weight matrix of every layer
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;
}

// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char *archivo)
{
	// Object for reading a file
	ifstream f(archivo);

	if (!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	std::vector<int> npl;

	// Read number of layers
	f >> nl;

	npl.resize(nl);

	// Read number of neurons in every layer
	for (int i = 0; i < nl; i++)
		f >> npl[i];

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				f >> layers[i].neurons[j].w[k];

	f.close();
	npl.clear();

	return true;
}
