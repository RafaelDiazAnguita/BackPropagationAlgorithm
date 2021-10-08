//============================================================================
// Introduction to computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // To obtain current time time()
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>

#include "imc/MultilayerPerceptron.h"


using namespace imc;
using namespace std;

int main(int argc, char **argv) {
    // Process arguments of the command line
    bool T_flag = 0, w_flag = 0, p_flag = 0,t_flag = 0, i_flag = 0,
            l_flag = 0, h_flag = 0, e_flag = 0, m_flag = 0,v_flag = 0,d_flag = 0;

    char *T_value = NULL, *w_value = NULL, *t_value = NULL, *i_value = NULL,
            *l_value = NULL, *h_value = NULL, *e_value = NULL, *m_value = NULL, *v_value = NULL, *d_value = NULL;
    int c;
    

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "T:w:p:t:i:l:h:e:m:v:d:")) != -1)
    {
        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch(c){
            case 't':
                t_flag = true;
                t_value = optarg;
                break;
            case 'i':
                i_flag = true;
                i_value = optarg;
                break;
            case 'l':
                l_flag = true;
                l_value = optarg;
                break;
            case 'h':
                h_flag = true;
                h_value = optarg;
                break;
            case 'e':
                e_flag = true;
                e_value = optarg;
                break;
            case 'm':
                m_flag = true;
                m_value = optarg;
                break;
            case 'v':
                v_flag = true;
                v_value = optarg;
                break;
            case 'd':
                d_flag = true;
                d_value = optarg;
                break;
            case 'T':
                T_flag = true;
                T_value = optarg;
                break;
            case 'w':
                w_flag = true;
                w_value = optarg;
                break;
            case 'p':
                p_flag = true;
                break;
            case '?':
                if (optopt == 'T' || optopt == 'w' || optopt == 'p' || optopt == 't' || optopt == 'i'
                    || optopt == 'l' || optopt == 'h' || optopt == 'e' || optopt == 'm' || optopt == 'v' || optopt == 'd')
                    fprintf (stderr, "The option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Unknown character `\\x%x'.\n",
                             optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }

    if (!p_flag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
    	MultilayerPerceptron mlp;

        // Parameters of the mlp.
        //iterations
        int iterations = 1000; 
        if( i_flag )    iterations = atoi(i_value); 
        //eta
        if( e_flag )    mlp.eta = atof( e_value );
        //mu
        if( m_flag )    mlp.mu = atof( m_value );
        //validationRatio
        if( v_flag )    mlp.validationRatio = atof( v_value );
        //decrementFactor
        if( d_flag )    mlp.decrementFactor = atof( d_value );



        // Read training and test data: call to mlp.readData(...)
    	Dataset * trainDataset = mlp.readData(t_value); 

        Dataset * testDataset;
        if ( T_flag )
    	    testDataset = mlp.readData( T_value ); 
        else 
            testDataset = trainDataset;

        // Initialize topology vector
        int layers = 1;
        if ( l_flag )
            layers = atoi( l_value );
    	
        std::vector<int> topology;
        topology.resize(layers+2);
        
        topology[0] = trainDataset->nOfInputs;
        topology[layers+1] = trainDataset->nOfOutputs; 

        if ( h_flag )
            for (size_t i = 1; i <= layers; i++)
                topology[i] = atoi(h_value);
        else
            for (size_t i = 1; i <= layers; i++)
                topology[i] = 5;

        // Initialize the network using the topology vector
        mlp.initialize(layers+2,topology);


        // Seed for random numbers
        int seeds[] = {1,2,3,4,5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
        double bestTestError = 1;
        int totalIterations = 0;
        
        for(int i=0; i<5; i++){
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
            mlp.runOnlineBackPropagation(trainDataset,testDataset,iterations,&(trainErrors[i]),&(testErrors[i]),totalIterations);
            cout << "We end!! => Final test error: " << testErrors[i] << endl;

            // We save the weights every time we find a better model
            if(w_flag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(w_value);
                bestTestError = testErrors[i];
            }
        }

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;
        
        // Obtain training and test averages and standard deviations
        for (size_t i = 0; i < 5; i++)
        {
            averageTrainError += trainErrors[i];
            averageTestError += testErrors[i];
        }
        averageTrainError /= 5;
        averageTestError /= 5;

        for (size_t i = 0; i < 5; i++)
        {
            stdTrainError += pow(fabs(trainErrors[i]-averageTrainError),2);
            stdTestError += pow(fabs(testErrors[i]-averageTestError),2);
        }
        stdTrainError = sqrt(stdTrainError/5);
        stdTestError = sqrt(stdTestError/5);
        
        
        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Total Iterations Mean: " << totalIterations << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error (Mean +- SD):  " << averageTestError << " +- " << stdTestError << endl;
        return EXIT_SUCCESS;
    }
    else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////
        
        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if(!w_flag || !mlp.readWeights(w_value))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to mlp.readData(...)
        Dataset *testDataset;
        testDataset = mlp.readData(T_value);
        if(testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;
    }

    
}

