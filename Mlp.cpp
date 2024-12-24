#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include "Mlp.h"

MLP::~MLP(){
    delete[] weights_input_hidden;
    delete[] weights_output_hidden;
    delete[] bias_hidden;
    delete[] bias_output;
}

MLP::MLP(uint inputSize, uint hiddenSize, uint outputSize)
: inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.01, 0.01);

    //Allocation des poids et des biais
    weights_input_hidden = new float[hiddenSize * inputSize];
    weights_output_hidden = new float[outputSize * hiddenSize];
    bias_hidden = new float[hiddenSize];
    bias_output = new float[outputSize];

    //Initialisation des poids et biais pour la couche cachée
    for(uint i=0; i< hiddenSize; i++){
        bias_hidden[i] = 0.0f;
        for(uint j=0;j<inputSize; j++){
            weights_input_hidden[i*inputSize +j] = dist(gen);
        }
    }

    //Initialisation des poids et biais pour la couche de sortie
    for(uint i=0; i< outputSize; i++){
        bias_output[i] = 0.0f;
        for(uint j=0;j<hiddenSize; j++){
            weights_output_hidden[i*hiddenSize +j] = dist(gen);
        }
    }
   
}

//Fonction d'activation ReLu
float MLP::relu(float x) const{
    return std::max(0.0f, x);
}


//Fonction d'activation softmax
std::vector<float> MLP::softmax(const std::vector<float>& z) const{
    float sum =0.0f;
    for(float val : z){
        sum += exp(val);
    }

    std::vector<float> proba;
    for(float val: z){
        proba.push_back(exp(val) / sum);
    }

    return proba;
}

//Propagation avant : forward propagation
std::vector<float> MLP::forward(const std::vector<float>& inputs)
{
    //couche cachée
    std::vector<float> hiddenL(hiddenSize, 0.0f);
    for(unsigned long i=0; i < hiddenSize; i++){
        float z = bias_hidden[i];
        for(unsigned long j=0; j < inputs.size(); j++){
            z += inputs[j] * weights_input_hidden[i * inputSize + j];
        }
        hiddenL[i] = relu(z);
    }

    //Couche sortie
    std::vector<float> outputL(outputSize, 0.0f);;
    for(unsigned long i =0; i < outputSize; i++){
        float z = bias_output[i];
        for(unsigned long j = 0; j < hiddenSize; j++){
            z += hiddenL[j] * weights_output_hidden[i * hiddenSize + j];
        }
        outputL[i] = z;
    }

    return softmax(outputL);
}